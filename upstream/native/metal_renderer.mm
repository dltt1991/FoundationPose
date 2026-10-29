// Inference-only Metal rasterizer, called through a small C ABI.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <cstring>
#include <stdexcept>

static const char *shader = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params { float pose[16]; float K[9]; float crop[4]; float light[3]; float ambient; float diffuse; uint textured; uint lighting; };
struct Vertex { float4 position [[position]]; float3 xyz; float3 normal; float3 color; float2 uv; float shade; };
float3 transform(float3 x, constant Params& p, bool point) {
    return float3(dot(x,float3(p.pose[0],p.pose[1],p.pose[2])),
                  dot(x,float3(p.pose[4],p.pose[5],p.pose[6])),
                  dot(x,float3(p.pose[8],p.pose[9],p.pose[10])))
           + (point ? float3(p.pose[3],p.pose[7],p.pose[11]) : float3(0));
}
vertex Vertex vs(uint id [[vertex_id]], device const float* v [[buffer(0)]],
                 device const uint* faces [[buffer(1)]], constant Params& p [[buffer(2)]]) {
    uint idx=faces[id]; uint base=idx*11;
    float3 x=transform(float3(v[base],v[base+1],v[base+2]),p,true);
    float3 n=transform(float3(v[base+3],v[base+4],v[base+5]),p,false);
    float3 projected=float3(dot(x,float3(p.K[0],p.K[1],p.K[2])),
                            dot(x,float3(p.K[3],p.K[4],p.K[5])),
                            dot(x,float3(p.K[6],p.K[7],p.K[8])));
    float2 pixel=projected.xy/projected.z;
    float2 ndc=(pixel-float2(p.crop[0],p.crop[1])) / (float2(p.crop[2],p.crop[3])-float2(p.crop[0],p.crop[1]));
    Vertex out;
    out.position=float4((ndc.x*2-1)*x.z,(1-ndc.y*2)*x.z,(x.z-.001f)*(100.f/(100.f-.001f)),x.z);
    // Match the CPU renderer: skip whole triangles that cross the near plane.
    for(uint j=0;j<3;j++) {
        uint t=faces[(id/3)*3+j]*11;
        if(transform(float3(v[t],v[t+1],v[t+2]),p,true).z<=.001f) out.position=float4(2,2,2,1);
    }
    out.xyz=x; out.normal=n; out.color=float3(v[base+8],v[base+9],v[base+10]);
    out.uv=float2(v[base+6],v[base+7]);
    out.shade=length(n)>0 ? clamp(-dot(n,float3(p.light[0],p.light[1],p.light[2]))/length(n),0.f,1.f) : 0.f;
    return out;
}
struct Fragment { float4 color [[color(0)]]; float4 xyz [[color(1)]]; float4 normal [[color(2)]]; };
fragment Fragment fs(Vertex in [[stage_in]], constant Params& p [[buffer(2)]], texture2d<float> tex [[texture(0)]]) {
    float3 c=in.color;
    if(p.textured) {
        // Explicit FP32 bilinear filtering avoids the hardware sampler's weight quantization.
        int2 size=int2(tex.get_width(),tex.get_height());
        float2 pixel=in.uv*float2(size)-.5f;
        int2 base=int2(floor(pixel)); float2 f=pixel-float2(base);
        c=float3(0);
        for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++) {
            int2 coord=((base+int2(dx,dy))%size+size)%size;
            c+=tex.read(uint2(coord)).rgb*(dx?f.x:1-f.x)*(dy?f.y:1-f.y);
        }
    }
    if(p.lighting) c*=p.ambient+p.diffuse*in.shade;
    Fragment out; out.color=float4(clamp(c,0.f,1.f),1); out.xyz=float4(in.xyz,1);
    out.normal=float4(length(in.normal)>0 ? normalize(in.normal) : float3(0),1);
    return out;
}
)METAL";

struct Params { float pose[16], K[9], crop[4], light[3], ambient, diffuse; uint32_t textured, lighting; };
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLRenderPipelineState> pipeline;
static id<MTLDepthStencilState> depthState;

static void initialize() {
    if(pipeline && depthState && queue) return;
    device=MTLCreateSystemDefaultDevice();
    if(!device) throw std::runtime_error("No Metal device available");
    queue=[device newCommandQueue];
    if(!queue) throw std::runtime_error("Metal command queue creation failed");
    NSError *error=nil;
    MTLCompileOptions *options=[MTLCompileOptions new]; options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:shader] options:options error:&error];
    if(!lib) throw std::runtime_error([[error localizedDescription] UTF8String]);
    MTLRenderPipelineDescriptor *desc=[MTLRenderPipelineDescriptor new];
    desc.vertexFunction=[lib newFunctionWithName:@"vs"]; desc.fragmentFunction=[lib newFunctionWithName:@"fs"];
    for(int i=0;i<3;i++) desc.colorAttachments[i].pixelFormat=MTLPixelFormatRGBA32Float;
    desc.depthAttachmentPixelFormat=MTLPixelFormatDepth32Float;
    pipeline=[device newRenderPipelineStateWithDescriptor:desc error:&error];
    if(!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
    MTLDepthStencilDescriptor *dd=[MTLDepthStencilDescriptor new];
    dd.depthCompareFunction=MTLCompareFunctionLess; dd.depthWriteEnabled=YES;
    depthState=[device newDepthStencilStateWithDescriptor:dd];
    if(!depthState) throw std::runtime_error("Metal depth state creation failed");
}

extern "C" int fp_metal_render(const float *vertices, int nv, const uint32_t *faces, int nf,
    const float *texture, int th, int tw, const float *K, const float *poses, const float *crops,
    int count, int height, int width, int textured, int lighting, const float *light,
    float ambient, float diffuse, float *rgb, float *depth, float *normal, float *xyz,
    char *errorText, int errorSize) {
    @autoreleasepool {
        try {
            initialize();
            id<MTLBuffer> vb=[device newBufferWithBytes:vertices length:nv*11*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> ib=[device newBufferWithBytes:faces length:nf*3*sizeof(uint32_t) options:MTLResourceStorageModeShared];
            MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:tw height:th mipmapped:NO];
            td.storageMode=MTLStorageModeShared; td.usage=MTLTextureUsageShaderRead;
            id<MTLTexture> tex=[device newTextureWithDescriptor:td];
            if(!vb || !ib || !tex) throw std::runtime_error("Metal mesh/texture allocation failed");
            [tex replaceRegion:MTLRegionMake2D(0,0,tw,th) mipmapLevel:0 withBytes:texture bytesPerRow:tw*16];
            // ponytail: cap temporary render targets to 16 poses; no persistent mesh cache yet.
            const int chunk=16; const size_t pixels=(size_t)width*height;
            td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:width height:height mipmapped:NO];
            td.textureType=MTLTextureType2DArray; td.arrayLength=std::min(chunk,count);
            td.storageMode=MTLStorageModeShared; td.usage=MTLTextureUsageRenderTarget;
            id<MTLTexture> targets[3];
            for(int i=0;i<3;i++) { targets[i]=[device newTextureWithDescriptor:td]; if(!targets[i]) throw std::runtime_error("Metal target allocation failed"); }
            td.pixelFormat=MTLPixelFormatDepth32Float; td.storageMode=MTLStorageModePrivate;
            id<MTLTexture> zbuffer=[device newTextureWithDescriptor:td];
            if(!zbuffer) throw std::runtime_error("Metal depth allocation failed");
            std::vector<float> data(pixels*4);
            for(int start=0;start<count;start+=chunk) {
                int size=std::min(chunk,count-start);
                id<MTLCommandBuffer> cmd=[queue commandBuffer];
                if(!cmd) throw std::runtime_error("Metal command buffer creation failed");
                for(int i=0;i<size;i++) {
                    Params p; memcpy(p.pose,poses+(start+i)*16,64); memcpy(p.K,K,36);
                    memcpy(p.crop,crops+(start+i)*4,16); memcpy(p.light,light,12);
                    p.ambient=ambient; p.diffuse=diffuse; p.textured=textured; p.lighting=lighting;
                    MTLRenderPassDescriptor *pass=[MTLRenderPassDescriptor renderPassDescriptor];
                    for(int a=0;a<3;a++) {
                        pass.colorAttachments[a].texture=targets[a]; pass.colorAttachments[a].slice=i;
                        pass.colorAttachments[a].loadAction=MTLLoadActionClear; pass.colorAttachments[a].storeAction=MTLStoreActionStore;
                        pass.colorAttachments[a].clearColor=MTLClearColorMake(0,0,0,0);
                    }
                    pass.depthAttachment.texture=zbuffer; pass.depthAttachment.slice=i;
                    pass.depthAttachment.loadAction=MTLLoadActionClear; pass.depthAttachment.storeAction=MTLStoreActionDontCare;
                    pass.depthAttachment.clearDepth=1;
                    id<MTLRenderCommandEncoder> enc=[cmd renderCommandEncoderWithDescriptor:pass];
                    if(!enc) throw std::runtime_error("Metal render encoder creation failed");
                    [enc setRenderPipelineState:pipeline]; [enc setDepthStencilState:depthState]; [enc setCullMode:MTLCullModeNone];
                    [enc setVertexBuffer:vb offset:0 atIndex:0]; [enc setVertexBuffer:ib offset:0 atIndex:1];
                    [enc setVertexBytes:&p length:sizeof(p) atIndex:2]; [enc setFragmentBytes:&p length:sizeof(p) atIndex:2];
                    [enc setFragmentTexture:tex atIndex:0];
                    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:nf*3];
                    [enc endEncoding];
                }
                [cmd commit]; [cmd waitUntilCompleted];
                if(cmd.status!=MTLCommandBufferStatusCompleted) {
                    const char *message=cmd.error ? [[cmd.error localizedDescription] UTF8String] : "Metal command did not complete";
                    throw std::runtime_error(message);
                }
                for(int i=0;i<size;i++) for(int a=0;a<3;a++) {
                    [targets[a] getBytes:data.data() bytesPerRow:width*16 bytesPerImage:pixels*16 fromRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0 slice:i];
                    float *out=(a==0?rgb:a==1?xyz:normal)+(start+i)*pixels*3;
                    for(size_t j=0;j<pixels;j++) {
                        for(int k=0;k<3;k++) out[j*3+k]=data[j*4+k];
                        if(a==1) depth[(start+i)*pixels+j]=data[j*4+2];
                    }
                }
            }
            return 0;
        } catch(const std::exception& exc) {
            snprintf(errorText,errorSize,"%s",exc.what()); return 1;
        }
    }
}
