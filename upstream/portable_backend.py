"""CPU geometry for model-based inference; network tensors may live on MPS.

This renderer implements perspective-correct interpolation and a z-buffer.
It is inference-only, not a differentiable replacement for nvdiffrast.
"""
from stage_profile import stage, profiled
import os
import numpy as np
import torch
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
from numba import njit, prange
from kornia.geometry.conversions import axis_angle_to_rotation_matrix as so3_exp_map
from torch.nn.functional import normalize


def rotation_6d_to_matrix(d6):
    a, b = d6[..., :3], d6[..., 3:]
    x = normalize(a, dim=-1)
    y = normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-2)


from inference_device import DEVICE


def cluster_poses(angle_diff, dist_diff, poses, symmetry_tfs):
    """Greedy clustering with the same ordering/thresholds as mycpp."""
    kept = []
    threshold = np.deg2rad(angle_diff)
    for pose in poses:
        duplicate = False
        for other in kept:
            if np.linalg.norm(pose[:3, 3] - other[:3, 3]) >= dist_diff:
                continue
            rots = (pose[None] @ symmetry_tfs)[:, :3, :3]
            trace = np.einsum('sij,ij->s', rots, other[:3, :3])
            if np.any(np.arccos(np.clip((trace - 1) / 2, -1, 1)) < threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(pose)
    return np.asarray(kept)


@njit(cache=True, nogil=True, parallel=True)
def _filter(depth, radius, zfar, erosion, sigma_d, sigma_r, diff, ratio):
    h, w = depth.shape
    out = np.zeros_like(depth)
    for y in prange(h):
        for x in range(w):
            count, bad, total, mean = 0, 0, 0, 0.
            center = depth[y, x]
            for v in range(max(0, y-radius), min(h, y+radius+1)):
                for u in range(max(0, x-radius), min(w, x+radius+1)):
                    d = depth[v, u]
                    total += 1
                    if d < .001 or d >= zfar or abs(d-center) > diff:
                        bad += 1
                    if .001 <= d < zfar:
                        count += 1
                        mean += d
            if erosion:
                if bad / total <= ratio:
                    out[y, x] = center
                continue
            if count == 0:
                continue
            mean /= count
            acc, weights = 0., 0.
            for v in range(max(0, y-radius), min(h, y+radius+1)):
                for u in range(max(0, x-radius), min(w, x+radius+1)):
                    d = depth[v, u]
                    if .001 <= d < zfar and abs(d-mean) < .01:
                        weight = np.exp(-((u-x)**2 + (v-y)**2)/(2*sigma_d**2)
                                        - (center-d)**2/(2*sigma_r**2))
                        weights += weight
                        acc += weight*d
            if weights > 0:
                out[y, x] = acc/weights
    return out


def _filtered(depth, radius, zfar, erosion, sigma_d=2., sigma_r=100000., diff=.001, ratio=.8):
    tensor = torch.is_tensor(depth)
    arr = depth.detach().cpu().numpy() if tensor else np.asarray(depth)
    out = _filter(np.ascontiguousarray(arr, dtype=np.float32), radius, zfar,
                  erosion, sigma_d, sigma_r, diff, ratio)
    return torch.as_tensor(out, device=depth.device) if tensor else out


def erode_depth(depth, radius=2, depth_diff_thres=.001, ratio_thres=.8, zfar=100, device=None):
    return _filtered(depth, radius, zfar, True, diff=depth_diff_thres, ratio=ratio_thres)


def bilateral_filter_depth(depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device=None):
    return _filtered(depth, radius, zfar, False, sigma_d=sigmaD, sigma_r=sigmaR)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


@njit(cache=True, nogil=True)
def _rasterize(pos, faces, normals, uv, tex, colors, textured, K, pose, crop,
               height, width, use_light, light_dir, ambient, diffuse):
    pts = pos @ pose[:3, :3].T + pose[:3, 3]
    vn = normals @ pose[:3, :3].T
    screen = pts @ K.T
    screen = screen[:, :2] / screen[:, 2:3]
    screen[:, 0] = (screen[:, 0]-crop[0]) * width/(crop[2]-crop[0])
    screen[:, 1] = (screen[:, 1]-crop[1]) * height/(crop[3]-crop[1])
    depth = np.zeros((height, width), np.float32)
    rgb = np.zeros((height, width, 3), np.float32)
    xyz = np.zeros((height, width, 3), np.float32)
    normal = np.zeros((height, width, 3), np.float32)
    intensity = np.zeros(len(pos), np.float32)
    for i in range(len(pos)):
        n = np.sqrt((vn[i]**2).sum())
        if n > 0:
            intensity[i] = max(0., min(1., -(vn[i]*light_dir).sum()/n))
    for face in faces:
        a, b, c = face
        # Demo/normal tracking poses keep the complete object in front of near.
        # ponytail: skip near-plane crossing triangles; add clipping for such scenes.
        if min(pts[a, 2], pts[b, 2], pts[c, 2]) <= .001:
            continue
        ax, ay = screen[a]
        bx, by = screen[b]
        cx, cy = screen[c]
        den = (by-cy)*(ax-cx) + (cx-bx)*(ay-cy)
        if abs(den) < 1e-12:
            continue
        for y in range(max(0, int(np.floor(min(ay, by, cy)))), min(height, int(np.ceil(max(ay, by, cy)))+1)):
            for x in range(max(0, int(np.floor(min(ax, bx, cx)))), min(width, int(np.ceil(max(ax, bx, cx)))+1)):
                wa = ((by-cy)*(x+.5-cx)+(cx-bx)*(y+.5-cy))/den
                wb = ((cy-ay)*(x+.5-cx)+(ax-cx)*(y+.5-cy))/den
                wc = 1-wa-wb
                if min(wa, wb, wc) < -1e-7:
                    continue
                weights = np.array([wa/pts[a, 2], wb/pts[b, 2], wc/pts[c, 2]])
                z = 1/weights.sum()
                if z >= 100 or (depth[y, x] > 0 and z >= depth[y, x]):
                    continue
                weights *= z
                depth[y, x] = z
                p, n, color = np.zeros(3), np.zeros(3), np.zeros(3)
                shade, tu, tv = 0., 0., 0.
                for j in range(3):
                    idx = face[j]
                    p += weights[j]*pts[idx]
                    n += weights[j]*vn[idx]
                    shade += weights[j]*intensity[idx]
                    if textured:
                        tu += weights[j]*uv[idx, 0]
                        tv += weights[j]*uv[idx, 1]
                    else:
                        color += weights[j]*colors[idx]
                if textured:
                    # Match nvdiffrast linear filtering with wrap and flipped UV v.
                    tx, ty = tu*tex.shape[1]-.5, tv*tex.shape[0]-.5
                    ix, iy = int(np.floor(tx)), int(np.floor(ty))
                    fx, fy = tx-ix, ty-iy
                    for dy in range(2):
                        for dx in range(2):
                            weight = (fx if dx else 1-fx)*(fy if dy else 1-fy)
                            color += weight*tex[(iy+dy) % tex.shape[0], (ix+dx) % tex.shape[1]]
                if use_light:
                    color *= ambient+diffuse*shade
                rgb[y, x] = np.minimum(1., np.maximum(0., color))
                xyz[y, x] = p
                norm = np.sqrt((n*n).sum())
                if norm > 0:
                    normal[y, x] = n/norm
    return rgb, depth, normal, xyz


@profiled('render')
def render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context=None,
           get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None,
           bbox2d=None, output_size=None, use_light=False, light_color=None,
           light_dir=np.array([0., 0., 1.]), light_pos=None, w_ambient=.8,
           w_diffuse=.5, extra=None):
    if projection_mat is not None or light_color is not None or light_dir is None:
        raise NotImplementedError('Portable renderer supports intrinsics and default directional lighting only.')
    if mesh_tensors is None:
        pos, faces, normals = np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals)
        textured = getattr(mesh.visual, 'uv', None) is not None and getattr(mesh.visual.material, 'image', None) is not None if hasattr(mesh.visual, 'material') else False
        if textured:
            uv = np.array(mesh.visual.uv, copy=True)
            uv[:, 1] = 1-uv[:, 1]
            tex = np.asarray(mesh.visual.material.image.convert('RGB'), dtype=np.float32)/255
        else:
            uv, tex = np.zeros((len(pos), 2)), np.zeros((1, 1, 3))
        colors = np.asarray(mesh.visual.to_color().vertex_colors if textured else mesh.visual.vertex_colors)[:, :3]/255.
    else:
        pos, faces, normals = (_numpy(mesh_tensors[k]) for k in ('pos', 'faces', 'vnormals'))
        textured = 'tex' in mesh_tensors
        uv = _numpy(mesh_tensors['uv']) if textured else np.zeros((len(pos), 2))
        tex = _numpy(mesh_tensors['tex'])[0] if textured else np.zeros((1, 1, 3))
        colors = np.zeros((len(pos), 3)) if textured else _numpy(mesh_tensors['vertex_color'])
    with stage('poses_to_cpu'):
        poses = _numpy(ob_in_cams).reshape(-1, 4, 4)
        height, width = output_size if output_size is not None else (H, W)
        crops = np.tile([0., 0., W, H], (len(poses), 1)) if bbox2d is None else _numpy(bbox2d).reshape(-1, 4)
    if len(crops) != len(poses) or not np.isfinite(crops).all() or np.any(crops[:, 2:] <= crops[:, :2]):
        raise ValueError('Each pose requires a finite crop with positive width and height.')
    arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in (pos, normals, uv, tex, colors, K)]
    pos, normals, uv, tex, colors, K = arrays
    ld = np.asarray(light_dir, dtype=np.float32)
    ld /= max(float(np.linalg.norm(ld)), 1e-12)
    with stage('rasterize_cpu'):
        outputs = [_rasterize(pos, np.asarray(faces, dtype=np.int64), normals, uv, tex, colors, textured,
                            K, pose.astype(np.float32), crop.astype(np.float32), int(height), int(width),
                            use_light, ld, w_ambient, w_diffuse) for pose, crop in zip(poses, crops)]
    device = ob_in_cams.device if torch.is_tensor(ob_in_cams) else DEVICE
    with stage('outputs_to_device'):
        rgb, depth, normal, xyz = [torch.as_tensor(np.stack(items), device=device) for items in zip(*outputs)]
    if extra is not None:
        extra['xyz_map'] = xyz
    return rgb, depth, normal if get_normal or use_light else None
