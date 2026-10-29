"""Optional macOS Metal rasterizer; compile locally with Xcode Command Line Tools."""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading

import numpy as np

_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def library():
    if sys.platform != 'darwin':
        raise RuntimeError('Metal renderer requires macOS.')
    if int(platform.mac_ver()[0].split('.')[0]) < 15:
        raise RuntimeError('Metal renderer requires macOS 15 or newer.')
    source = Path(__file__).resolve().parent / 'native/metal_renderer.mm'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    output = source.parent / 'build' / f'metal-{digest}.dylib'
    if not output.exists():
        output.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as directory:
            temporary = Path(directory) / 'renderer.dylib'
            subprocess.run(['xcrun', 'clang++', '-std=c++17', '-mmacosx-version-min=15.0', '-O3', '-fobjc-arc', '-dynamiclib',
                            str(source), '-framework', 'Foundation', '-framework', 'Metal', '-o', str(temporary)], check=True)
            temporary.replace(output)
    lib = ctypes.CDLL(str(output))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.fp_metal_render.argtypes = [ptr, integer, ptr, integer, ptr, integer, integer,
                                   ptr, ptr, ptr, integer, integer, integer, integer, integer,
                                   ptr, ctypes.c_float, ctypes.c_float, ptr, ptr, ptr, ptr, ptr, integer]
    lib.fp_metal_render.restype = integer
    return lib


def rasterize(pos, faces, normals, uv, tex, colors, textured, K, poses, crops,
              height, width, use_light, light_dir, ambient, diffuse):
    if np.ndim(faces) != 2 or np.shape(faces)[1] != 3:
        raise ValueError('Metal requires triangular faces with shape F×3.')
    if np.shape(K) != (3, 3) or np.ndim(poses) != 3 or np.shape(poses)[1:] != (4, 4):
        raise ValueError('Metal requires K 3×3 and poses N×4×4.')
    if np.shape(crops) != (len(poses), 4) or not 0 < height <= 16384 or not 0 < width <= 16384:
        raise ValueError('Metal requires matching crops and output dimensions in 1–16384.')
    if np.ndim(tex) != 3 or np.shape(tex)[2] != 3 or min(np.shape(tex)[:2]) < 1:
        raise ValueError('Metal requires a nonempty H×W×3 texture.')
    if np.shape(light_dir) != (3,) or any(not np.isfinite(a).all() for a in (pos, normals, uv, tex, colors, K, poses, crops, light_dir)):
        raise ValueError('Metal renderer inputs must be finite and light direction must be 3D.')
    if np.ndim(pos) != 2 or np.shape(pos)[1] != 3 or any(np.shape(a) != np.shape(pos) for a in (normals, colors)) or np.shape(uv) != (len(pos), 2):
        raise ValueError('Metal requires matching N×3 vertices/normals/colors and N×2 UVs.')
    if not np.isfinite([ambient, diffuse]).all():
        raise ValueError('Lighting coefficients must be finite.')
    vertices = np.ascontiguousarray(np.concatenate((pos, normals, uv, colors), axis=1), dtype=np.float32)
    faces = np.asarray(faces)
    if faces.size == 0 or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices) or not len(poses):
        raise ValueError('Metal requires nonempty valid mesh indices and poses.')
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    tex = np.ascontiguousarray(np.concatenate((tex, np.ones((*tex.shape[:2], 1), np.float32)), axis=2), dtype=np.float32)
    K, poses, crops, light_dir = [np.ascontiguousarray(a, dtype=np.float32) for a in (K, poses, crops, light_dir)]
    shape = (len(poses), int(height), int(width))
    rgb, depth, normal, xyz = [np.empty(s, np.float32) for s in (shape+(3,), shape, shape+(3,), shape+(3,))]
    error = ctypes.create_string_buffer(2048)
    def address(a):
        return a.ctypes.data
    # Native pipeline/queue creation and use are serialized across Python callers.
    with _LOCK:
        code = library().fp_metal_render(address(vertices), len(vertices), address(faces), len(faces),
            address(tex), tex.shape[0], tex.shape[1], address(K), address(poses), address(crops),
            len(poses), height, width, int(textured), int(use_light), address(light_dir), ambient, diffuse,
            address(rgb), address(depth), address(normal), address(xyz), error, len(error))
    if code:
        raise RuntimeError('Metal renderer: '+error.value.decode())
    return rgb, depth, normal, xyz
