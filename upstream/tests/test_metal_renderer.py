"""Run the renderer's geometry/appearance contract on actual Metal hardware."""
import os
import sys
import unittest
from unittest.mock import patch
import test_portable


@unittest.skipUnless(sys.platform == 'darwin', 'Metal requires macOS')
class MetalRendererTests(test_portable.PortableTests):
    def setUp(self):
        self.backend = patch.dict(os.environ, FOUNDATIONPOSE_RENDERER='metal')
        self.backend.start()
        self.addCleanup(self.backend.stop)

    def test_native_backend_is_available(self):
        from metal_renderer import library
        self.assertIsNotNone(library())

    def test_metal_rejects_cuda_before_loading_device(self):
        import subprocess
        from pathlib import Path
        result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'run_portable.py'),
                                 '--renderer', 'metal', '--device', 'cuda', '--check'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Metal renderer requires macOS', result.stderr)

    def test_batch_geometry_matches_cpu_away_from_edges(self):
        import numpy as np
        import torch
        import trimesh
        from portable_backend import render
        from scipy.ndimage import binary_erosion
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=.13)
        mesh.visual.vertex_colors = np.tile([195, 110, 45, 255], (len(mesh.vertices), 1))
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 19, axis=0)
        for i, pose in enumerate(poses):
            pose[:3, :3] = trimesh.transformations.rotation_matrix(i*.17, [1, 2, 3])[:3, :3]
            pose[:3, 3] = [0, 0, .4+i*.015]
        crops = np.tile([7., 6., 53., 54.], (len(poses), 1))
        args = dict(K=np.array([[80., 0, 30], [0, 80, 30], [0, 0, 1]], np.float32),
                    H=60, W=60, ob_in_cams=torch.tensor(poses), mesh=mesh, output_size=(48, 48),
                    bbox2d=crops, use_light=True)
        values = []
        for backend in ('numba', 'metal'):
            extra = {}
            with patch.dict(os.environ, FOUNDATIONPOSE_RENDERER=backend):
                rgb, depth, normal = render(**args, extra=extra)
            values.append(tuple(x.numpy() for x in (rgb, depth, normal, extra['xyz_map'])))
        a, b = values
        overlap = (a[1]>0)&(b[1]>0)
        union = (a[1]>0)|(b[1]>0)
        self.assertGreater(overlap.sum()/union.sum(), .995)
        interior = np.stack([binary_erosion(mask, iterations=1) for mask in overlap])
        # Hardware interpolation has finite subpixel precision. Geometry uses metre units.
        for i, tolerance in enumerate((3e-4, 1e-4, 6e-4, 1e-4)):
            np.testing.assert_allclose(a[i][interior], b[i][interior], atol=tolerance)

    def test_skip_near_plane_crossing_triangle(self):
        import numpy as np
        import torch
        import trimesh
        from portable_backend import render
        mesh = trimesh.Trimesh([[-.1, -.1, -.01], [.1, -.1, 1], [0, .1, 1]], [[0, 1, 2]], process=False)
        _, depth, _ = render(K=np.eye(3), H=20, W=20, ob_in_cams=torch.eye(4)[None], mesh=mesh)
        np.testing.assert_array_equal(depth.numpy(), 0)

    def test_native_boundary_rejects_invalid_face_shape(self):
        import numpy as np
        from metal_renderer import rasterize
        with self.assertRaisesRegex(ValueError, 'triangular'):
            rasterize(np.zeros((4, 3)), np.array([[0, 1, 2, 3]]), np.zeros((4, 3)),
                      np.zeros((4, 2)), np.zeros((1, 1, 3)), np.zeros((4, 3)), False,
                      np.eye(3), np.eye(4)[None], np.array([[0, 0, 20, 20]]),
                      20, 20, False, np.array([0, 0, 1]), .8, .5)

    def test_varying_wrapped_texture_matches_cpu(self):
        import numpy as np
        import torch
        import trimesh
        from PIL import Image
        from portable_backend import render
        from scipy.ndimage import binary_erosion
        mesh = trimesh.Trimesh([[-.7, -.7, 1], [1.4, -1.4, 2], [0, 2.1, 3]], [[0, 1, 2]], process=False)
        texture = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.array([[-.2, .1], [1.3, -.3], [.5, 1.2]]), image=Image.fromarray(texture))
        values = []
        for backend in ('numba', 'metal'):
            with patch.dict(os.environ, FOUNDATIONPOSE_RENDERER=backend):
                rgb, depth, _ = render(K=np.array([[20., 0, 20], [0, 20, 20], [0, 0, 1]]),
                                       H=40, W=40, ob_in_cams=torch.eye(4)[None], mesh=mesh)
            values.append((rgb.numpy(), depth.numpy()))
        interior = binary_erosion((values[0][1][0]>0)&(values[1][1][0]>0))
        np.testing.assert_allclose(values[0][0][0][interior], values[1][0][0][interior], atol=5e-4)
