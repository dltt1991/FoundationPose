import tempfile
import unittest
from pathlib import Path
import numpy as np
from run_portable import validate_frame, make_output_dir


class RunnerTests(unittest.TestCase):
    def test_stl_scale_and_explicit_white_material(self):
        import trimesh
        from run_portable import load_object_mesh
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'object.stl'
            trimesh.creation.box(extents=[180, 140, 38]).export(path)
            mesh = load_object_mesh(path, .001, [255, 255, 255])
            np.testing.assert_allclose(mesh.extents, [.18, .14, .038])
            self.assertTrue(np.all(mesh.visual.vertex_colors == [255, 255, 255, 255]))
            np.testing.assert_allclose(trimesh.load(path).extents, [180, 140, 38])

    def test_realsense_depth_rejects_saturation_and_out_of_range(self):
        from run_portable import realsense_depth_metres
        raw = np.array([[0, 500, 1000, 3000, 65535, 6000]], dtype=np.uint16)
        depth = realsense_depth_metres(raw, .001, .1, 3.)
        np.testing.assert_allclose(depth, [[0, .5, 1., 3., 0, 0]], atol=1e-6)
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_array_equal(raw, [[0, 500, 1000, 3000, 65535, 6000]])

    def test_rejects_invalid_depth_and_empty_initial_mask(self):
        rgb = np.zeros((10, 12, 3), np.uint8)
        depth = np.ones((10, 12), np.float32)
        k = np.array([[10, 0, 6], [0, 10, 5], [0, 0, 1]], np.float32)
        mask = np.ones(depth.shape, bool)
        validate_frame(rgb, depth, k, mask)
        with self.assertRaises(ValueError):
            validate_frame(rgb, depth, k, np.zeros_like(mask))
        with self.assertRaises(ValueError):
            validate_frame(rgb, np.zeros_like(depth), k)
        depth[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_frame(rgb, depth, k, mask)

    def test_output_directory_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run'
            make_output_dir(path)
            (path / 'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                make_output_dir(path)
            self.assertEqual((path / 'keep.txt').read_text(), 'keep')

    def test_modified_brown_rectification_matches_sdk_projection_formula(self):
        from run_portable import color_rectification_maps
        x, y = color_rectification_maps(np.eye(3), [.1, 0, .02, .03, 0], (4, 3), 'modified_brown_conrady')
        self.assertAlmostEqual(float(x[1, 1]), 1.404, places=5)
        self.assertAlmostEqual(float(y[1, 1]), 1.384, places=5)

    def test_rectification_agrees_with_installed_realsense_sdk(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            self.skipTest('RealSense SDK is optional')
        from run_portable import color_rectification_maps
        intr = rs.intrinsics()
        intr.width, intr.height = 16, 10
        intr.fx, intr.fy, intr.ppx, intr.ppy = 12., 11., 7., 4.
        intr.coeffs = [.1, -.01, .02, .03, .001]
        k = np.array([[12., 0, 7], [0, 11., 4], [0, 0, 1]])
        for model in (rs.distortion.none, rs.distortion.brown_conrady,
                      rs.distortion.modified_brown_conrady, rs.distortion.inverse_brown_conrady):
            intr.model = model
            x, y = color_rectification_maps(k, intr.coeffs, (16, 10), str(model).split('.')[-1])
            for u, v in [(0, 0), (7, 4), (15, 9)]:
                projected = rs.rs2_project_point_to_pixel(intr, [(u-7)/12, (v-4)/11, 1.])
                np.testing.assert_allclose([x[v, u], y[v, u]], projected, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
