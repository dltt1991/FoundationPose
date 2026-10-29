import unittest
from portable_backend import cluster_poses, erode_depth, bilateral_filter_depth, render
import numpy as np
import torch
import trimesh


class PortableTests(unittest.TestCase):
    def test_initial_translation_is_float32_for_mps(self):
        from estimater import FoundationPose
        est = FoundationPose.__new__(FoundationPose)
        est.debug = 0
        center = est.guess_translation(np.ones((3, 3), np.float32), np.ones((3, 3), bool), np.eye(3))
        self.assertEqual(center.dtype, np.float32)

    def test_crop_accepts_numpy_float64_mesh_diameter_on_mps(self):
        from Utils import compute_crop_window_tf_batch
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        poses = torch.eye(4, device=device)[None]
        poses[:, 2, 3] = 1
        tf = compute_crop_window_tf_batch(poses=poses, K=np.diag([100., 100., 1.]), mesh_diameter=np.float64(.2),
                                          out_size=(160, 160), method='box_3d')
        self.assertEqual(tf.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tf).all())

    def test_cluster_respects_symmetry_and_translation(self):
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        poses[1, :3, :3] = np.diag([-1, -1, 1])
        poses[2, 0, 3] = 1
        poses[3, :3, :3] = trimesh.transformations.rotation_matrix(.7, [1, 0, 0])[:3, :3]
        sym = np.stack([np.eye(4), poses[1]])
        self.assertEqual(len(cluster_poses(30, .1, poses, sym)), 3)

    def test_depth_filter_preserves_plane_and_removes_isolated_outlier(self):
        depth = np.ones((7, 7), np.float32)
        np.testing.assert_allclose(erode_depth(depth), depth)
        np.testing.assert_allclose(bilateral_filter_depth(depth), depth)
        depth[3, 3] = 2
        self.assertEqual(erode_depth(depth)[3, 3], 0)
        depth[:] = 0
        np.testing.assert_array_equal(bilateral_filter_depth(depth), depth)

    def render_triangle(self, poses, bbox=None):
        # Two triangles: a red foreground at z=1 and a blue background at z=2.
        pos = np.array([[-.8, -.8, 1], [.8, -.8, 1], [0, .8, 1],
                        [-1.6, -1.6, 2], [1.6, -1.6, 2], [0, 1.6, 2]], np.float32)
        mesh = trimesh.Trimesh(pos, [[0, 1, 2], [3, 4, 5]], process=False)
        mesh.visual.vertex_colors = [[255, 0, 0, 255]] * 3 + [[0, 0, 255, 255]] * 3
        extra = {}
        rgb, depth, _ = render(K=np.array([[10, 0, 10], [0, 10, 10], [0, 0, 1]]),
                               H=20, W=20, ob_in_cams=torch.tensor(poses, dtype=torch.float32),
                               mesh=mesh, output_size=(20, 20), bbox2d=bbox, extra=extra)
        return rgb.numpy(), depth.numpy(), extra['xyz_map'].numpy()

    def test_renderer_zbuffer_xyz_and_pose(self):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, 2, 3] = 1
        rgb, depth, xyz = self.render_triangle(poses)
        self.assertAlmostEqual(depth[0, 10, 10], 1)
        self.assertAlmostEqual(depth[1, 10, 10], 2)
        np.testing.assert_allclose(rgb[0, 10, 10], [1, 0, 0])
        np.testing.assert_allclose(xyz[0, 10, 10], [.05, .05, 1], atol=1e-6)
        self.assertEqual(depth[0, 0, 0], 0)

    def test_renderer_crop_matches_ray_geometry(self):
        _, depth, xyz = self.render_triangle(np.eye(4)[None], np.array([[5, 5, 15, 15]]))
        self.assertEqual(depth[0, 10, 10], 1)
        np.testing.assert_allclose(xyz[0, 10, 10], [.025, .025, 1], atol=1e-6)

    def test_renderer_uses_perspective_correct_depth(self):
        mesh = trimesh.Trimesh([[-.8, -.8, 1], [1.6, -1.6, 2], [0, 2.4, 3]], [[0, 1, 2]], process=False)
        extra = {}
        _, depth, _ = render(K=np.array([[10, 0, 10], [0, 10, 10], [0, 0, 1]]),
                             H=20, W=20, ob_in_cams=torch.eye(4)[None], mesh=mesh, extra=extra)
        weights = np.linalg.solve(np.array([[2, 18, 10], [2, 2, 18], [1, 1, 1]]), [10.5, 10.5, 1])
        expected = 1 / np.dot(weights, [1, .5, 1/3])
        self.assertAlmostEqual(float(depth[0, 10, 10]), expected, places=5)
        np.testing.assert_allclose(extra['xyz_map'][0, 10, 10].numpy(), [.05*expected, .05*expected, expected], atol=1e-6)

    def test_texture_v_orientation_and_directional_light(self):
        from PIL import Image
        mesh = trimesh.Trimesh([[-.8, -.8, 1], [.8, -.8, 1], [0, .8, 1]], [[0, 2, 1]], process=False)
        texture = np.array([[[255, 0, 0], [255, 0, 0]], [[0, 0, 255], [0, 0, 255]]], np.uint8)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.tile([.5, .75], (3, 1)), image=Image.fromarray(texture))
        rgb, _, normals = render(K=np.array([[10, 0, 10], [0, 10, 10], [0, 0, 1]]),
                                 H=20, W=20, ob_in_cams=torch.eye(4)[None], mesh=mesh, use_light=True)
        np.testing.assert_allclose(rgb[0, 10, 10].numpy(), [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(normals[0, 10, 10].numpy(), [0, 0, -1], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
