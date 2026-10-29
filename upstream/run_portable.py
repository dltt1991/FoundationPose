"""FoundationPose model-based inference on MPS/CPU, with optional RealSense RGB-D."""
import argparse
from contextlib import nullcontext
import csv
import json
import logging
import os
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_object_mesh(path, scale, color=None):
    import trimesh
    mesh = trimesh.load(path, force='mesh')
    mesh.apply_scale(scale)
    if color is not None:
        if len(color) != 3 or any(c < 0 or c > 255 for c in color):
            raise ValueError('Mesh color requires three RGB values in 0–255.')
        rgba = np.tile([*color, 255], (len(mesh.vertices), 1)).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    return mesh


def realsense_depth_metres(raw_depth, scale, min_depth=.1, max_depth=3.):
    """Convert Z16; reject zero, saturated values and the configured work range."""
    if not np.isfinite(scale) or scale <= 0 or not 0 <= min_depth < max_depth or not np.isfinite(max_depth):
        raise ValueError('Depth scale and working range must be finite and positive.')
    depth = raw_depth.astype(np.float32)*scale
    # Allow float32 conversion roundoff at inclusive working-range limits.
    epsilon = 4*np.finfo(np.float32).eps*max_depth
    valid = (raw_depth > 0) & (raw_depth < 65535) & np.isfinite(depth) & (depth >= min_depth-epsilon) & (depth <= max_depth+epsilon)
    np.clip(depth, min_depth, max_depth, out=depth)
    depth[~valid] = 0
    return depth


def validate_frame(rgb, depth, k, mask=None):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError('RGB and aligned depth must have matching H×W dimensions.')
    if rgb.dtype != np.uint8 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError('RGB must be uint8; depth must be floating point in metres.')
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError('Depth must be finite and nonnegative; invalid pixels must be zero.')
    if np.count_nonzero((depth >= .001) & (depth < 100)) < 4:
        raise ValueError('Frame has fewer than four valid depth pixels; stop and reinitialize.')
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Camera intrinsics must be a finite 3×3 matrix with positive focal lengths.')
    if mask is not None:
        if mask.shape != depth.shape or np.count_nonzero((mask > 0) & (depth >= .001) & (depth < 100)) < 4:
            raise ValueError('Initial object mask must match the image and contain at least four valid depth pixels.')


def make_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for name in ('ob_in_cam', 'track_vis'):
        (path / name).mkdir()
    return path


def color_rectification_maps(k, coeffs, size, model):
    """Map pinhole output pixels to color pixels using SDK forward distortion.

    Modified/Inverse Brown follow rs2_project_point_to_pixel in librealsense
    src/rs.cpp; their tangential terms differ from standard OpenCV Brown.
    """
    if model == 'brown_conrady':
        return cv2.initUndistortRectifyMap(k, np.asarray(coeffs), None, k, size, cv2.CV_32FC1)
    u, v = np.meshgrid(np.arange(size[0], dtype=np.float32), np.arange(size[1], dtype=np.float32))
    if model == 'none':
        return u, v
    if model not in ('modified_brown_conrady', 'inverse_brown_conrady'):
        raise ValueError(f'Unsupported color distortion model: {model}')
    x, y = (u-k[0, 2])/k[0, 0], (v-k[1, 2])/k[1, 1]
    r2 = x*x+y*y
    a, b, p1, p2, c = coeffs
    factor = 1+a*r2+b*r2*r2+c*r2*r2*r2
    x, y = x*factor, y*factor
    dx = x+2*p1*x*y+p2*(r2+2*x*x)
    dy = y+2*p2*x*y+p1*(r2+2*y*y)
    return (dx*k[0, 0]+k[0, 2]).astype(np.float32), (dy*k[1, 1]+k[1, 2]).astype(np.float32)


def demo_frames(scene):
    k = np.loadtxt(scene / 'cam_K.txt').astype(np.float32).reshape(3, 3)
    files = sorted((scene / 'rgb').glob('*.png'))
    if not files:
        raise FileNotFoundError(f'No RGB frames found in {scene / "rgb"}')
    for i, file in enumerate(files):
        bgr = cv2.imread(str(file), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(scene / 'depth' / file.name), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None:
            raise FileNotFoundError(f'Missing/unreadable RGB or depth: {file.name}')
        mask = None
        if i == 0:
            mask = cv2.imread(str(scene / 'masks' / file.name), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f'Missing first-frame mask: {file.name}')
            if mask.ndim == 3:
                mask = mask.max(axis=2)
            mask = mask > 0
        yield file.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), raw.astype(np.float32) / 1000., k, mask


def select_mask(rgb):
    """Freeze first frame and trace object outline; no segmentation model required."""
    points = []
    title = 'Object mask: click outline; Enter accept; R reset; Q cancel'
    cv2.namedWindow(title)

    def click(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(title, click)
    try:
        while True:
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if points:
                cv2.polylines(image, [np.array(points, np.int32)], len(points) >= 3, (0, 255, 0), 2)
            cv2.imshow(title, image)
            key = cv2.waitKey(30) & 255
            if key in (13, 32) and len(points) >= 3:
                mask = np.zeros(rgb.shape[:2], np.uint8)
                cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
                return mask.astype(bool)
            if key == ord('r'):
                points.clear()
            if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt
    finally:
        cv2.destroyWindow(title)


def realsense_frames(args):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError('pyrealsense2 is not installed for this Python. See README_MAC.md for macOS SDK build instructions.') from exc
    pipeline, cfg = rs.pipeline(), rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    started = False
    try:
        profile = pipeline.start(cfg)
        started = True
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float32)
        model = str(intr.model).split('.')[-1]
        map_x, map_y = color_rectification_maps(k, intr.coeffs, (intr.width, intr.height), model)
        for _ in range(30):
            pipeline.wait_for_frames(10000)
        i = 0
        while True:
            frames = align.process(pipeline.wait_for_frames(10000))
            color, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth_frame:
                continue
            rgb = np.asanyarray(color.get_data()).copy()
            depth = realsense_depth_metres(np.asanyarray(depth_frame.get_data()), scale, args.min_depth, args.max_depth)
            rgb = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)
            # The exact frozen first RGB-D pair is used for mask creation and registration.
            yield f'{i:06d}', rgb, depth, k, None
            i += 1
    finally:
        if started:
            pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['demo', 'realsense'], default='demo')
    parser.add_argument('--scene', type=Path, default=ROOT / 'demo_data/mustard0')
    parser.add_argument('--mesh', type=Path, help='Required for your own RealSense object; mesh dimensions in metres.')
    parser.add_argument('--mesh-scale', type=float, default=1., help='Use 0.001 for a mesh authored in millimetres.')
    parser.add_argument('--mesh-color', type=int, nargs=3, metavar=('R', 'G', 'B'), help='Override mesh appearance with an RGB color in 0–255; useful for untextured STL.')
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--refiner-backend', choices=['torch', 'coreml'], default='torch', help='Optional native Core ML refiner prototype; scoring remains PyTorch.')
    parser.add_argument('--coreml-models', type=Path, default=ROOT / 'weights/coreml-refiner-fp32')
    parser.add_argument('--coreml-compute-units', choices=['ALL', 'CPU_AND_GPU', 'CPU_AND_NE', 'CPU_ONLY'], default='CPU_AND_GPU')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=3, help='0 runs all frames; default is a short validation.')
    parser.add_argument('--est-iters', type=int, default=5)
    parser.add_argument('--track-iters', type=int, default=2)
    parser.add_argument('--output', type=Path, help='Must not already exist.')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--pipeline', action='store_true', help='Replay demo data through the live capture/inference pipeline; always enabled for RealSense.')
    parser.add_argument('--target-fps', type=float, default=10., help='Display refresh target and measured pose throughput target.')
    parser.add_argument('--capture-fps', type=float, default=0., help='0 preserves demo nanosecond timestamps; positive values override replay rate. RealSense uses 30Hz.')
    parser.add_argument('--save-frames', action='store_true', help='Also save live/replay overlay images; can reduce display throughput.')
    parser.add_argument('--profile-init', action='store_true', help='Write synchronized setup/first-frame stage profiles; disabled for normal speed measurements.')
    parser.add_argument('--serial', help='Optional RealSense serial number.')
    parser.add_argument('--min-depth', type=float, default=.1, help='RealSense minimum accepted depth in metres.')
    parser.add_argument('--max-depth', type=float, default=3., help='RealSense maximum accepted depth in metres.')
    parser.add_argument('--check', action='store_true', help='Check weights, inputs and selected torch backend without inference.')
    args = parser.parse_args()
    if args.refiner_backend == 'coreml' and args.device == 'cuda':
        parser.error('Core ML requires cpu or mps on macOS; CUDA is unsupported.')
    if args.mesh_color is not None and any(c < 0 or c > 255 for c in args.mesh_color):
        parser.error('--mesh-color requires three integers in 0–255.')
    if not np.isfinite(args.min_depth) or not np.isfinite(args.max_depth) or not 0 <= args.min_depth < args.max_depth:
        parser.error('Depth limits must be finite and satisfy 0 <= min-depth < max-depth.')
    if not np.isfinite(args.target_fps) or not np.isfinite(args.capture_fps) or args.target_fps <= 0 or args.capture_fps < 0:
        parser.error('Target FPS must be positive and replay FPS nonnegative; both must be finite.')
    if args.batch_size < 1 or min(args.est_iters, args.track_iters) < 1 or args.max_frames < 0 or not np.isfinite(args.mesh_scale) or args.mesh_scale <= 0:
        parser.error('Batch size, iteration counts and mesh scale must be positive; max-frames must be nonnegative.')
    if args.source == 'realsense' and args.mesh is None:
        parser.error('--mesh is required for RealSense; use the model of the physical object you will track.')
    os.environ['FOUNDATIONPOSE_DEVICE'] = args.device
    os.environ['FOUNDATIONPOSE_BATCH_SIZE'] = str(args.batch_size)
    os.environ['FOUNDATIONPOSE_REFINER_BACKEND'] = args.refiner_backend
    os.environ['FOUNDATIONPOSE_COREML_MODELS'] = str(args.coreml_models.resolve())
    os.environ['FOUNDATIONPOSE_COREML_COMPUTE_UNITS'] = args.coreml_compute_units
    from inference_device import DEVICE
    import torch
    mesh_path = args.mesh or args.scene / 'mesh/textured_simple.obj'
    required = [mesh_path]
    for stamp in ('2023-10-28-18-33-37', '2024-01-11-20-02-45'):
        required += [ROOT / 'weights' / stamp / name for name in ('config.yml', 'model_best.pth')]
    if args.refiner_backend == 'coreml':
        required.append(args.coreml_models / 'manifest.json')
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error('Missing inputs:\n' + '\n'.join(missing))
    if args.source == 'demo':
        first = next(demo_frames(args.scene))
        validate_frame(*first[1:])
    else:
        import importlib.util
        if importlib.util.find_spec('pyrealsense2') is None:
            parser.error('pyrealsense2 is unavailable. See README_MAC.md.')
    print(f'device={DEVICE}; torch={torch.__version__}; mesh={mesh_path}', flush=True)
    if args.check:
        return
    from estimater import FoundationPose
    from Utils import draw_posed_3d_box, draw_xyz_axis, set_seed
    import trimesh
    logging.getLogger().setLevel(logging.WARNING)
    set_seed(0)
    out = make_output_dir(args.output or ROOT / 'debug' / f'portable-{time.time_ns()}')
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(device=DEVICE, torch=torch.__version__, mesh=str(mesh_path), pose_convention='object_to_camera; OpenCV axes; translation in metres')
    (out / 'run.json').write_text(json.dumps(metadata, indent=2))
    mesh = load_object_mesh(mesh_path, args.mesh_scale, args.mesh_color)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2])
    started = time.perf_counter()
    from stage_profile import StageProfile, device_synchronize
    setup_profile = StageProfile(synchronize=lambda: device_synchronize(DEVICE)) if args.profile_init else None
    with setup_profile or nullcontext():
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                             mesh=mesh, debug=0, debug_dir=str(out))
    if setup_profile is not None:
        setup_profile.save(out / 'setup_profile.json')
    print(f'Initialization: {time.perf_counter()-started:.2f}s; {len(est.rot_grid)} hypotheses; output={out}', flush=True)
    frames = demo_frames(args.scene) if args.source == 'demo' else realsense_frames(args)
    if args.source == 'realsense' or args.pipeline:
        from live_runner import run_live
        run_live(args, frames, est, out, to_origin, bbox)
        return
    processed = 0
    try:
        with (out / 'timings.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['frame', 'stage', 'inference_seconds'])
            writer.writeheader()
            for i, (frame_id, rgb, depth, k, mask) in enumerate(frames):
                validate_frame(rgb, depth, k, mask)
                if i == 0:
                    cv2.imwrite(str(out / 'initial_rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(out / 'initial_mask.png'), mask.astype(np.uint8)*255)
                    np.save(out / 'initial_depth_m.npy', depth)
                    np.savetxt(out / 'cam_K.txt', k)
                start = time.perf_counter()
                if i == 0:
                    profiler = StageProfile(synchronize=lambda: device_synchronize(DEVICE)) if args.profile_init else None
                    with profiler or nullcontext():
                        pose = est.register(K=k, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.est_iters)
                    if profiler is not None:
                        profiler.save(out / 'registration_profile.json')
                else:
                    pose = est.track_one(K=k, rgb=rgb, depth=depth, iteration=args.track_iters)
                if DEVICE == 'mps':
                    torch.mps.synchronize()
                elif DEVICE == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                if not np.isfinite(pose).all() or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=.01):
                    raise RuntimeError(f'Invalid predicted pose in frame {frame_id}')
                np.savetxt(out / 'ob_in_cam' / f'{frame_id}.txt', pose)
                vis = draw_posed_3d_box(k, img=rgb, ob_in_cam=pose @ np.linalg.inv(to_origin), bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=.05, K=k, is_input_rgb=True)
                bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out / 'track_vis' / f'{frame_id}.png'), bgr)
                stage = 'register' if i == 0 else 'track'
                writer.writerow(dict(frame=frame_id, stage=stage, inference_seconds=f'{elapsed:.6f}'))
                stream.flush()
                print(f'{frame_id}: {stage} {elapsed:.3f}s ({1/elapsed:.2f} inference FPS)', flush=True)
                processed += 1
                if args.display:
                    cv2.imshow('FoundationPose portable', bgr)
                    if cv2.waitKey(1) & 255 in (27, ord('q')):
                        break
                if args.max_frames and processed >= args.max_frames:
                    break
    finally:
        frames.close()
        cv2.destroyAllWindows()
    print(f'Saved {processed} frames to {out}', flush=True)


if __name__ == '__main__':
    main()
