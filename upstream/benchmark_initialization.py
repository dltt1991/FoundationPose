"""Repeat first-frame registration on saved RGB-D, without camera permissions."""
import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_input(directory=None):
    from run_portable import demo_frames, validate_frame
    if directory is None:
        _, rgb, depth, k, mask = next(demo_frames(ROOT / 'demo_data/mustard0'))
        config = dict(mesh=str(ROOT / 'demo_data/mustard0/mesh/textured_simple.obj'), mesh_scale=1., mesh_color=None)
    else:
        config = json.loads((directory / 'run.json').read_text())
        bgr = cv2.imread(str(directory / 'initial_rgb.png'))
        raw_mask = cv2.imread(str(directory / 'initial_mask.png'), cv2.IMREAD_GRAYSCALE)
        if bgr is None or raw_mask is None:
            raise FileNotFoundError('Input directory must contain initial_rgb.png and initial_mask.png.')
        rgb, mask = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), raw_mask > 0
        depth = np.load(directory / 'initial_depth_m.npy', allow_pickle=False)
        k = np.loadtxt(directory / 'cam_K.txt').astype(np.float32)
    validate_frame(rgb, depth, k, mask)
    return rgb, depth, k, mask, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Saved run directory; default is official mustard0 first frame.')
    parser.add_argument('--output', type=Path, required=True, help='A directory that must not already exist.')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--refiner-backend', choices=['torch', 'coreml'], default='torch')
    parser.add_argument('--coreml-models', type=Path, default=ROOT / 'weights/coreml-refiner-fp32')
    parser.add_argument('--coreml-compute-units', choices=['ALL', 'CPU_AND_GPU', 'CPU_AND_NE', 'CPU_ONLY'], default='CPU_AND_GPU')
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--profile', action='store_true', help='One additional instrumented registration; separate from speed results.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--track-steps', type=int, default=0, help='Optional static-input tracking checks after registration; not a live-camera FPS test.')
    args = parser.parse_args()
    if min(args.runs, args.batch_size) < 1 or args.track_steps < 0:
        parser.error('Runs and batch size must be positive; track steps must be nonnegative.')
    os.environ['FOUNDATIONPOSE_DEVICE'] = args.device
    os.environ['FOUNDATIONPOSE_BATCH_SIZE'] = str(args.batch_size)
    os.environ['FOUNDATIONPOSE_REFINER_BACKEND'] = args.refiner_backend
    os.environ['FOUNDATIONPOSE_COREML_MODELS'] = str(args.coreml_models.resolve())
    os.environ['FOUNDATIONPOSE_COREML_COMPUTE_UNITS'] = args.coreml_compute_units
    from stage_profile import StageProfile, device_synchronize, stage
    from run_portable import load_object_mesh, make_output_dir
    from estimater import FoundationPose
    from Utils import set_seed
    import torch
    logging.getLogger().setLevel(logging.WARNING)
    rgb, depth, k, mask, config = load_input(args.input)
    out = make_output_dir(args.output)
    set_seed(0)
    mesh = load_object_mesh(config['mesh'], config.get('mesh_scale', 1.), config.get('mesh_color'))
    setup = StageProfile(synchronize=lambda: device_synchronize(args.device))
    with setup, stage('setup'):
        est = FoundationPose(mesh.vertices, mesh.vertex_normals, mesh=mesh, debug=0, debug_dir=str(out))
    setup.save(out / 'setup_profile.json')
    result = dict(device=args.device, renderer='numba', refiner_backend=args.refiner_backend,
                  coreml_compute_units=args.coreml_compute_units if args.refiner_backend == 'coreml' else None,
                  torch=torch.__version__,
                  input=str(args.input) if args.input else 'official mustard0 first frame',
                  mesh=config['mesh'], mesh_scale=config.get('mesh_scale', 1.),
                  mesh_color=config.get('mesh_color'), input_shape=list(rgb.shape),
                  mesh_faces=len(mesh.faces), hypotheses=len(est.rot_grid),
                  estimation_iterations=5, batch_size=args.batch_size, runs=[])
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    for index in range(args.runs+int(args.profile)):
        instrumented = index == args.runs
        profiler = StageProfile(synchronize=lambda: device_synchronize(args.device)) if instrumented else None
        device_synchronize(args.device)
        started = time.perf_counter()
        with profiler if profiler is not None else nullcontext():
            pose = est.register(K=k, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
        device_synchronize(args.device)
        seconds = time.perf_counter()-started
        if not np.isfinite(pose).all():
            raise RuntimeError('Non-finite predicted pose')
        entry = dict(index=index, instrumented=instrumented, seconds=seconds, pose=pose.tolist(),
                     selected_candidate=int(est.best_id.cpu()),
                     max_score=float(est.scores.max().cpu()))
        result['runs'].append(entry)
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        if profiler is not None:
            profiler.save(out / 'registration_profile.json')
        print(f'{args.refiner_backend} run={index} instrumented={instrumented} {seconds:.3f}s candidate={entry["selected_candidate"]}', flush=True)
    if args.track_steps:
        timings = []
        for _ in range(args.track_steps):
            device_synchronize(args.device)
            start = time.perf_counter()
            pose = est.track_one(rgb=rgb, depth=depth, K=k, iteration=2)
            device_synchronize(args.device)
            timings.append(time.perf_counter()-start)
            if not np.isfinite(pose).all():
                raise RuntimeError('Non-finite tracked pose')
        result['tracking'] = dict(protocol='Repeated unchanged snapshot, two refinement iterations; not live-camera FPS.',
                                  seconds=timings, final_pose=pose.tolist())
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        print(f'Static-input tracking median={np.median(timings):.4f}s', flush=True)
    print(f'Results: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
