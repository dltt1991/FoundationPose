"""Pipelined RGB-D inference and main-thread OpenCV display."""
from collections import deque
from contextlib import nullcontext
import csv
import json
import threading
import time

import cv2
import numpy as np
import torch

from inference_device import DEVICE
from live_pipeline import LivePipeline


def replay_offset(frame_id, first_id, index, capture_fps):
    if capture_fps > 0:
        return index/capture_fps
    try:
        offset = (int(frame_id)-int(first_id))/1e9
    except ValueError as exc:
        raise ValueError('Timestamp replay requires nanosecond frame IDs; set --capture-fps for other data.') from exc
    if offset < 0:
        raise ValueError('Replay timestamps must be ordered.')
    return offset


def run_live(args, frames, est, out, to_origin, bbox):
    from run_portable import validate_frame, select_mask
    from Utils import draw_posed_3d_box, draw_xyz_axis
    from stage_profile import StageProfile, device_synchronize

    first = next(frames)
    initial_time = time.perf_counter()
    if first[-1] is None:
        first = (*first[:-1], select_mask(first[1]))
    validate_frame(*first[1:])
    cv2.imwrite(str(out / 'initial_rgb.png'), cv2.cvtColor(first[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / 'initial_mask.png'), first[-1].astype(np.uint8)*255)
    np.save(out / 'initial_depth_m.npy', first[2])
    np.savetxt(out / 'cam_K.txt', first[3])
    registered = threading.Event()
    stop_replay = threading.Event()

    def source():
        try:
            if args.source == 'demo':
                # Replay the scene after initial registration, equivalent to
                # holding the physical object still during initialization.
                while not registered.wait(.05):
                    if stop_replay.is_set():
                        return
                base_time = time.perf_counter()
                for index, frame in enumerate(frames, start=1):
                    deadline = base_time+replay_offset(frame[0], first[0], index, args.capture_fps)
                    if stop_replay.wait(max(0., deadline-time.perf_counter())):
                        return
                    yield frame
            else:
                yield from frames
        finally:
            frames.close()

    stream = (out / 'timings.csv').open('w', newline='')
    writer = csv.DictWriter(stream, fieldnames=['frame', 'stage', 'inference_seconds'])
    writer.writeheader()
    count = 0

    def infer(frame):
        nonlocal count
        frame_id, rgb, depth, k, mask = frame
        validate_frame(rgb, depth, k, mask)
        start = time.perf_counter()
        if count == 0:
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
        writer.writerow(dict(frame=frame_id, stage='register' if count == 0 else 'track', inference_seconds=f'{elapsed:.6f}'))
        stream.flush()
        count += 1
        registered.set()
        return pose

    pipeline = LivePipeline(source(), infer, first, max_results=args.max_frames, initial_time=initial_time).start()
    result_version, latest = 0, None
    display_times = deque(maxlen=60)
    ages = deque(maxlen=60)
    display_deadline, report_at = time.perf_counter(), 0.
    stats = {}
    vis = cv2.cvtColor(first[1], cv2.COLOR_RGB2BGR)
    try:
        while True:
            pipeline.raise_if_failed()
            item = pipeline.results.wait(result_version, timeout=.001)
            if item is not None:
                result_version, latest = item
                if result_version > 1:
                    ages.append(time.perf_counter()-latest.sample.captured_at)
                frame_id, rgb, depth, k, _ = latest.sample.value
                image = draw_posed_3d_box(k, img=rgb, ob_in_cam=latest.value @ np.linalg.inv(to_origin), bbox=bbox)
                image = draw_xyz_axis(image, ob_in_cam=latest.value, scale=.05, K=k, is_input_rgb=True)
                vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if args.save_frames:
                    cv2.imwrite(str(out / 'track_vis' / f'{frame_id}.png'), vis)
            now = time.perf_counter()
            if now >= display_deadline:
                stats = pipeline.stats()
                if args.display:
                    display = vis.copy()
                    if latest is None:
                        raw = pipeline.frames.wait(0, timeout=0)
                        if raw is not None:
                            display = cv2.cvtColor(raw[1].value[1], cv2.COLOR_RGB2BGR)
                        text = 'Initializing pose; keep object still'
                    else:
                        age = now-latest.sample.captured_at
                        text = f'Capture {stats["capture_fps"]:.1f}  Pose {stats["pose_fps"]:.1f} FPS  Age {age*1000:.0f}ms'
                    cv2.putText(display, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                    cv2.imshow('FoundationPose portable', display)
                    display_times.append(time.perf_counter())
                display_deadline += 1/args.target_fps
                if display_deadline < now:
                    display_deadline = now+1/args.target_fps
            if args.display and cv2.waitKey(1) & 255 in (27, ord('q')):
                break
            if now >= report_at:
                ui_fps = (len(display_times)-1)/(display_times[-1]-display_times[0]) if len(display_times) > 1 else 0.
                print(f'capture={stats.get("capture_fps", 0):.1f} pose={stats.get("pose_fps", 0):.1f} display={ui_fps:.1f} FPS; '
                      f'processed={stats.get("processed", 0)} dropped={stats.get("dropped", 0)}', flush=True)
                report_at = now+1
            if pipeline.results.closed and pipeline.results.wait(result_version, timeout=0) is None:
                break
            if not args.display:
                time.sleep(.002)
    finally:
        stop_replay.set()
        pipeline.stop()
        pipeline.join()
        stream.close()
        cv2.destroyAllWindows()
        stats = pipeline.stats()
        display_fps = (len(display_times)-1)/(display_times[-1]-display_times[0]) if len(display_times) > 1 else 0.
        stats.update(target_fps=args.target_fps, display_enabled=args.display,
                     discarded_total=stats['captured']-stats['processed'],
                     display_fps=display_fps,
                     tracking_latency_ms_median=float(np.median(ages)*1000) if ages else None,
                     measured_pose_fps_meets_target=stats['pose_fps_overall'] >= args.target_fps if stats['processed'] >= 31 else None)
        (out / 'pipeline_metrics.json').write_text(json.dumps(stats, indent=2))
    pipeline.raise_if_failed()
    print(f'Saved {count} pose updates to {out}; target={args.target_fps:g} FPS, measured tracking={stats["pose_fps"]:.2f} FPS', flush=True)
