"""Read D455 RGB-D without a mesh or pose model; save a bounded capture report."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from run_portable import realsense_depth_metres


def summarize_frame_timing(numbers, arrivals):
    ids = np.asarray(numbers, dtype=np.int64)
    duration = float(arrivals[-1]-arrivals[0])
    if ids.shape != (len(arrivals), 2) or len(arrivals) < 2 or duration <= 0:
        raise ValueError('Timing requires at least two RGB-D observations over a positive duration.')
    result = dict(observed_framesets=len(arrivals), frameset_fps=(len(arrivals)-1)/duration)
    for column, name in enumerate(('color', 'depth')):
        delta = np.diff(ids[:, column])
        result[name] = dict(first_frame=int(ids[0, column]), last_frame=int(ids[-1, column]),
                            duplicate_transitions=int(np.count_nonzero(delta == 0)),
                            backward_transitions=int(np.count_nonzero(delta < 0)),
                            skipped_frame_numbers=int(np.maximum(delta-1, 0).sum()),
                            fresh_fps=float(np.count_nonzero(delta > 0)/duration))
    return result


def needs_usb_permission_hint(started, error):
    text = str(error).lower()
    return not started and any(word in text for word in ('permission', 'access', 'no device connected'))


def main():
    import pyrealsense2 as rs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--min-depth', type=float, default=.1)
    parser.add_argument('--max-depth', type=float, default=3.)
    parser.add_argument('--infrared', action='store_true', help='Also capture both infrared images for depth diagnostics.')
    args = parser.parse_args()
    if args.frames < 2:
        parser.error('--frames must be at least 2')
    if not np.isfinite(args.min_depth) or not np.isfinite(args.max_depth) or not 0 <= args.min_depth < args.max_depth:
        parser.error('Depth limits must satisfy 0 <= min-depth < max-depth and be finite.')
    root = Path(__file__).resolve().parent
    out = args.output or root / 'debug' / f'd455-check-{datetime.now().strftime("%Y%m%d-%H%M%S")}-{time.time_ns()%1000000:06d}'
    out.mkdir(parents=True, exist_ok=False)
    rs.log_to_file(rs.log_severity.warn, str(out / 'sdk.log'))
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if args.infrared:
        for index in (1, 2):
            config.enable_stream(rs.stream.infrared, index, 640, 480, rs.format.y8, 30)
    started = False
    arrivals, numbers, camera_times = [], [], []
    report = dict(status='starting', output=str(out.resolve()), requested_frames=args.frames)
    try:
        context = rs.context()
        report['enumerated_devices'] = len(context.query_devices())
        print(f'Enumerated devices: {report["enumerated_devices"]}', flush=True)
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        report.update(device=device.get_info(rs.camera_info.name),
                      firmware=device.get_info(rs.camera_info.firmware_version),
                      usb=device.get_info(rs.camera_info.usb_type_descriptor),
                      depth_scale_m=device.first_depth_sensor().get_depth_scale())
        sensor = device.first_depth_sensor()
        report['depth_sensor_options'] = {
            str(option): sensor.get_option(option)
            for option in (rs.option.emitter_enabled, rs.option.laser_power, rs.option.exposure,
                           rs.option.gain, rs.option.enable_auto_exposure, rs.option.visual_preset)
            if sensor.supports(option)
        }
        print(f'{report["device"]}; USB {report["usb"]}; depth scale {report["depth_scale_m"]} m', flush=True)
        for _ in range(30):
            pipeline.wait_for_frames(10000)
        for _ in range(args.frames):
            frames = pipeline.wait_for_frames(10000)
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                raise RuntimeError('Frameset is missing a color or depth frame.')
            arrivals.append(time.perf_counter())
            numbers.append([color.get_frame_number(), depth.get_frame_number()])
            camera_times.append([color.get_timestamp(), depth.get_timestamp()])
        timing = summarize_frame_timing(numbers, arrivals)
        report['stream_timing'] = timing
        warnings = []
        for name in ('color', 'depth'):
            if timing[name]['duplicate_transitions'] or timing[name]['backward_transitions']:
                warnings.append(f'{name}: {timing[name]["duplicate_transitions"]} repeated, '
                                f'{timing[name]["backward_transitions"]} backward frame-number transitions')
        report['warnings'] = warnings
        for warning in warnings:
            print(f'Frame timing warning: {warning}; continuing to save diagnostic images.', flush=True)
        native_depth = np.asanyarray(frames.get_depth_frame().get_data()).copy()
        native_m = realsense_depth_metres(native_depth, report['depth_scale_m'], args.min_depth, args.max_depth)
        cv2.imwrite(str(out / 'depth_native_raw.png'), native_depth)
        if args.infrared:
            for index in (1, 2):
                infrared = frames.get_infrared_frame(index)
                if infrared:
                    cv2.imwrite(str(out / f'infrared_{index}.png'), np.asanyarray(infrared.get_data()))
        aligned = rs.align(rs.stream.color).process(frames)
        color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
        rgb = np.asanyarray(color_frame.get_data()).copy()
        raw_depth = np.asanyarray(depth_frame.get_data()).copy()
        depth_m = realsense_depth_metres(raw_depth, report['depth_scale_m'], args.min_depth, args.max_depth)
        valid = depth_m > 0
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        streams_advance = all(timing[name]['fresh_fps'] > 0 and timing[name]['backward_transitions'] == 0 for name in ('color', 'depth'))
        status = ('ok' if streams_advance else 'capture_unstable') if valid.any() else 'depth_unusable'
        report.update(status=status, acquired_frames=len(arrivals),
                      working_range_m=[args.min_depth, args.max_depth],
                      saturated_depth_fraction=float((raw_depth == 65535).mean()),
                      native_valid_depth_fraction=float((native_m > 0).mean()),
                      native_saturated_depth_fraction=float((native_depth == 65535).mean()),
                      capture_fps=(len(arrivals)-1)/(arrivals[-1]-arrivals[0]),
                      rgb_shape=list(rgb.shape), depth_shape=list(raw_depth.shape),
                      valid_depth_fraction=float(valid.mean()),
                      depth_p05_p50_p95_m=np.percentile(depth_m[valid], [5, 50, 95]).tolist() if valid.any() else None,
                      color_frame_number=int(numbers[-1][0]), depth_frame_number=int(numbers[-1][1]),
                      intrinsics=dict(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                                      width=intr.width, height=intr.height,
                                      distortion=str(intr.model), coefficients=list(intr.coeffs)))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / 'color.png'), bgr)
        cv2.imwrite(str(out / 'depth_raw.png'), raw_depth)
        np.save(out / 'depth_m.npy', depth_m)
        scaled = np.clip(depth_m/args.max_depth, 0, 1)*255
        heatmap = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
        heatmap[~valid] = 0
        cv2.imwrite(str(out / 'preview.png'), np.hstack([bgr, heatmap]))
        np.savetxt(out / 'cam_K.txt', [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        median = f'{np.median(depth_m[valid]):.3f}m' if valid.any() else 'none'
        print(f'Captured: {len(arrivals)} framesets; fresh RGB {timing["color"]["fresh_fps"]:.2f} FPS, '
              f'fresh depth {timing["depth"]["fresh_fps"]:.2f} FPS; '
              f'valid depth ({args.min_depth:g}–{args.max_depth:g}m) {valid.mean():.1%}; median depth {median}; '
              f'saturated {report["saturated_depth_fraction"]:.1%}', flush=True)
        if not valid.any():
            print('RGB-D transport works, but no depth lies inside the configured working range.', file=sys.stderr)
            return 2
        if not streams_advance:
            return 3
    except Exception as exc:
        report.update(status='error', error=str(exc))
        print(f'Capture failed: {exc}', file=sys.stderr, flush=True)
        if needs_usb_permission_hint(started, exc):
            print('On macOS, USB capture may require running this command with sudo in your own terminal.', file=sys.stderr)
        return 1
    finally:
        if started:
            try:
                pipeline.stop()
            except Exception as exc:
                report['stop_error'] = str(exc)
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        (out / 'frame_timing.json').write_text(json.dumps(dict(host_arrivals=arrivals, frame_numbers=numbers,
                                                             camera_timestamps_ms=camera_times), indent=2))
        print(f'Report: {out.resolve() / "report.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
