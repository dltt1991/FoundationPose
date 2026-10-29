# FoundationPose Mac 本机原型

基于上游 `a1b694b`，分支 `macos-mps`。目标机器：Apple M4 MacBook Air，10 核 GPU，24GB 内存。

保留官方预训练权重和 model-based 位姿估计/跟踪算法，PyTorch 网络运行在 MPS；Numba CPU 光栅器替代 nvdiffrast，深度过滤及旋转聚类也在 CPU 执行。不需要编译 CUDA、PyTorch3D、Warp 或 mycpp，不包含 model-free / BundleSDF 重建。

## 环境

所有命令在本 README 所在的 `upstream/` 目录执行。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt
```

当前工作区已创建环境，下载官方权重和 mustard0 数据，直接执行运行命令即可。`requirements-mac.txt` 固定已验证的直接依赖；完整环境版本保存在 `docs/macos-environment.txt`。

## 官方数据

首次在其他机器准备时，从[官方 README](https://github.com/NVlabs/FoundationPose#data-prepare)获取：

- Refiner：`weights/2023-10-28-18-33-37/{config.yml,model_best.pth}`。
- Scorer：`weights/2024-01-11-20-02-45/{config.yml,model_best.pth}`。
- 示例解压为 `demo_data/mustard0/`，包含 `cam_K.txt`、`rgb/`、`depth/`、`masks/`、`mesh/`。

需要原权重和配置，不能只下载 `.pth`。示例深度原文件是毫米，程序读入后转换为米。

## 离线运行

```bash
.venv/bin/python run_portable.py --device mps --check
.venv/bin/python run_portable.py --device mps --max-frames 3
```

默认只跑 3 帧，首帧使用全部 252 个候选旋转、5 次细化，后续帧跟踪细化 2 次。跑完整序列并显示叠加图：

```bash
.venv/bin/python run_portable.py --device mps --max-frames 0 --display
```

使用 `--device cpu` 排查 MPS 差异。`--batch-size 8` 控制神经网络编码批次；评分的跨候选注意力仍然覆盖全部候选。首帧初始化会明显慢于后续跟踪，首次使用还有编译/预热开销。

输出位于独立的 `debug/portable-<时间戳>/`，也可用 `--output <尚不存在的目录>` 指定：

- `ob_in_cam/*.txt`：原始物体模型到相机的 4×4 变换，OpenCV 坐标约定（x 右、y 下、z 前），平移单位米。
- `track_vis/*.png`：3D 框和坐标轴叠加。
- `timings.csv`：逐帧推理耗时，包含预处理、渲染、网络及 MPS 同步，不包含读盘和保存图片；其倒数不是相机端到端 FPS。
- `run.json`、`cam_K.txt`、`initial_rgb.png`、`initial_mask.png`、`initial_depth_m.npy`：运行配置及首帧输入。

已有输出目录会被拒绝，不执行上游 demo 中的清空目录操作。

## 10 FPS 流水与低延迟预览

RealSense 自动启用流水；可以用官方数据验证：

```bash
.venv/bin/python run_portable.py --pipeline --display --max-frames 0
```

采集线程、推理线程和主线程显示独立运行。输入和结果各只有一个最新值槽，推理来不及时覆盖旧输入，避免积压；每个结果保存其对应的 RGB-D，框始终绘制在同一帧图像上。深度过滤使用 4 个 CPU 线程并释放 GIL，固定裁剪几何和模型中心变换缓存，跟踪仍保留 2 次细化。可用 `NUMBA_NUM_THREADS` 覆盖深度过滤线程数。

默认按 demo 文件名中的纳秒时间戳回放：737 帧跨 71.09 秒，不能直接当作固定 30 FPS 视频，否则动作被加速近 3 倍并可能丢失跟踪。只有做吞吐压力测试时才显式加 `--capture-fps 30`；该设置不用于评估真实时间下的跟踪质量。

`--target-fps 10` 控制主线程显示目标并检查位姿吞吐。窗口和日志分别显示 Capture、Pose、Display FPS，以及结果对应输入的年龄。初始化估计期间有独立预览，物体应保持不动。追踪输出目录额外包含 `pipeline_metrics.json`：

- `pose_fps`：最近窗口中实际完成的位姿更新率；`pose_fps_overall` 是整个跟踪阶段完成率，不含注册。
- `display_fps`：GUI 刷新率，和位姿更新率分别统计。
- `measured_pose_fps_meets_target`：至少完成 30 次跟踪后按全程吞吐判定，否则为 null。源数据本身间隔较大时也可能低于目标。
- `tracking_latency_ms_median`：最近窗口中，从源生成器交付 RGB-D 到主线程收到对应位姿的时间；不包含传感器曝光、SDK 对齐/矫正前的时间或屏幕扫描延迟。
- `dropped`：消费输入时跳过的旧帧；`discarded_total` 包含结束时未处理的输入。

流水默认只保存位姿、计时、配置与首帧；增加 `--save-frames` 可保存接收到的结果叠加图，会增加主线程开销。Q/Esc 退出显示；退出会等待当前推理和相机阻塞读取结束，不会中途打断 GPU 内核。

本机实测和质量对照见 [docs/macos-validation.md](docs/macos-validation.md)。10 FPS 是位姿更新性能目标，D455 的 RGB-D 采集已验证约 30 FPS，实际位姿吞吐仍需接入物体模型测试。

## D455 相机入口

用户提供的白色阶梯件 `SR300_step_target_M4.stl` 已配置毫米缩放及白色材质，直接运行 `sudo ./run_sr300.sh`。尺寸、坐标及操作说明见 [docs/sr300-model.md](docs/sr300-model.md)。

```bash
.venv/bin/python run_portable.py --source realsense \
  --mesh /absolute/path/to/your-object.obj \
  --device mps --max-frames 0 --display
```

相机必须对着**与 mesh 对应的真实物体**。模型尺寸必须是米；毫米模型增加 `--mesh-scale 0.001`。OBJ 纹理文件和 MTL 保持相对路径可访问。不能直接用 mustard 模型去跟踪其他物体。

实时输入默认仅接受 `0.1–3m` 深度，并过滤 Z16 的零值和 65535 上限值；可用 `--min-depth`、`--max-depth` 调整工作范围。不要把上限值换算出的 `65.535m` 当成正常测距。

不需要模型的独立采集诊断（macOS USB 权限要求在本机终端执行）：

```bash
sudo .venv/bin/python check_realsense.py --frames 60 --infrared
```

输出保存在 `debug/d455-check-*`：彩色图、对齐深度、未对齐原始深度、两张红外图、预览和报告。有效率按配置的工作范围统计，饱和值另报；60帧采集FPS不包含随后单帧对齐和保存开销。

多路同步可能重复使用某一路图像；诊断程序会分别记录RGB/深度重复、倒退、跳号和实际新帧FPS，保留图像继续诊断。`capture_fps` 是frameset返回速率，实际各路更新率见 `stream_timing.color.fresh_fps`、`stream_timing.depth.fresh_fps`。无推进或编号倒退会标记 `capture_unstable`，不会被报告为正常；失败时也保存已取得的帧编号和时间戳。

程序启动 640×480 / 30Hz RGB-D 流，等待曝光稳定，把深度对齐到彩色图，读取 SDK 内参和深度尺度，并矫正彩色相机畸变。推理帧率由处理耗时决定。第一帧会冻结：鼠标沿物体边缘点击多边形，Enter/空格确认，R 重画，Q/Esc 退出；随后开始估计和跟踪。初始化时必须保留真实物体的轮廓，避免把大块背景纳入。

当前没有失踪自动检测或重定位，物体离开视野或跟踪漂移后应停止并重新初始化。D455 的 IMU 不参与此方案。USB 实机 RGB-D 采集已验证；目标物体上的位姿和帧率仍需接入模型后测试。

### macOS Python SDK

本机已完成 SDK v2.59.1 编译，`pyrealsense2` 可导入，通过虚拟环境中的 `foundationpose-realsense.pth` 加载本地扩展，D455 已接入并通过管理员权限采集验证：60 个 RGB-D frameset，RGB 与深度均约 30 FPS。PyPI 稳定包没有匹配的 macOS ARM64 / CPython 3.11 wheel，重建环境时需要从 SDK 构建。官方资料：[macOS 安装](https://github.com/realsenseai/librealsense/blob/master/doc/installation_osx.md)、[Python 绑定](https://github.com/realsenseai/librealsense/tree/master/wrappers/python)。

下面是本地构建配置（SDK v2.59.1），不需要全局 `sudo make install`：

```bash
brew install cmake libusb pkg-config
git clone --depth 1 --branch v2.59.1 https://github.com/realsenseai/librealsense.git downloads/librealsense
cmake -S downloads/librealsense -B downloads/librealsense/build-mac \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_WITH_DDS=OFF -DBUILD_ROSBAG2=OFF -DCHECK_FOR_UPDATES=OFF -DBUILD_LEGACY_PYBACKEND=OFF
cmake --build downloads/librealsense/build-mac --target pyrealsense2 -j4
export PYTHONPATH="$PWD/downloads/librealsense/build-mac/Release${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -c 'import pyrealsense2 as rs; print("connected cameras:", len(rs.context().query_devices()))'
```

SDK macOS 文档指出部分 USB 操作可能需要提升权限；仅在实际出现 USB 权限问题时按其说明排查。本程序不会自动提权。连接相机、SDK 能枚举到设备后，再验证彩色与深度采集。

## 测试与限制

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖 Z-buffer、透视正确插值、XYZ、裁剪、纹理方向、光照、深度过滤、对称旋转聚类、MPS 精度兼容、评分分批语义及输出目录保护。

CPU 光栅器没有训练梯度，与 nvdiffrast 不保证像素级一致，近裁剪面穿过三角形时会跳过该三角形（物体应完整处于相机前方）。默认方向光和纹理/顶点色已支持，其他投影矩阵和光源模式不支持。当前离线演示没有真值位姿，叠加图只能作目视检查；不能据此宣称达到论文精度。
