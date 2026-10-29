# 初始化分段计时与 Core ML 原生推理原型

实验分支：`perf/macos-native-prototype`。默认仍为原有 Torch/MPS 网络和 CPU/Numba 渲染。

这一步没有把整个工程重写成C++，而是将占时最多的细化网络接到Apple原生Core ML运行时，保留Python调度、原评分网络与渲染器，验证原生后端的实际价值。

## 先测瓶颈

```bash
cd FoundationPose/upstream
.venv/bin/python run_portable.py --max-frames 1 --profile-init
```

D455白色阶梯件可以使用：

```bash
sudo ./run_sr300.sh --profile-init
```

输出目录增加 `setup_profile.json` 和 `registration_profile.json`。记录各阶段调用次数、包含子阶段时间与排除子阶段后的独占时间。关闭计时后不增加设备同步；打开计时会在阶段边界同步MPS，可能改变调度，因此最终加速结论使用**没有分段同步**的重复运行耗时。

阶段包括模型构建、深度预处理、候选生成、裁剪几何、CPU渲染、结果上送、图像变换、归一化、细化网络、位姿更新、评分编码及跨候选注意力。

## 无需相机权限的重复基准

不传 `--input` 时使用官方mustard0第一帧：

```bash
.venv/bin/python benchmark_initialization.py \
  --output debug/init-torch --runs 2 --profile
```

已有实物运行首帧时，可以传其输出目录，目录必须包含 `run.json`、`initial_rgb.png`、`initial_mask.png`、`initial_depth_m.npy` 和 `cam_K.txt`：

```bash
.venv/bin/python benchmark_initialization.py \
  --input debug/portable-实际时间戳 \
  --output debug/init-torch-real --runs 2 --profile
```

两次普通运行后，`--profile`会额外执行一次分段计时；该次不计入普通性能数据。初始化固定5次细化，批次默认8，默认模型生成252候选。`result.json`记录输入/模型配置、候选数、每次耗时、选中候选和完整位姿，便于同输入对照。首次运行包含缓存与预热影响，应分别查看首次与热运行。

## 导出原生模型

Core ML为可选依赖，普通运行不需要安装。已验证工具链：Python3.11、PyTorch2.15、coremltools9.0。Core ML工具尚未官方验证PyTorch2.15；本实验仅验证当前RefineNet图、固定输入及下方对照结果。

先在主环境安装可选推理运行时，不变更原有PyTorch/NumPy版本：

```bash
uv pip install --python .venv/bin/python -r requirements-coreml.txt
```

转换工具在NumPy2.4上会遇到标量转换错误。使用独立导出环境固定NumPy2.2.6，并复用主环境中的PyTorch等依赖：

```bash
uv venv --python 3.11 .venv-coreml
.venv-coreml/bin/python - <<'PY'
from pathlib import Path
import site
path = Path(site.getsitepackages()[0]) / 'foundationpose-baseline.pth'
path.write_text(str(Path('.venv/lib/python3.11/site-packages').resolve()) + '\n')
PY
uv pip install --python .venv-coreml/bin/python numpy==2.2.6 coremltools==9.0
.venv-coreml/bin/python export_refiner_coreml.py
```

默认导出到 `weights/coreml-refiner-fp32/`，目录必须尚不存在。保留官方原始权重不变，生成固定批次1/4/8的FP32模型及 `.mlmodelc` 编译产物，覆盖跟踪单帧、252候选批次8及最后4个候选。导出时关闭PyTorch MHA trace快速路径，运行随机输入的Torch/CoreML输出对照，默认绝对误差门限为1e-4。

`manifest.json`记录原始检查点SHA256、版本、批次及误差。所有批次通过后才设置 `validated=true`。未通过验证、模型缺失或检查点不匹配的导出会被拒绝加载。编译模型与导出环境不上传，需要在目标Mac重新生成。

当前不推荐FP16：一次随机输入实验的平移头最大绝对误差约0.0615，尚未建立足够的真实输入精度保证；默认保持FP32。

## 使用原生后端

```bash
.venv/bin/python benchmark_initialization.py \
  --output debug/init-coreml --runs 2 --profile \
  --refiner-backend coreml --coreml-compute-units CPU_AND_GPU
```

同一个真实首帧目录应同时传给Torch与CoreML基准。`--coreml-compute-units ALL` 允许系统自动选择计算单元，但不能据此宣称使用了Neural Engine；实际分配由Core ML决定。

实物运行：

```bash
sudo ./run_sr300.sh --refiner-backend coreml
```

只替换细化网络，评分仍为Torch；不减少候选和细化次数。原生推理需要把MPS输入读回CPU数组，再将输出送回原设备；这些开销包含在普通A/B总耗时内，并在分段计时中单独记录。

原型仅用于macOS的CPU/MPS路径，不支持CUDA；默认批次8与导出保持一致，其他批次可能需要重新导出。其他计算单元、物体、姿态、光照下的泛化和持续跟踪精度仍需进一步验证。

## 本次阶段性结果

同一真实白色阶梯件首帧，官方权重、252候选、初始化5次细化、批次8：

| 普通运行（无分段同步） | 首次 | 第二次 |
|---|---:|---:|
| Torch/MPS | 28.244s | 25.020s |
| Torch/MPS，在CoreML测量后复测 | 28.716s | 24.864s |
| Core ML FP32，CPU_AND_GPU | 24.535s | 24.099s |
| Core ML FP32，ALL | 24.027s | 23.923s |

首次Torch运行受Numba/设备预热影响，不能将首次差值全部算作后端收益。热运行仅改善约4%，这是单场景少量重复的观察值，不是稳定加速保证。只含计时和配置的机器可读报告见 [native-measurements.json](native-measurements.json)，不包含私有相机图像或物体位姿。

基线分段计时：初始化24.817s；细化网络12.097s（48.7%）、评分编码2.062s（8.3%）、CPU渲染8.325s（33.5%）。模型构建另计约1.512s。

FP32固定批次导出的随机输入最大绝对误差约3.3e-6。真实首帧的Torch与CoreML CPU_AND_GPU选中相同候选75，最终位姿最大矩阵元素差约1.8e-7，平移差约1.5e-8m。它们只是实现一致性检查，不是实物真值精度。

补充静态跟踪检查：初始化后，对同一RGB-D快照连续执行30步，每步2次细化，Torch中位耗时约63ms，Core ML ALL约58ms。两条路径均产生有限位姿；最终位姿最大矩阵元素差约 3.9e-07，平移差约 2.1e-08m。这只检查静态输入的执行开销与一致性，不代表D455持续采集、显示下的实时帧率或动态跟踪精度。可用上述基准命令追加 `--track-steps 30` 复现。

结论：当前数据不支持因为Python调度就整体重写C++。原生网络后端收益较小，下一项更值得研究的是占约三分之一初始化时间的批量GPU渲染；该阶段之后已增加可选 [Metal渲染器及对照结果](metal-renderer.md)。本分支保留可选Core ML原型，不切换默认后端。

## 验证

Core ML阶段通过31项单元测试、Python语法编译、启动脚本语法检查及Torch/CoreML输入检查。真实保存首帧完成初始化A/B和30步静态跟踪；尚未验证Core ML在D455持续运动场景中的延迟与精度。
