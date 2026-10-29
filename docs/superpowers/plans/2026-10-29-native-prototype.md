# Mac 原生优化实施计划

**Goal:** 测量真实首帧初始化瓶颈，交付可对照的可选原生原型。

**Architecture:** 现有推理接口中加入默认关闭的上下文计时；独立离线基准使用保存的首帧或官方数据。保留默认后端，根据实测将细化网络接入可选Core ML运行时。

**Tech Stack:** Python 3.11、PyTorch MPS、标准库计时/JSON、Numba、可选Core ML FP32运行时；Metal渲染延后。

## 全局约束

- 分支 `perf/macos-native-prototype`；保持main运行行为与本地数据。
- 默认不启用同步计时或原生原型。
- 不减少252候选、5次初始化细化、2次跟踪细化来制造加速。
- 真实相机图像、权重、编译产物保持忽略；公开报告仅含统计。

## 执行

- [x] 为 `upstream/stage_profile.py` 写计时聚合、嵌套独占时间、关闭计时无同步的失败测试；实现并验证。
- [x] 在estimator、refiner/scorer和portable renderer中插入阶段；增加 `--profile-init` 可选输出。
- [x] 实现 `upstream/benchmark_initialization.py`，可读取保存首帧或官方demo；采集原实现的分段结果和无分段同步耗时。
- [x] 根据实测优先实现Core ML细化网络原型：细化网络12.10s（48.7%）、评分网络2.06s（8.3%）、CPU渲染8.32s（33.5%）。新增可选模型导出及原生推理包装，保留Torch后端。
- [x] 保持原有渲染器，复跑几何/纹理/深度测试；使用相同权重和真实首帧做完整A/B。CoreML FP32在CPU_AND_GPU及ALL模式均选中候选75，热运行仅改善约4%，不切换默认后端。Metal批量渲染作为后续建议，本阶段不实现。
- [x] 代码审查、31项测试、Torch/CoreML输入检查、使用说明及性能报告；交付独立分支。静态30步跟踪中位Torch63ms、CoreML58ms，不代表实时相机FPS。
