# Metal Renderer Implementation Plan

**Goal:** 在相同候选和细化次数下减少CPU渲染耗时。
**Architecture:** `portable_backend.render`共享输入/输出约定；`metal_renderer.py`负责ctypes与编译，`native/metal_renderer.mm`负责Metal渲染。
**Tech Stack:** Python标准库、NumPy、Objective-C++、系统Foundation/Metal。

## 约束

继续perf/macos-native-prototype；默认Numba；252候选、初始化5次与跟踪2次不变；数据、权重、编译产物不上传。

## 步骤

- [ ] 写 `tests/test_metal_renderer.py`，继承现有几何/纹理测试并选Metal；运行确认因后端缺失失败。
- [ ] 实现原生顶点/片元渲染与ctypes包装；运行上述测试，比较CPU/GPU几何内部误差。
- [ ] 在run_portable与benchmark_initialization增加 `--renderer {numba,metal}` 并记录选择；拒绝CUDA+Metal。
- [ ] 相同保存首帧运行普通A/B及分段计时；增加静态跟踪检查，确认候选和位姿差异。
- [ ] 更新使用说明与报告，完整测试、代码审查、提交推送分支。
