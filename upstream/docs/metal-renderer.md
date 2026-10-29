# 可选 Metal 渲染后端

分支：`perf/macos-native-prototype`。将候选位姿的CPU光栅化移至Apple Metal，网络仍默认Torch/MPS；候选数和细化次数保持不变。默认渲染器仍是Numba，可以随时对照。

## 使用

要求macOS 15+、支持Metal的GPU、Xcode Command Line Tools（`xcode-select --install`）。不需要新增Python包。首次使用会调用系统clang++编译Objective-C++共享库到忽略的 `native/build/`；Metal着色器在首次渲染时编译。编译或运行失败会明确报错，不静默回退。此后复用本机编译产物。

```bash
cd FoundationPose/upstream
./run_sr300.sh --check --renderer metal
sudo ./run_sr300.sh --renderer metal
```

回到参考后端：

```bash
sudo ./run_sr300.sh --renderer numba
```

保存的真实首帧基准（输入格式见 [原生原型说明](native-prototype.md)）：

```bash
.venv/bin/python benchmark_initialization.py \
  --input debug/portable-实际时间戳 \
  --output debug/init-metal --renderer metal --runs 2 --profile --track-steps 30
```

输出目录必须不存在。普通运行用于性能比较；`--profile`额外运行一次带阶段同步的初始化。`--track-steps`重复同一RGB-D快照，仅测静态跟踪执行开销，不能当成真实相机FPS。需要相机实测时，可加 `--profile-init` 查看初始化阶段。

可选组合Core ML的细化网络：

```bash
sudo ./run_sr300.sh --renderer metal --refiner-backend coreml --coreml-compute-units ALL
```

需要先按原生原型说明导出FP32模型。当前组合收益有限，建议先使用Metal＋Torch。

## 本机测量

同一D455真实白色阶梯件首帧，M4 / 24GB、252候选、5次初始化细化、批次8；普通运行不加阶段同步：

| 组合 | 第一次 | 第二次 | 静态跟踪中位（30步，每步2次细化） |
|---|---:|---:|---:|
| Numba＋Torch（复测） | 24.477s | 24.598s | 62.1ms |
| Metal＋Torch | 16.600s | 16.559s | 55.6ms |
| Metal＋Core ML ALL | 16.482s | 16.150s | 56.1ms |

Metal＋Torch热运行约24.60s→16.56s，耗时下降约33%，约1.49倍速度。这是单输入少量重复结果，首次本机编译和缓存预热需要另计；没有降低候选数或细化次数。初次原型含编译运行18.38s，后续重复约16.4–16.6s。Core ML组合并未显示值得切换默认网络的额外收益。

分段同步测量中，Metal渲染合计约0.473s，对照此前CPU渲染约8.325s；细化网络约12.309s，评分编码约2.122s。优化后主要热点已转为网络推理。分段同步会改变调度，因此性能结论使用表中普通运行。

Numba＋Torch与Metal＋Torch热运行选中候选分别为75和104；最终位姿相差约0.138mm、0.065°。Metal＋Core ML则选中225，最终位姿相差约0.026mm、0.033°，不保证候选索引一致。角度差由相对旋转的轴角计算，以减轻float32矩阵非正交对小角度trace公式的影响。

只含配置、耗时和差值的报告见 [metal-measurements.json](metal-measurements.json)，不含相机图像或物体绝对位姿。

## 实现与一致性范围

渲染器使用Metal顶点/片元管线与深度测试，每次最多16个位姿的临时目标；输出透视插值后的RGB、米制深度、相机坐标XYZ和法线。保留裁剪、纹理环绕、FP32显式双线性采样、逐顶点方向光及近面交叉三角形整片跳过的参考语义。双线性采样采用显式读取，避免硬件采样权重量化。

Python通过标准库ctypes调用本机共享库；原生提交完成后再读取输出，仍有CPU读回和MPS上送，尚未实现与Torch的零拷贝共享。共享队列入口串行化，不缓存所有模型资源；这些简化在当前尺寸下不是主要耗时。

硬件填充规则和子像素插值与CPU浮点光栅化有差异，尤其是轮廓与内部遮挡边界，不能保证逐像素或候选索引完全相同。几何、遮挡、裁剪、透视深度、纹理方向/环绕、光照、19位姿跨批次测试均覆盖。真实阶梯件渲染对照的轮廓IoU约0.99987，重叠内部像素深度差99分位约0.030mm，少量遮挡边界可达到毫米量级。

实物首帧初始化对照显示，Metal与Numba选中候选索引不同，但收敛到接近的最终位姿。位姿差只衡量实现一致性，不是真值精度。当前尚未验证D455持续运动场景中的延迟、掉帧和跟踪精度，因此保留显式开关。

## 复现与参考

本轮45项单元测试全部通过，包括14项Metal测试；默认与Metal启动检查、Python语法编译及脚本语法检查通过。代码审查指出的原生命令创建失败处理已补齐，命令完成后才读回。

运行全部测试：`.venv/bin/python -m unittest discover -s tests -v`；macOS上Metal测试实际调用GPU，其他平台跳过。

Metal数组渲染目标按位姿切片；CPU读取前等待命令完成。对应系统接口说明：[render attachment slice](https://developer.apple.com/documentation/metal/mtlrenderpassattachmentdescriptor/slice)、[texture getBytes](https://developer.apple.com/documentation/metal/mtltexture/getbytes(_:bytesperrow:bytesperimage:from:mipmaplevel:slice:))。
