# VLIF 模型用于 THz-Net 数据协议下的图像重建对比

本仓库保留上传的 VLIF/SUNet 模型，在 DIV2K 上使用 **PSF 模糊 → 降采样 → 加性高斯噪声** 构造监督配对。训练和评测共用同一数据协议，方便后续更换模型。

**复现边界：** `OE_THz_Net_Submission.pdf` 的 §3.2 明确给出 DIV2K 800/100、Adam、MSE、Y 通道 PSNR/SSIM；Table 4 最终设置为 20 epochs、batch 32、patch 192、LR 5e-6。但没有公开 PSF 宽度、核大小、噪声强度或采样实现。因此本仓库是**遵循已公开设置、补充明确实验假设的对比管线**，不能宣称逐值复现原作者数据或论文分数。

- [原文与实现逐项对照、下载选择及全部假设](docs/THZ_PROTOCOL.md)
- [上传模型的真实架构、张量尺寸和脉冲单元解析](docs/MODEL_ARCHITECTURE.md)
- [上游 VLIF 去雨项目说明（保留出处）](docs/VLIF_UPSTREAM_README.md)

## 数据：只下载两个 HR 压缩包

| 下载项 | 用途 |
|---|---|
| [DIV2K_train_HR.zip](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) | `0001.png`–`0800.png`，训练 |
| [DIV2K_valid_HR.zip](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) | `0801.png`–`0900.png`，验证/合成评测 |

[DIV2K 官方主页](https://data.vision.ee.ethz.ch/cvl/DIV2K/)。无需下载 bicubic/unknown/mild/difficult/wild LR 包，也不加入 Flickr2K。原文真实 THz 跨系统测试集需向作者取得，不能用 DIV2K 验证集替代后声称复现了真实 THz 结果。

解压为：

```text
data/DIV2K/
  DIV2K_train_HR/0001.png ... 0800.png
  DIV2K_valid_HR/0801.png ... 0900.png
```

默认严格检查 800/100 张图的编号。原图始终作为 HR 目标；数据集返回原生 LR，模型外的适配层进行双三次放大，再交给原有同尺寸 VLIF 网络。

## 环境

建议 Python 3.10+。先按机器安装匹配的 PyTorch / torchvision CPU 或 CUDA 版本，然后：

```sh
python -m pip install -r requirements.txt
```

默认采用 SpikingJelly 的 `torch` 后端，可在 CPU/CUDA 上运行。已安装匹配 CUDA 的 CuPy 时，可显式加 `--backend cupy`。无需原去雨脚本的 warmup_scheduler、thop、LPIPS、TensorBoard 等依赖。

## 训练与评测

```sh
python train.py --data-root data/DIV2K --config configs/thznet_div2k.json --output runs/vlif_x2
python test.py --data-root data/DIV2K --weights runs/vlif_x2/model_best.pth --output results/vlif_x2
```

`configs/thznet_div2k.json` 中的 `blur_sigma=1.6`、`kernel_size=21`、`noise_std=0.01` 是**本次固定的示例退化参数，非原文公开数值**。取得作者参数或实测标定后，修改该配置，所有对比模型一起重新训练。噪声单位是归一化图像 [0,1]，0.01 约相当于 8-bit 的 2.55 灰度级。

不同倍率单独训练，不混合为一次实验：

```sh
python train.py --data-root data/DIV2K --scale 3 --output runs/vlif_x3
python train.py --data-root data/DIV2K --scale 4 --output runs/vlif_x4
```

实际要比较哪个倍率，就先完成该倍率；配置中默认 ×2 只是首个实验。所有覆盖参数都会写入有效协议。恢复同一实验：

```sh
python train.py --data-root data/DIV2K --output runs/vlif_x2 --resume runs/vlif_x2/model_last.pth
```

恢复时应提供与原运行相同的配置及覆盖参数；数据、代码、模型、执行设置不一致会拒绝恢复。默认保存每轮 `model_last.pth`，按验证 Y-PSNR 保存 `model_best.pth`。不要用最终测试集选择 checkpoint。

真实 THz 原生 LR 图像，有同名、已配准 HR 参考时：

```sh
python test.py --input-dir data/THz/input --target-dir data/THz/target --weights runs/vlif_x2/model_best.pth --output results/thz_x2
```

没有 HR 参考时，去掉 `--target-dir`，仅输出重建图，不计算 PSNR/SSIM。灰度 PNG 会复制为三通道以保持模型不变；带负值的探测器矩阵或高位深原始数据需要先明确校准规则，不能当成 8-bit PNG 随意归一化。原生 LR 的每个尺寸乘以倍率后必须与参考一致。

## 验证与显存

默认验证整幅图，以模型输入尺度的 `192×192` 重叠 tile（overlap 32）推理并平均融合，每个 tile 前后重置脉冲状态。`--tile-size 0` 可关闭分块。分块、插值与边界处理均为公开声明的实现选择，论文未给出这些细节。训练和测试默认沿用同一分块设置；修改会记录在评测报告中。

batch 32 是论文配置，不保证任何 GPU 都能容纳本 VLIF 网络。若改 batch、patch 或退化参数进行资源适配，必须在对比中注明并给所有模型一致设置。

## 后续新模型

数据协议与网络解耦；保持配置、数据和 seed 不变，只替换工厂和输入接口。例如新模型接收原生 LR 并在内部放大：

```sh
python train.py --data-root data/DIV2K --model-factory new_model:build_model --model-kwargs '{"scale":2}' --model-input native --spatial-multiple 1 --output runs/newmodel_x2
```

`new_model.py` 需由你后续提供；上述命令仅展示接口。Windows CMD 的 JSON 引号写法不同，可把参数先写成变量或使用 PowerShell。若新模型仍是同尺寸恢复网络，沿用 `--model-input bicubic`。输入输出均约定为 BCHW RGB，输出为对应 HR，训练脚本会检查形状。不同模型必须比较相同的 `manifest.json` 中协议和 `data_fingerprint`。

实验保存：有效配置、文件 SHA256、依赖版本、数据代码哈希、模型源码哈希、原图名单、每轮指标和 bicubic 基线。验证对固定；同一 epoch/原图的训练裁剪与噪声不依赖 worker、shuffle 顺序或模型随机数。

## 自动检查

```sh
python -m pip install pytest
python -m pytest -q
```

测试仅用临时生成的图像和随机张量验证协议、数值、接口及完整模型的前反向；不代表已完成 DIV2K 训练或复现原文 PSNR。

## 出处与许可

模型来自用户上传的 VLIF 修改版；[VLIF 上游项目](https://github.com/ChrisChen1023/VLIF) 与已有 MIT LICENSE 保留。原文为仓库内的 [THz-Net PDF](OE_THz_Net_Submission.pdf)。DIV2K 图像按数据集原始条款使用。

## 新增 submission 模型与钥匙原始矩阵推理

`submission` 内的 ORBITALIF 模型已接入相同 DIV2K 配置；无需另建退化数据或调用原云去除训练脚本。

```sh
python -m submission.train_div2k --data-root data/DIV2K --output runs/submission_snn_x2
python -m submission.test_div2k --data-root data/DIV2K --weights runs/submission_snn_x2/model_best.pth --output results/submission_snn_x2
```

对新增的钥匙OPJU和`wyd01(1).xls`，使用原始数值入口：

```sh
python infer_measurement.py --opju "钥匙(1).opju" --normalization minmax --weights runs/submission_snn_x2/model_best.pth --output results/key_x2
```

此入口同样接受原先根目录模型的协议checkpoint。它保留含负值的101×101矩阵，使用显式输入映射、三通道复制，按checkpoint倍率重建并逆映射回原始数值尺度，导出NPY/TSV和预览图。没有HR参考时不计算PSNR/SSIM。**min-max输入映射不等于物理透射率校准。**

[详细命令、OPJU读取方式与配置一致性说明](docs/SUBMISSION_DIV2K_KEY.md)。
