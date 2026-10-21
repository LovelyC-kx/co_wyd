# THz-Net 原文与当前数据管线的对应关系

依据仓库 `OE_THz_Net_Submission.pdf`；不把其他论文或原 VLIF 去雨配置混入这套对比。原文没有提供可下载的训练代码和完整数值退化参数，不能声称与原作者生成的输入逐像素相同。

## 1. 你提供的 DIV2K 下载说明应该怎样选

仅下载官方 **Train Data (HR images)**、**Validation Data (HR images)** 两项。对应 800 张 `0001..0800` 与 100 张 `0801..0900`。不下载网站上的 LR bicubic / unknown、NTIRE 2018 mild / difficult / wild。那些是各自赛道预制的输入，不能直接代替 THz-Net 的 PSF + 采样 + 噪声退化。也不额外拼 Flickr2K，否则训练数据不再与原文一致。

官网总数据是 1000 张，但测试集 HR `0901..1000` 未公开。论文 §3.2 写“900 images (800 for training, 100 for validation and testing)”，与公开的 train HR + valid HR 对应。因此当前把 100 张作为固定合成验证/评测集；真实 THz 系统 A/B/C 是另一个测试域。论文未说明这 100 张内部是否另有验证/测试划分，本实现没有擅自细分，也不把它称为独立于调参的最终测试集。

## 2. 逐项对照

| 项目 | 原文位置与已知内容 | 本实现 |
|---|---|---|
| 训练原图 | §3.2，DIV2K 800 张 | 严格检查 train HR 0001–0800 |
| 验证原图 | §3.2，100 张 validation/testing | valid HR 0801–0900，全图固定退化 |
| 前向模型 | §2.1 Eq.1，y=S(x*h)+z | 高斯卷积→双三次降采样→高斯噪声 |
| PSF 类型 | §2.1 Eq.2，高斯近似 | 各 RGB 通道使用同一个归一化各向同性高斯核 |
| PSF 宽度/核大小 | 未给定 | rho=1.6 HR pixels，21×21，**固定示例假设** |
| 噪声 | §2.1，z∼N(0,beta²)，未给 beta | beta=.01，[0,1] 单位，**固定示例假设** |
| 采样算子 | 未给实现细节 | PyTorch bicubic，antialias=True，align_corners=False，**实现选择** |
| 倍率 | Fig.10 比较 ×2/×3/×4 | 三个倍率分别训练，默认先 ×2；不声称主表只用 ×2 |
| 优化器/损失 | §3.2，Adam / MSE | Adam / RGB MSE；取消去雨 SSIM 损失、AdamW |
| 训练轮数 | §3.5、Table 4 最终 20 | 默认 20 |
| 学习率 | §3.5 最终 5×10⁻⁶ | 默认 5e-6；常数 LR，调度未公开 |
| batch | §3.5 最终 32 | 默认 32；每轮 800 个随机 crop，25 个 batch |
| patch | §3.5 最终 192，未说明 HR/LR | 明确采用 HR 192×192；LR 分别 96/64/48，**约定** |
| 通道 | 数据为 RGB，评测 Y；训练颜色转换未详述 | 保留 RGB 三通道，MSE 比较 RGB，**实现约定** |
| 数据增强/采样次数 | 未说明 | 默认无翻转/旋转；每图每 epoch 一个 crop，**约定** |
| 随机 seed | 未说明 | 1234，**约定** |
| Adam beta/eps/decay | 未说明 | (0.9,0.999) / 1e-8 / 0，**约定** |
| 训练 LR 裁剪 | 未说明 | 加噪后不截断，保持加性高斯模型；可显式配置 |
| 指标 | §3.2，YCbCr Y 上 PSNR/SSIM | 浮点结果先 clamp[0,1]；BT.601 studio Y，峰值范围 1 |
| SSIM 及 metric border | 未说明 | 11×11 Gaussian sigma1.5，valid window；border0，**约定** |
| 真实 THz 测试 | §3.1 三个4.3THz透射系统；Data Availability为合理请求获取 | 不在仓库；需取得图像及参考定义后接入，不制造实验结果 |

论文 §2.2.2 提及残差缩放 0.1、§3.5 最终选 0.05，这是原 THz-Net 的结构设定；同理，32 residual blocks 也是 THz-Net 本体参数。当前对比保留上传的 VLIF 结构，不修改其块数/残差缩放来冒充 THz-Net。

## 3. 配对生成和模型适配

```text
RGB HR PNG /255
  train: 每图每epoch裁剪一个192×192 HR patch（默认不增强）
  valid: 全图，只从右/下去掉不能被倍率整除的少量像素
→ 高斯PSF卷积（边界反射）
→ bicubic降采样（antialias=True）
→ 在LR网格上添加高斯噪声（默认不截断）
→ 原生LR + 未退化HR，作为所有模型共用的配对
```

裁剪发生于退化之前；反射边界、核截断与 bicubic 的额外抗混叠均为本实现选择，原文没有公开。严谨复现需作者给出 rho/beta、核大小、采样函数及完整参数，最好还需给一对实际生成的 LR/HR 作逐像素校验。

上传 VLIF 为同尺寸残差网络。因此模型边界使用 `LR → bicubic放大至HR尺寸（antialias=False）→ VLIF → HR`，不向网络内部添加学习上采样头。这一适配层应在论文方法中写明，不应当作 THz-Net 原始上采样模块。未来原生 SR 网络可接收同一 LR，使用 `--model-input native`，输出尺度由适配器检查。

模型输入在右/下复制填充到8倍数，网络输出裁掉这部分填充；没有拉伸 HR 标签。原尺寸不能整除倍率的验证原图会 modcrop 以定义整数 LR/HR 配对。

## 4. 公平性与可重复性

- 训练 crop/噪声由 `(seed, epoch, image_id, split)` 唯一决定；模型随机数、worker、shuffle顺序不影响配对。同一个epoch所有模型看同样的crop/噪声。
- 验证不随机裁剪、不增强；退化固定在epoch0，不随训练轮次变化；按图平均 Y-PSNR、Y-SSIM。
- 原始文件内容、配置、数据代码、依赖版本形成 data_fingerprint，检查文件编号阻止训练/验证混用。
- 默认无 AMP、无 warmup、无 cosine、无 weight decay、无额外 loss；这些都不是原文已报告配置。
- 默认全图分块验证：模型输入尺度192，overlap32，均匀平均融合；每块前后重置SNN状态。该设置是显存实现约定，分块可能影响全局注意力与边缘输出，必须对比时注明。
- 若batch32不适合当前网络显存，改变batch/梯度累积会改变训练协议；不要不记录地降低batch然后声称设置完全一致。
- 训练与验证MSE计算不把网络输出截断；指标计算时截断预测值到[0,1]。指标基于PNG保存之前的浮点结果，保存PNG后重评存在量化差异。
- bicubic基线使用相同的已加噪声LR和相同HR目标；不把无噪声LR给基线。
- `allow-partial`和step/image限制仅供检查管线，在manifest中明确标记，不能当正式完整训练。

## 5. 真实THz数据与后续模型

没有HR参考的THz图只能做推理、线剖面或其他另行设计的评价；不能计算有参考PSNR/SSIM。`test.py --input-dir ...`不再对实测输入重复施加合成退化。

8-bit灰度显示图会复制为RGB，以保持上传模型三通道结构。带负值的原始电压、强度/透射率等数据须明确物理定义、暗场/参考场和归一化，再单独实现输入校准，不把任意min-max映射默认为强度校准。

后续新模型只改变factory/kwargs以及native或bicubic输入契约，保留本配置、原图、退化代码、seed、epoch采样、loss和评价。适配器会检查HR输出形状。新模型若需要额外输入（PSF、噪声水平、多频/多帧），必须单独定义这些量，不能悄悄改变当前单图任务。

`model_best.pth`按100张验证图选择，不代表独立真实THz测试分数。最终跨系统性能应在未用于调参的真实THz集上报告，并明确参考图像的获得方式。
