# 上传模型的实际架构

分析对象是 `model.py` 的工厂 `model()`，不是仅根据上游 VLIF README 推测。当前上传版已经更改了顶层 SUNet、像素重排及多级脉冲量化模块。

## 1. 主干与实际块数

工厂指定 dim=48，T=4，use_refinement=False。默认可训练参数 **14,023,911**。虽然构造参数的 block 列表有4项，forward只建立3个主层级；列表第0和第3项不会按其值建立对应层。第1层已被SUNet替代。

输入 BCHW RGB，输出同尺寸 BCHW RGB。静态图像复制为4个时间步，**不是4帧实测序列**。

| 顺序 | 模块 | 输出（省略batch，保留time） |
|---|---|---|
| 输入重复 | 同一图像复制T=4 | 4×3×H×W |
| 浅层嵌入 | 3×3 Conv，3→48 | 4×48×H×W |
| Encoder L1 | 1个SUNet_Level1_Block | 4×48×H×W |
| Down 1→2 | LIF→stride2 Conv→tdBN | 4×96×H/2×W/2 |
| Encoder L2 | 4个Spiking_Residual_Block | 4×96×H/2×W/2 |
| Down 2→3 | LIF→stride2 Conv→tdBN | 4×192×H/4×W/4 |
| Encoder L3 | 8个Spiking_Residual_Block | 4×192×H/4×W/4 |
| Decoder L3 | 2个Spiking_Residual_Block | 4×192×H/4×W/4 |
| Up 3→2 | bilinear×2→LIF→Conv→tdBN | 4×96×H/2×W/2 |
| L2 跳连融合 | 与encoder L2拼接；LIF→1×1 Conv→tdBN | 4×96×H/2×W/2 |
| Decoder L2 | 2个Spiking_Residual_Block | 4×96×H/2×W/2 |
| Up 2→1 | bilinear×2→LIF→Conv→tdBN | 4×48×H×W |
| L1 跳连融合 | 与encoder L1拼接；LIF→1×1 Conv→tdBN | 4×48×H×W |
| Decoder L1 | 1个SUNet_Level1_Block | 4×48×H×W |
| Additional | 再加1个SUNet_Level1_Block | 4×48×H×W |
| 可选 refinement | True时额外4个Spiking_Residual_Block；默认关闭 | 4×48×H×W |
| 重建 | 时间平均→普通3×3 Conv，48→3→加输入图 | 3×H×W |

SUNet内部和所有残差块中的PixelUnshuffle需要偶数特征尺寸；最深主层在H/4处仍要除2，故输入H/W都需为8倍数。SR适配器负责右/下填充并裁回，网络本体不改变目标尺度。

## 2. Spiking_Residual_Block到底算什么

第一组使用普通LIF取h，定义l=x-h，再组合：

`c = a_h*h + a_l*(x-h) + x*h`，然后Conv→tdBN。

第二组对上述特征做相同形式的组合，但h来自PixelShuffleLIFBlock，再Conv→tdBN。两组的a_h/a_l是独立可学习标量，初始均为1。

输出顺序为：`x + MDA(branch + shortcut_conv_bn(x))`。MDA为SpikingJelly的时间、通道、空间注意力串联并带ReLU；不是Transformer的Q/K/V自注意力。

代码注释把h/l称为高频/低频，但这只是机制解释标签。`h=LIF(x)`或像素重排脉冲变换，不等价于严格的傅里叶高通；`x-h`也没有数学保证只含低频。两标量初始为1时，组合先化简为`x+x*h`，并非两条互不重叠频带。若论文要声称频率选择，应通过频谱响应或受控扰动单独检验。

## 3. 两种空间到“时间”的路径

### PixelShuffleLIFBlock

1. 对每个时间步做PixelUnshuffle(2)：T×B×C×H×W → T×B×4C×H/2×W/2。
2. 把2×2内4个像素偏移位置展开到时间轴：T=4变为T'=16，通道仍为C。
3. 用mem_update逐步处理这16个位置/时间的组合序列。
4. 重新堆回4C通道；**使用双线性插值恢复H/W，再1×1 Conv把4C降为C**。

名称虽含PixelShuffle，恢复路径实际上没有inverse PixelShuffle。这个模块把局部空间位置排列引入有顺序的脉冲递推，因此可能对偏移顺序敏感。

### SUNet_Level1_Block

先经过1个Spiking_Residual_Block，再把2×2像素偏移展开成16步并送入mem_update。随后：

- TimeAttention：对C/H/W池化，为16个伪时间步产生权重（不是对16帧真实视频加权）。
- Conv3D(C→C, kernel=(4,1,1), stride=(4,1,1)) + ReLU：把16步压回4步，空间保持H/2×W/2。
- 两组 LIF→Conv→tdBN→MDA。
- bilinear上采样×2，与最初残差块输出相加，再MDA。

这个小U结构在encoder L1、decoder L1和额外增强层共出现3次。

## 4. 两类脉冲单元

普通LIFNode：threshold=.15、tau使用SpikingJelly默认2、decay_input=False、hard reset到0；没有声明detach_reset，保持库默认False。它有跨时间步的膜状态，需要每个独立样本/batch/tile之后reset。

自定义mem_update：首步m=x；之后`m_t=.25*(m_(t-1)-s_(t-1).detach())+x_t`。发放是：

`q(m) = round(clamp(m,0,4))/4`，即输出0、.25、.5、.75、1五级值；反向在[0,4]内给1/4梯度，外侧为0。mem_update的状态是forward内部局部变量，每次调用自动从零开始。

这两个单元的泄漏、重置和输出量化规则不同，不能全部概括成同一个标准二值LIF。

## 5. 对THz任务的意义与限制

这是有脉冲内部计算的混合恢复网络：存在连续值残差、乘法门控、1×1/3D卷积、注意力和浮点输出。不能仅凭模型名声称所有计算都是二值事件驱动，或直接把理论SNN能耗当成实测能耗。

网络没有学习式×2/×3/×4外部SR头；内部down/up只用于特征尺度变化，最终仍与输入同尺寸。因此本次在模型之外使用bicubic预放大，数据集本身保留原生LR和原始HR，便于后续native SR模型复用。

当前保持3通道。真实灰度8-bit图像通过复制通道适配，不能据此声称获得了额外THz谱信息。

## 6. 本次对model.py的改动范围

- 去掉硬编码.cuda()，工厂允许指定device/backend；默认torch，CUDA可显式选cupy。
- UpSampling把逐步双线性插值改为对T×B合并批量插值，保持相同插值规则、输出和参数。
- 修复PyPI 0.0.0.0.14的MDA缺少nn.Module继承和self.relu：保留其现有time/channel/spatial计算，确保注意力权重可调用、可转移设备、加入优化器并保存；[官方当前实现](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/layer/attention.py)也包含nn.Module和ReLU。旧release直接统计会遗漏57,176个注意力参数；14,023,911是正确注册后的数量。
- 补上tdBN折叠后NCHW维度检查：SpikingJelly 0.0.0.0.14继承_BatchNorm但没有实现该检查，在当前PyTorch上报NotImplementedError；权重初始化、运行统计和归一化公式不变。
- 显式检查T=4和输入尺寸/通道，避免错误参数拖到中间层才出错。

没有替换网络、调整宽度或block数，没有把原THz-Net的32个块/残差缩放强行套入，也没有添加学习SR头。上游MIT许可保留。
