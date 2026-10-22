# submission 模型的 DIV2K 适配与钥匙实测推理

从 `co_wyd` 根目录运行本页命令。这里的训练/评测复用根目录已建立的 THz-Net 风格协议。原 submission 云去除与联邦入口仍按原实验使用，不拿其 CUHK 数据、Charbonnier–SSIM、AdamW、warmup/cosine 配置来进行这次 THz 对比。

## 1. 接入了哪个模型

`submission/cloud_removal_v2` 共用 `cloud_removal_v1/models`。新增入口通过 `submission.models_div2k:build_model` 构造上传的网络，没有重新实现其主干。

| --backbone | 网络 | 默认参数量 |
|---|---|---:|
| snn | ORBITALIF/VLIFNet，含FSTA频谱注意力、FreMLP与AGFM门控跳连 | 2,302,901 |
| ann | 同架构的ReLU替换版本 | 2,302,901 |
| plain | 上传的PlainUNet | 1,311,603 |

默认 dim24、T4、refinement关闭，输出与输入同尺寸，因此沿用根目录模型同样的 **原生LR→bicubic预上采样→恢复网络→HR** 适配器。网络输入在右/下复制填充到8倍数，再裁回输出；原始输入数值不会因尺寸适配而丢行/丢列。

v1/v2是训练系统版本，不是两套不同骨干。此处进行集中式模型对比，不新增卫星分区或联邦聚合实验。

## 2. 与上一次模型完全共用的配置

以下文件没有改变：

- `configs/thznet_div2k.json`
- `dataset_load.py`
- `thz_protocol.py`
- `sr_runtime.py`
- `metrics.py`

| 项目 | 同一设置 |
|---|---|
| DIV2K | train HR 0001–0800，valid HR 0801–0900 |
| 合成输入 | 高斯PSF卷积→bicubic降采样→LR加性高斯噪声 |
| 退化示例参数 | sigma1.6、核21、noise_std.01 |
| 训练 | Adam、RGB MSE、20 epochs、batch32、HR patch192、LR5e-6 |
| 随机性 | seed1234；按epoch/image_id确定裁剪/噪声，不依赖模型和worker |
| 评测 | 固定验证退化、同一Y通道PSNR/SSIM、相同192/32分块方式 |
| 倍率 | 默认×2；×3/×4分别训练 |

这些退化数值仍是上次明确声明的固定实现选择；原 THz-Net 稿件未公开精确PSF、噪声幅度、采样细节。此次没有把它们重新解释成作者参数。

同一环境、相同原图和相同覆盖参数下，两套模型的 `manifest.json` 中 `protocol`、`data_fingerprint` 与 `execution` 应一致。自动检查已实际验证这一点；模型源码哈希和模型参数量则各自记录。后续修改FSTA或兼容模块也会被checkpoint的组件哈希检测到。

## 3. 安装、训练和验证

先安装匹配机器的PyTorch/torchvision，再执行根目录的 `pip install -r requirements.txt`。CUDA安装匹配的CuPy后可加 `--backend cupy`，默认torch后端支持CPU/CUDA。

```sh
python -m submission.train_div2k --data-root data/DIV2K --backbone snn --output runs/submission_snn_x2
python -m submission.test_div2k --data-root data/DIV2K --weights runs/submission_snn_x2/model_best.pth --output results/submission_snn_x2
```

ANN或Plain对照只替换backbone与输出目录：

```sh
python -m submission.train_div2k --data-root data/DIV2K --backbone ann --output runs/submission_ann_x2
python -m submission.train_div2k --data-root data/DIV2K --backbone plain --output runs/submission_plain_x2
```

倍率通过同一根目录选项覆盖：

```sh
python -m submission.train_div2k --data-root data/DIV2K --scale 3 --output runs/submission_snn_x3
python -m submission.train_div2k --data-root data/DIV2K --scale 4 --output runs/submission_snn_x4
```

`--dim`、`--use-refinement`是模型选项；其余参数交给根目录train.py。正式对比保留上表配置，资源调试可用 `--allow-partial`、step限制和小patch，输出会标为smoke test。batch32是否适合当前GPU需实测，不能将较小batch的调试结果当成相同配置。

## 4. 钥匙文件实际是什么

仓库中的文件为：

- `钥匙(1).opju`：Origin项目容器，头部为 `CPYUA 4.3445 200`；包含Book1工作表、图形/虚拟矩阵信息和工作表缩略图。缩略图只用于核查来源，绝不作为推理像素。
- `wyd01(1).xls`：扩展名虽然是xls，内容是ASCII制表符文本，CRLF换行，**不是二进制Excel**。没有表头、坐标行或坐标列；读取全部101行、101列，不丢掉首行首列。

实际数值范围 `[-0.0233003, 0.000236512]`。这是一幅含负值的单通道测量，不能按8-bit图片除255，也不能直接把负值截成0。文件没有提供物理量单位、参考场/暗场、扫描步长或绝对空间方向，因此目前只保留原始数字尺度，不将其宣称为已校准透射率。

默认行视作y、列视作x，保留文件顺序，不旋转、翻转或转置。Origin工作表显示的A(X)/B(Y)列指定不会被误用为需要删除的坐标列。

`configs/key_wyd01.json`记录此对文件的SHA256、形状和已知Origin工作表名。它用于定位与防止误配，不表示已经在Linux解码了完整OPJU数值表。

## 5. 跨平台推理：OPJU与已有数值导出配套使用

先只核查输入，不需要模型权重：

```sh
python infer_measurement.py --opju "钥匙(1).opju" --normalization minmax --inspect-only --output results/key_inspect
```

已知的这份项目会自动在旁边查找 `wyd01(1).xls` 并检查两份文件哈希；也可以显式指定：

```sh
python infer_measurement.py --opju "钥匙(1).opju" --matrix "wyd01(1).xls" --normalization minmax --weights runs/submission_snn_x2/model_best.pth --output results/submission_key_x2
```

等价的submission入口为：

```sh
python -m submission.infer_measurement --opju "钥匙(1).opju" --normalization minmax --weights runs/submission_snn_x2/model_best.pth --output results/submission_key_x2
```

上次根目录模型的协议checkpoint同样可用：

```sh
python infer_measurement.py --matrix "wyd01(1).xls" --opju "钥匙(1).opju" --normalization minmax --weights runs/vlif_x2/model_best.pth --output results/firstmodel_key_x2
```

**数值输入来自已有文本导出**，不是用PNG缩略图重建矩阵。报告会明确 `numeric_project_read=false`、`companion_verified_against_project_values=false`，以及该仓库文件对的哈希是否匹配。未知OPJU不能仅凭名称自动匹配；需要显式提供导出矩阵，或使用下一节Origin读取。

此入口不会给实测输入再次施加模糊、降采样或噪声。权重倍率×2将101×101原生输入重建为202×202；×3/×4分别303×303、404×404，不裁掉第101行/列。

## 6. 数值映射、极性与输出

`--normalization`必须显式指定：

- `minmax`：整幅测量的最小/最大值作为low/high，令 `u=(x-low)/(high-low)`。
- `affine --low ... --high ...`：使用用户给定的统一边界。用于多次测量需要固定尺度的情况；超出边界的输入会截到[0,1]，数量记录在报告。
- `--polarity positive`默认“大值更亮”，当前钥匙为亮轮廓。`negative`则令网络输入为 `1-u`，逆映射会恢复同一原始符号体系。不自动根据“金属应暗”等物理猜测改变极性。

两者都是网络输入适配，**不是透射率、吸收系数或电压标定**。若有暗场/参考场与单位，应另行给出物理校准规则。不同模型对比钥匙时必须使用相同输入映射、边界、极性及原始文件。

单通道映射复制为3通道以保留模型结构。重建RGB通过 `0.299R+0.587G+0.114B` 合为全范围灰度，再按原low/high和极性逆映射。这里不用带16偏置的studio-range Y去逆映射测量值。

输出包括：

| 文件 | 含义 |
|---|---|
| input_signal.npy / .tsv | 原始101×101数值，不量化 |
| input_normalized.npy / .tsv | 输入映射后的单通道数据 |
| restored_rgb.npy | 网络浮点RGB输出，保留越界值 |
| restored_normalized.npy / .tsv | 浮点灰度重建，保留越界值 |
| restored_signal.npy / .tsv | 逆映射到原始数字尺度的重建矩阵 |
| bicubic_signal.npy / .tsv | 同一输入的bicubic基线，逆映射到相同数字尺度 |
| input/restored/bicubic_preview.png | 仅作显示，截断至[0,1]后8-bit量化 |
| coordinates.json | LR/HR像素中心索引；没有虚构μm/mm单位 |
| measurement.json | 原图/项目/checkpoint哈希、数值范围、映射、模型配置、尺寸、越界计数等 |

浮点NPY/TSV用于后续分析，PNG只用于看图。重建超出原观察范围时保留在浮点输出中，并记录越界数，避免悄悄掩盖模型过冲。没有独立HR参考，`metrics=null`，不把输入或bicubic当成真值计算PSNR。

## 7. Windows/Origin直接读取OPJU工作表

已安装Origin及其 `originpro` 包的Windows环境可使用官方API：项目只读打开，工作表用`to_list2()`，矩阵表用`to_np2d()`。第一次测试请先保存已有Origin工作，以免打开项目时出现保存询问。

```sh
python infer_measurement.py --opju "钥匙(1).opju" --origin-sheet "[Book1]Sheet1" --matrix "wyd01(1).xls" --normalization minmax --inspect-only --output results/key_origin_check
```

这条路径会读取项目中的真实数值并与已有导出逐项比较，允许文本显示精度对应的小量误差；形状或值不一致则报错。成功时报告 `numeric_project_read=true`、`companion_verified_against_project_values=true`。去掉 `--matrix` 可直接从OPJU取值。再将 `--inspect-only` 替换为 `--weights ...` 即可推理。

若源数据是Origin矩阵表，应使用正确范围名，另加 `--origin-sheet-type m --origin-matrix-index 0`。不给范围时不会随便选择当前激活图或工作表。

官方接口参考：

- [project.open / find_sheet](https://docs.originlab.com/originpro/namespaceoriginpro_1_1project.html)
- [WSheet.to_list2](https://docs.originlab.com/originpro/classoriginpro_1_1worksheet_1_1WSheet.html)
- [Origin矩阵数据交换](https://docs.originlab.com/python/examples/origin-python-data-exchange/)

此分支已做API调用契约检查；当前环境没有Windows/Origin，尚未进行真实COM/Origin联调。已实际读取并检查的是仓库内OPJU文件信息、PNG缩略图和配套ASCII数值矩阵。

## 8. 兼容与验证

上传submission原 `_sj_compat.py` 会全局改写SpikingJelly类，可能改变根目录旧模型的实例化。本次改为submission局部tdBN/MDA类，保持上传模型的公式与state_dict键，不让导入顺序改变另一套模型。原集中式消融中的相应注意力调用也使用同一局部类。将当前submission与上传版本加载相同state_dict进行CPU前向对照，最大差为0；参数键和尺寸一致。

CPU PyTorch2.8环境下29项检查通过：三种submission骨干前反向、原模型参数完整性、两模型数据指纹一致、训练保存/重载评测、两模型的实测矩阵推理、Origin API契约、负值/极性逆映射和真实钥匙文件检查。

检查使用临时小图/短训练验证管线，不是完整DIV2K训练结果。没有报告未经训练权重在钥匙上的恢复质量；没有运行完整20epoch训练、真实Windows/Origin联调或CUDA性能评测。
