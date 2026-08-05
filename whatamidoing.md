# GenComm 代码说明

## 1. 项目定位

GenComm 是一个面向自动驾驶协同感知的研究代码库，目标是在多智能体场景下完成 3D 目标检测。代码在 OpenCOOD 和 HEAL 的基础上扩展，核心思路是让不同传感器/不同检测器的智能体在中间 BEV 表示上进行协同。

当前仓库里最重要的几条主线是：

- 单一模态或同构协同的基础训练
- 异构协同的两阶段训练
- 基于 GenComm 的条件扩散式消息重建/对齐
- 面向 V2X-Real、OPV2V、DAIR-V2X、V2XSet 等数据集的配置分发

## 2. 入口脚本

### 训练入口

- [opencood/tools/train.py](opencood/tools/train.py)
- [opencood/tools/train_ddp.py](opencood/tools/train_ddp.py)

这两个脚本负责：

1. 读取 YAML 配置
2. 构建数据集
3. 创建模型和损失
4. 组织训练、验证、保存 checkpoint
5. 训练结束后自动调用对应推理脚本做测试

`train.py` 的核心流程比较直接：

- `yaml_utils.load_yaml(...)` 载入配置
- `build_dataset(...)` 创建训练集和验证集
- `train_utils.create_model(hypes)` 创建模型
- `train_utils.create_loss(hypes)` 创建损失
- `train_utils.setup_optimizer(...)` 和 `train_utils.setup_lr_schedular(...)` 处理优化器和学习率调度
- 进入 epoch 循环，前向、反向、保存 checkpoint、验证

### 推理入口

- [opencood/tools/inference.py](opencood/tools/inference.py)
- [opencood/tools/inference_v2xreal.py](opencood/tools/inference_v2xreal.py)
- [opencood/tools/inference_heter_in_order.py](opencood/tools/inference_heter_in_order.py)
- [opencood/tools/inference_v2xreal_heter_in_order.py](opencood/tools/inference_v2xreal_heter_in_order.py)

不同数据集和训练范式会走不同的推理脚本，尤其是 V2X-Real 需要专用脚本。

## 3. 模型分发机制

模型不是在代码里写死的，而是通过 YAML 中的 `model.core_method` 动态加载。

加载逻辑在 [opencood/tools/train_utils.py](opencood/tools/train_utils.py) 里：

- 根据 `core_method` 拼出模块路径，比如 `opencood.models.heter_model_baseline_w_gencomm_stage1`
- 导入这个 Python 文件
- 在模块中寻找一个类名，要求忽略大小写后与文件名去掉下划线的形式一致

这意味着：

- 文件名要对
- 类名也要对
- YAML 里的 `core_method` 也要对

例如：

- `heter_model_baseline_w_gencomm_stage1`
- 对应文件 [opencood/models/heter_model_baseline_w_gencomm_stage1.py](opencood/models/heter_model_baseline_w_gencomm_stage1.py)
- 对应类名应为 `HeterModelBaselineWGenCommStage1`

如果类名不匹配，就会报“backbone not found in models folder”，但实际问题不是 backbone，而是模型类没找到。

## 4. GenComm 主模型

### Stage 1

[opencood/models/heter_model_baseline_w_gencomm_stage1.py](opencood/models/heter_model_baseline_w_gencomm_stage1.py)

这个文件负责 GenComm 的基础协同模型。它的流程可以概括为：

1. 为每个模态构建 encoder
2. 通过 backbone 提取 BEV 特征
3. 用 shrinker 压缩特征图
4. 用 message extractor 提取协作消息
5. 把消息送入 GenComm
6. GenComm 生成/补全特征后，再进入 enhancer 和 fusion_net
7. 最后走共享检测头输出分类、回归和方向预测

文件中的关键成员：

- `self.gencomm = GenComm(args['gencomm'])`
- `encoder_{modality}`
- `backbone_{modality}`
- `shrinker_{modality}`
- `message_extractor_{modality}`
- `self.enhancer`
- `self.fusion_net`
- `self.cls_head / self.reg_head / self.dir_head`

训练时的 forward 主要逻辑位于：

- [opencood/models/heter_model_baseline_w_gencomm_stage1.py:174-297](opencood/models/heter_model_baseline_w_gencomm_stage1.py#L174-L297)

这里可以看到：

- 从 `data_dict` 里取出 `agent_modality_list`、`pairwise_t_matrix`、`record_len`
- 逐模态编码和提特征
- 组装 `heter_feature_2d` 和 `heter_message`
- 经过 `self.gencomm(...)`
- 用 `pred_feature` 替换原始特征
- 再做 enhancer、fusion、head 预测

### Stage 2

[opencood/models/heter_model_baseline_w_gencomm_stage2.py](opencood/models/heter_model_baseline_w_gencomm_stage2.py)

stage2 和 stage1 类似，但它会冻结更多模块，只训练异构协同阶段需要更新的部分。它的特点是：

- `self.fix_modules` 指定被冻结的模块
- `model_train_init_stage2()` 遍历这些模块并关闭梯度
- 仍然保留 encoder/backbone/shrinker/message_extractor/fusion head 的整套结构
- 推理时支持缺失消息场景

stage2 forward 的核心流程也在模型文件中，整体与 stage1 一致，但冻结策略更强，更适合异构协同的第二阶段训练。

## 5. 数据集加载

V2X-Real 的数据加载基类在：

- [opencood/data_utils/datasets/basedataset/v2xreal_basedataset.py](opencood/data_utils/datasets/basedataset/v2xreal_basedataset.py)

这个类负责：

- 读取 `root_dir` / `validate_dir`
- 枚举场景文件夹
- 构建样本索引
- 读取激光雷达、相机、标注、位姿等信息
- 生成训练所需的中间字典

这里有几个容易混淆的点：

- `root_dir` 只在训练时使用
- `validate_dir` 只在验证/测试时使用
- `dataset_mode` 必须匹配配置里的类型
- `max_cav` 控制一个样本最多读取多少个协作智能体

在 V2X-Real 配置里，训练数据路径通常应写成：

```yaml
root_dir: "/data/bohnsix/datasets/v2xreal/train"
validate_dir: "/data/bohnsix/datasets/v2xreal/test"
test_dir: "/data/bohnsix/datasets/v2xreal/test"
```

## 6. 配置文件组织

所有实验都通过 YAML 驱动，主要目录包括：

- `opencood/hypes_yaml/opv2v/`
- `opencood/hypes_yaml/v2xreal/`
- `opencood/hypes_yaml/dairv2x/`
- `opencood/hypes_yaml/v2xset/`
- `opencood/hypes_yaml/v2xsim2/`

GenComm 的 V2X-Real 配置集中在：

- `opencood/hypes_yaml/v2xreal/GenComm_yamls/gencomm/stage1/`
- `opencood/hypes_yaml/v2xreal/GenComm_yamls/gencomm/stage2/`

这些 YAML 里最核心的几项是：

- `model.core_method`
- `loss.core_method`
- `fusion.core_method`
- `fusion.dataset`
- `heter.assignment_path`
- `heter.mapping_dict`
- `train_params.batch_size`
- `train_params.max_cav`
- `root_dir / validate_dir / test_dir`

## 7. 训练过程

### Stage 1

Stage1 是同构协同基座训练。每个模态单独训练，得到各自的基础 checkpoint。

流程：

1. 复制 stage1 YAML 到日志目录中的 `config.yaml`
2. 通过 `train.py --model_dir <log_dir>` 启动
3. 训练结束后保存 `net_epoch_bestval_at*.pth`

### Stage 2

Stage2 是异构协同训练。这里会先把 stage1 的 checkpoint 合并，再继续训练异构协同所需的模块。

合并逻辑由：

- [opencood/tools/heal_tools.py](opencood/tools/heal_tools.py)

完成。这个脚本会自动寻找各目录下最好的 checkpoint，并把它们合并成一个新的异构初始化权重。

### Stage 3

Stage3 是推理和测试。对于普通方法可以直接跑 `inference.py`，对于 V2X-Real 则应使用专用的推理入口。

## 8. 这次实际改过的内容

在当前工作中，已经做过几类修改：

- 修复 GenComm stage1 / stage2 类名与 `core_method` 不一致的问题
- 将训练脚本里的 SwanLab 默认关闭
- 把 V2X-Real 的 batch size 从 2 调成 1
- 把 V2X-Real 的数据路径改为 `/data/bohnsix/datasets/v2xreal`

这些改动都围绕同一个目标：让 V2X-Real 上的 GenComm 训练和测试流程可以正常启动，并且尽量降低显存占用和外部依赖影响。

## 9. 关键调试结论

几个最重要的结论可以直接记住：

- `train.py` 会默认训练完后自动跑推理
- `model_dir` 给了以后，`-y` 会被忽略，直接读 `config.yaml`
- GenComm 的模型类名必须和 `core_method` 对得上
- `m1` 显存比 `m2` 大，主要是 backbone 更深，不是 batch size
- V2X-Real 要用专门的推理脚本，不要直接套 OPV2V 的命令

## 10. 当前工作建议

如果继续做 V2X-Real 训练，建议按这个顺序：

1. 确认 `config.yaml` 中的路径已经改对
2. 确认 stage1 / stage2 的类名匹配已经修复
3. 先跑 `m1` 的 stage1，确认数据加载和前向正常
4. 再跑 `m2`、`m3`、`m4`
5. 最后进入 stage2 合并和训练
6. 训练完成后再走 V2X-Real 的专用推理脚本

