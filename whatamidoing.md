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

## 10. m1 / m2 / m3 / m4 的区别

这四个配置都属于 V2X-Real 上的 GenComm stage1，同样使用：

- `model.core_method: heter_model_baseline_w_gencomm_stage1`
- `loss.core_method: point_pillar_v2xreal_gencomm_loss`
- `fusion.core_method: intermediateheterv2xreal`
- `fusion.dataset: v2xreal`
- `assignment_path: opencood/modality_assign/v2xreal_4modality.json`

它们的主要区别不在训练流程，而在**ego 模态、模态映射、backbone 深度、以及最后特征维度**上。

### 1. 共同点

四个配置都在做同一件事：

- 用同一个 V2X-Real 数据集划分
- 用同一个异构协同框架
- 用同一个 GenComm stage1 模型类
- 用同一个消息提取器和融合流程

也就是说，`m1`、`m2`、`m3`、`m4` 不是四个完全不同的方法，而是四个**以不同智能体为基座**的训练配置。

### 2. ego 模态不同

每个配置都把自己的模态设成 ego：

- `m1_att.yaml` 的 `ego_modality` 是 `m1`
- `m2_att.yaml` 的 `ego_modality` 是 `m2`
- `m3_att.yaml` 的 `ego_modality` 是 `m3`
- `m4_att.yaml` 的 `ego_modality` 是 `m4`

这意味着训练时，当前模态会作为主干协同基座，其他模态会按 `mapping_dict` 映射到该基座上。

### 3. mapping_dict 的作用

stage1 里 `mapping_dict` 的写法本质上是：把所有智能体都临时映射到当前 ego 模态。

例如 `m1_att.yaml` 里：

```yaml
mapping_dict:
  m1: m1
  m2: m1
  m3: m1
  m4: m1
```

表示训练这一版时，所有模态都按 `m1` 的结构来走。

`m2 / m3 / m4` 的配置也是同样逻辑，只是目标 ego 变了。

### 4. backbone 差异

这是最关键的区别，也是显存差异的来源。

#### m1

`m1` 使用的是最完整的 backbone：

```yaml
backbone_args:
  layer_nums: [3, 5, 8]
  layer_strides: [2, 2, 2]
  num_filters: [64, 128, 256]
  upsample_strides: [1, 2, 4]
  num_upsample_filter: [128, 128, 128]
```

这意味着它有三层特征提取和三路上采样，最后拼接后的 `input_dim` 是 384。

#### m2

`m2` 比 `m1` 少一层：

```yaml
backbone_args:
  layer_nums: [3, 5]
  layer_strides: [2, 2]
  num_filters: [64, 128]
  upsample_strides: [1, 2]
  num_upsample_filter: [128, 128]
```

最后 `input_dim` 是 256。

#### m3

`m3` 更轻，只保留最浅的一层：

```yaml
backbone_args:
  layer_nums: [3]
  layer_strides: [2]
  num_filters: [64]
  upsample_strides: [1]
  num_upsample_filter: [128]
```

最后 `input_dim` 是 128。

#### m4

`m4` 最特殊，backbone 直接设为 identity：

```yaml
backbone_args: 'identity'
```

也就是说它不再走完整的 BEV backbone，只保留很轻的后续压缩和消息提取逻辑，所以最后 `input_dim` 只有 64。

### 5. 显存和速度差异

四者的显存/速度通常是：

```text
m1 > m2 > m3 > m4
```

原因是：

- `m1` backbone 最深，特征图最多
- `m2` 次之
- `m3` 更浅
- `m4` 几乎不做 backbone 提取

所以你看到 `m1` 显存比 `m2` 大，是正常现象，不是 batch size 的问题。

### 6. 训练时的实际含义

这四个配置分别对应四种不同的基座训练：

- `m1`：最强的 LiDAR 基座，最重，但表达能力最好
- `m2`：中等复杂度基座
- `m3`：轻量基座
- `m4`：最轻，适合极简输入路径或特殊异构设置

在 stage2 里，这四个基座会被重新组合，用来训练 `m1m2`、`m1m3`、`m1m4` 等异构组合。

### 7. 直接看配置时的快速判断

你可以只看这几个字段判断差异：

- `ego_modality`
- `mapping_dict`
- `backbone_args`
- `shrink_header.input_dim`

其中最能说明显存差异的是 `backbone_args` 和 `input_dim`。

## 11. YAML 配置格式说明

这里以 `opencood/hypes_yaml/v2xreal/GenComm_yamls/gencomm/stage1/m1_att.yaml` 为例说明配置文件的结构。这个仓库几乎所有训练和推理行为都由 YAML 驱动，所以读懂 YAML 比直接看入口脚本更重要。

### 1. 顶层实验信息

```yaml
name: V2XReal_GenComm_stage1_m1_att
root_dir: "/data/bohnsix/datasets/v2xreal/train"
validate_dir: "/data/bohnsix/datasets/v2xreal/test"
test_dir: "/data/bohnsix/datasets/v2xreal/test"

dataset_mode: "vc"
yaml_parser: "load_general_params"
```

这些字段决定实验名、数据路径和 YAML 解析方式。

- `name`：实验名称，训练时会用于日志目录和 SwanLab/TensorBoard run 名称。
- `root_dir`：训练集路径。
- `validate_dir`：验证集路径。
- `test_dir`：测试集路径。部分脚本会读这个字段，虽然 V2X-Real 的 dataset 基类主要区分 `root_dir` 和 `validate_dir`。
- `dataset_mode`：V2X-Real 的场景模式，当前是 `vc`。
- `yaml_parser`：指定 YAML 后处理函数，一般保持默认即可。

### 2. 训练参数

```yaml
train_params:
  batch_size: &batch_size 1
  epoches: 20
  eval_freq: 2
  save_freq: 2
  max_cav: 5
```

- `batch_size`：每个 DataLoader batch 的样本数。当前为了降低显存，已经设为 1。
- `&batch_size`：YAML 锚点，表示把这个值命名为 `batch_size`，后面可以用 `*batch_size` 引用。
- `epoches`：训练轮数，仓库里拼写就是 `epoches`，不要改成 `epochs`。
- `eval_freq`：每隔多少个 epoch 做一次验证。
- `save_freq`：每隔多少个 epoch 保存一次普通 checkpoint。
- `max_cav`：一个样本中最多使用多少个协作智能体。V2X-Real 场景里常设为 5。

### 3. 基础感知范围和输入类型

```yaml
comm_range: 70
input_source: ['lidar']
label_type: 'lidar'
cav_lidar_range: &cav_lidar [-102.4, -51.2, -15, 102.4, 51.2, 15]
```

- `comm_range`：通信范围，超出范围的协作方不会参与融合。
- `input_source`：输入源。当前 V2X-Real GenComm stage1 只使用 LiDAR。
- `label_type`：标签坐标系/标签来源，当前是 LiDAR。
- `cav_lidar_range`：点云空间范围，格式是 `[x_min, y_min, z_min, x_max, y_max, z_max]`。
- `&cav_lidar`：把这个范围保存成锚点，后面用 `*cav_lidar` 复用，避免多个地方写不一致。

### 4. heter 异构配置

```yaml
heter:
  assignment_path: "opencood/modality_assign/v2xreal_4modality.json"
  ego_modality: &ego_modality "m1"
  mapping_dict:
    m1: m1
    m2: m1
    m3: m1
    m4: m1
  modality_setting:
    m1:
      sensor_type: &sensor_type_m1 'lidar'
      core_method: &core_method_m1 "point_pillar"
```

这一段定义“当前实验有哪些智能体类型，以及它们如何映射”。

- `assignment_path`：场景中每个 CAV/RSU 对应 `m1/m2/m3/m4` 的分配表。
- `ego_modality`：当前训练的 ego 模态。`m1_att.yaml` 里是 `m1`。
- `mapping_dict`：把 assignment 文件里的模态映射到当前实验实际使用的模态。stage1 中通常把所有模态都映射到当前 ego，例如全部映射成 `m1`。
- `modality_setting`：每种模态的传感器、检测器和预处理配置。

在当前 V2X-Real stage1 配置中，`m1-m4` 都是 LiDAR + PointPillar，不是跨 camera/lidar 的差异，主要靠 backbone 深度制造异构性。

### 5. modality_setting 中的 preprocess

```yaml
preprocess:
  core_method: 'SpVoxelPreprocessor'
  args:
    voxel_size: &voxel_size [0.4, 0.4, 30]
    max_points_per_voxel: 32
    max_voxel_train: 64000
    max_voxel_test: 640000
  cav_lidar_range: *cav_lidar
```

这是单个模态内部的点云预处理配置。

- `core_method: SpVoxelPreprocessor`：使用稀疏 voxel 预处理器。
- `voxel_size`：体素大小。当前 z 方向设为 30，说明主要在 BEV 平面划分网格。
- `max_points_per_voxel`：每个 voxel 最多保留多少点。
- `max_voxel_train`：训练时最多保留多少 voxel。
- `max_voxel_test`：测试时最多保留多少 voxel。
- `cav_lidar_range: *cav_lidar`：引用顶层定义的 LiDAR 范围。

### 6. fusion 数据集融合配置

```yaml
fusion:
  core_method: 'intermediateheterv2xreal'
  dataset: 'v2xreal'
  args:
    proj_first: false
    grid_conf: None
    data_aug_conf: None
```

这一段控制 dataset builder 会实例化哪个数据集类。

- `core_method: intermediateheterv2xreal`：使用 V2X-Real 专用的异构中间融合 dataset。
- `dataset: v2xreal`：指定基础数据集类是 V2X-Real。
- `proj_first`：是否先做投影再融合，当前关闭。
- `grid_conf` / `data_aug_conf`：占位字段，在 LiDAR-only 配置中通常不实际使用。

### 7. data_augment 数据增强

```yaml
data_augment:
  - NAME: random_world_flip
    ALONG_AXIS_LIST: [ 'x' ]
  - NAME: random_world_rotation
    WORLD_ROT_ANGLE: [ -0.78539816, 0.78539816 ]
  - NAME: random_world_scaling
    WORLD_SCALE_RANGE: [ 0.95, 1.05 ]
```

这是训练时的数据增强列表。YAML 中 `-` 表示 list 元素。

当前包括：

- 沿 x 轴随机翻转
- 随机旋转，范围约为正负 45 度
- 随机缩放，范围是 0.95 到 1.05

### 8. 顶层 preprocess 和 anchor 配置

```yaml
preprocess:
  core_method: 'SpVoxelPreprocessor'
  args:
    voxel_size: [0.4, 0.4, 30]
    max_points_per_voxel: 32
    max_voxel_train: 64000
    max_voxel_test: 640000
  cav_lidar_range: *cav_lidar
  num_class: &num_class 3
  anchor_generator_config: &anchor_generator_config
    - 'class_name': 'vehicle'
      'anchor_sizes': [ [ 3.9, 1.6, 1.56 ] ]
```

这里的顶层 `preprocess` 既包含预处理器配置，也包含 anchor 生成需要的类别信息。

- `num_class: 3`：V2X-Real 当前检测三类。
- `anchor_generator_config`：每个类别的 anchor 尺寸、旋转、匹配阈值。
- `vehicle / pedestrian / truck`：三个检测类别。
- `feature_map_stride`：anchor 对应的特征图下采样倍率。
- `matched_threshold / unmatched_threshold`：正负样本匹配阈值。

注意，文件中注释写了部分字段 `useful/useless`，这是因为某些值实际由模态内部 preprocess 使用，顶层这里更多是为了兼容 postprocessor 和旧代码接口。

### 9. postprocess 后处理配置

```yaml
postprocess:
  core_method: 'VoxelPostprocessor'
  gt_range: *cav_lidar
  anchor_args:
    cav_lidar_range: *cav_lidar
    r: &anchor_yaw [0, 90]
    num: &anchor_num 2
    anchor_generator_config: *anchor_generator_config
  target_args:
    pos_threshold: 0.6
    neg_threshold: 0.45
    score_threshold: 0.2
  order: 'hwl'
  max_num: 150
  nms_thresh: 0.15
  dir_args: &dir_args
    dir_offset: 0.7853
    num_bins: 2
    anchor_yaw: *anchor_yaw
```

这部分决定标签生成、预测框解码和 NMS。

- `VoxelPostprocessor`：使用 voxel 检测后处理器。
- `gt_range`：GT 框有效范围。
- `anchor_args`：anchor 生成参数。
- `anchor_yaw`：anchor 朝向。
- `anchor_num`：每个位置的 anchor 数量。
- `target_args`：正负样本和置信度阈值。
- `order: hwl`：框尺寸顺序。
- `max_num`：单帧最多保留多少目标，保证 batch 内尺寸一致。
- `nms_thresh`：NMS 阈值。
- `dir_args`：方向分类头配置。

### 10. model 模型配置

```yaml
model:
  core_method: heter_model_baseline_w_gencomm_stage1
  args:
    ego_modality: *ego_modality
    lidar_range: *cav_lidar
```

`model.core_method` 是模型分发的关键。它会被 `train_utils.create_model` 解析成：

```text
opencood.models.heter_model_baseline_w_gencomm_stage1
```

然后代码会在该文件中寻找类名：

```text
HeterModelBaselineWGenCommStage1
```

所以文件名、类名和 YAML 字段必须匹配。

### 11. model.args.m1 编码器和 backbone

```yaml
m1:
  core_method: *core_method_m1
  sensor_type: *sensor_type_m1
  encoder_args:
    voxel_size: *voxel_size
    lidar_range: *cav_lidar
    pillar_vfe:
      use_norm: true
      with_distance: false
      use_absolute_xyz: true
      num_filters: [64]
    point_pillar_scatter:
      num_features: 64
```

这段是 `m1` 的 PointPillar encoder 配置。

- `pillar_vfe`：pillar 特征编码器，把点云 pillar 编码成局部特征。
- `num_filters: [64]`：pillar 特征输出通道。
- `point_pillar_scatter`：把 pillar 特征散射回 BEV 特征图。
- `num_features: 64`：scatter 后的 BEV 通道数。

backbone 配置如下：

```yaml
backbone_args:
  layer_nums: [3, 5, 8]
  layer_strides: [2, 2, 2]
  num_filters: [64, 128, 256]
  upsample_strides: [1, 2, 4]
  num_upsample_filter: [128, 128, 128]
```

含义是三层 BEV backbone，每层卷积块数量分别是 3、5、8，输出通道分别是 64、128、256，再通过三路上采样统一到可拼接尺度。

```yaml
shrink_header:
  kernal_size: [ 3 ]
  stride: [ 2 ]
  padding: [ 1 ]
  dim: [ 256 ]
  input_dim: 384
```

`input_dim: 384` 来自三路上采样特征拼接：`128 + 128 + 128 = 384`。shrink header 会把拼接后的特征压到 `256` 通道。

### 12. GenComm 模块配置

```yaml
enhancer:
  in_ch: 256
message_extractor:
  in_ch: 256
  out_ch: 2

gencomm:
  model:
    embed_dim: 258
    in_channels: 256
    out_ch: 256
    ch: 8
    ch_mult: [1, 1]
    num_res_blocks: 2
    attn_resolutions: [16]
    dropout: 0.0
    resamp_with_conv: True
  diffusion:
    beta_schedule: linear
    beta_start: 0.0005
    beta_end: 0.02
    num_diffusion_timesteps: 3
```

这部分是 GenComm 的核心。

- `message_extractor.in_ch: 256`：从 256 通道 BEV 特征中提取消息。
- `message_extractor.out_ch: 2`：消息通道数为 2。
- `gencomm.model.in_channels: 256`：输入特征通道。
- `gencomm.model.embed_dim: 258`：通常可以理解为特征通道 256 加上消息条件 2。
- `gencomm.model.out_ch: 256`：生成/重建后的特征仍是 256 通道。
- `ch / ch_mult / num_res_blocks / attn_resolutions`：扩散 U-Net/去噪网络的结构超参数。
- `num_diffusion_timesteps: 3`：扩散步数，当前设置很小，训练和推理更轻。

### 13. 融合和检测头配置

```yaml
fusion_method: att
att:
  feat_dim: 256

in_head: 256
num_class: *num_class
anchor_number: *anchor_num
dir_args: *dir_args
```

- `fusion_method: att`：使用 attention fusion。
- `att.feat_dim: 256`：融合模块输入特征维度。
- `in_head: 256`：检测头输入通道。
- `num_class`：引用前面的 `&num_class`，当前为 3。
- `anchor_number`：引用前面的 `&anchor_num`，当前为 2。
- `dir_args`：引用方向分类配置。

这些字段会直接决定 `cls_head / reg_head / dir_head` 的输出通道数。

### 14. loss 损失配置

```yaml
loss:
  core_method: point_pillar_v2xreal_gencomm_loss
  args:
    cls_weight: 1.0
    reg: 2.0
    num_class: *num_class
    generate_weight: 1
```

损失函数同样是动态分发的。

- `loss.core_method` 会导入 `opencood.loss.point_pillar_v2xreal_gencomm_loss`。
- `cls_weight`：分类损失权重。
- `reg`：回归损失权重。
- `num_class`：类别数。
- `generate_weight`：GenComm 生成/重建特征损失的权重。

### 15. optimizer 和 lr_scheduler

```yaml
optimizer:
  core_method: Adam
  lr: 0.002
  args:
    eps: 1e-10
    weight_decay: 1e-4

lr_scheduler:
  core_method: multistep
  gamma: 0.1
  step_size: [10, 15]
```

- `optimizer.core_method: Adam`：使用 Adam 优化器。
- `lr: 0.002`：初始学习率。
- `eps`：Adam 的数值稳定项。
- `weight_decay`：权重衰减。
- `lr_scheduler.core_method: multistep`：多阶段学习率衰减。
- `gamma: 0.1`：每次衰减乘以 0.1。
- `step_size: [10, 15]`：第 10、15 个 epoch 调整学习率。

### 16. YAML 语法要点

这个仓库里的 YAML 大量使用以下语法：

```yaml
key: value
parent:
  child: value
list:
  - item1
  - item2
anchor_value: &some_name 123
reuse_value: *some_name
```

需要特别注意：

- 缩进表示层级，不能乱用 tab。
- `&name` 是锚点定义。
- `*name` 是引用锚点。
- `#` 后面是注释。
- 方括号 `[1, 2, 3]` 是行内 list。
- `- NAME: xxx` 是 list 中的一个 dict。
- 字符串可以带引号，也可以不带，但路径建议带引号。

### 17. 最容易改错的字段

改配置时优先小心这些字段：

- `model.core_method`：会影响模型文件和类名分发。
- `loss.core_method`：会影响损失类分发。
- `fusion.core_method` 和 `fusion.dataset`：会影响 dataset 类分发。
- `ego_modality`：会影响当前训练的是哪个模态基座。
- `mapping_dict`：会影响场景中的智能体被映射到哪种模型结构。
- `shrink_header.input_dim`：必须和 backbone 多尺度拼接后的通道数一致。
- `in_head` / `att.feat_dim` / `message_extractor.in_ch`：必须和 shrink 后的 BEV 特征通道一致。
- `num_class` 和 anchor 配置：必须和数据集类别一致。

## 12. 当前工作建议

如果继续做 V2X-Real 训练，建议按这个顺序：

1. 确认 `config.yaml` 中的路径已经改对
2. 确认 stage1 / stage2 的类名匹配已经修复
3. 先跑 `m1` 的 stage1，确认数据加载和前向正常
4. 再跑 `m2`、`m3`、`m4`
5. 最后进入 stage2 合并和训练
6. 训练完成后再走 V2X-Real 的专用推理脚本

