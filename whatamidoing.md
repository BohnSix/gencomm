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

## 12. HeterModelBaselineWGenCommStage1 模型详解

`HeterModelBaselineWGenCommStage1` 位于 `opencood/models/heter_model_baseline_w_gencomm_stage1.py`，是 GenComm stage1 使用的主模型类。它不是一个单纯的 encoder，而是一个完整的协同感知检测模型：从 LiDAR 点云编码，到 BEV backbone，到消息提取、GenComm 特征生成、协同融合，最后输出检测头预测。

### 1. 这个模型解决什么问题

stage1 的目标是训练某一种智能体类型的同构协同基座。以 V2X-Real 的 `m1_att.yaml` 为例，`mapping_dict` 会把 `m1/m2/m3/m4` 都映射成 `m1`，所以这一轮训练中所有协作方都按同一个 `m1` 特征提取结构处理。

模型要学到三件事：

- 如何从单车 LiDAR 点云提取 BEV 特征
- 如何从 BEV 特征中抽取低维通信消息
- 如何用 GenComm 根据消息生成/重建用于融合的协作特征

### 2. 初始化入口

类定义是：

```python
class HeterModelBaselineWGenCommStage1(nn.Module):
    def __init__(self, args):
```

这里的 `args` 来自 YAML 的：

```yaml
model:
  args:
```

也就是说，YAML 中 `model.args` 下的字段会直接决定这个模型搭哪些模块。

最先初始化的是：

```python
self.args = args
self.gencomm = GenComm(args['gencomm'])
self.missing_message = args.get('missing_message', False)
```

含义：

- `self.gencomm` 是条件扩散生成模块，负责根据消息条件生成/重建 BEV 特征。
- `missing_message` 是推理阶段的特殊开关，用来模拟协作消息缺失。

### 3. 自动发现 m1/m2/m3/m4 模态

模型会扫描 `args` 中所有形如 `m数字` 的 key：

```python
modality_name_list = list(args.keys())
modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
self.modality_name_list = modality_name_list
```

例如 stage1 `m1_att.yaml` 里只有：

```yaml
model:
  args:
    m1:
      ...
```

那么 `self.modality_name_list = ['m1']`。

stage2 的 `m1m2_att.yaml` 里会有 `m1` 和 `m2` 两组配置，那么列表就是 `['m1', 'm2']`。

### 4. 每个模态会构建哪些模块

初始化时会对 `self.modality_name_list` 逐个循环：

```python
for modality_name in self.modality_name_list:
    model_setting = args[modality_name]
```

每个模态都会构建四类模块：

- encoder
- backbone
- shrinker
- message extractor

#### encoder

encoder 通过反射从 `opencood.models.heter_encoders` 中加载：

```python
encoder_filename = "opencood.models.heter_encoders"
encoder_lib = importlib.import_module(encoder_filename)
target_model_name = model_setting['core_method'].replace('_', '')
```

例如 YAML 里：

```yaml
core_method: point_pillar
```

模型会在 `heter_encoders.py` 里寻找类名忽略大小写后等于 `pointpillar` 的 encoder 类。

构建后挂到模型上：

```python
setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
```

对于 `m1`，实际就是：

```python
self.encoder_m1 = PointPillar(...)
```

#### backbone

backbone 有两种情况：

```python
if model_setting['backbone_args'] == 'identity':
    setattr(self, f"backbone_{modality_name}", nn.Identity())
else:
    setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(...))
```

所以：

- `m1/m2/m3` 会构建不同深度的 `BaseBEVBackbone`
- `m4` 会使用 `nn.Identity()`，基本跳过 backbone

这也是 `m1 > m2 > m3 > m4` 显存差异的主要来源。

#### shrinker

```python
setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))
```

shrinker 的作用是把 backbone 多尺度拼接后的特征压到统一通道数。比如 `m1` 的 backbone 三路上采样拼接后是 384 通道，`shrink_header` 会把它变成 256 通道。

#### message extractor

```python
setattr(
    self,
    f"message_extractor_{modality_name}",
    MessageExtractorv2(args['message_extractor']['in_ch'], args['message_extractor']['out_ch'])
)
```

`MessageExtractorv2` 位于 `opencood/models/gencomm_modules/message_extractor_v2.py`。当前实现内部使用 `BEVDeformableExtractor`：

- 先用卷积预测 deformable convolution 的 offset
- 再用 `DeformConv2d` 提取局部可变形特征
- 通过一个通道注意力模块增强特征
- 最后压到 `out_ch` 个消息通道

在当前 YAML 中：

```yaml
message_extractor:
  in_ch: 256
  out_ch: 2
```

因此消息是从 256 通道 BEV 特征中提取出的 2 通道条件图。

### 5. 坐标变换相关参数

模型保存了 BEV 范围：

```python
self.cav_range = args['lidar_range']
self.H = (self.cav_range[4] - self.cav_range[1])
self.W = (self.cav_range[3] - self.cav_range[0])
self.fake_voxel_size = 1
```

forward 中会调用：

```python
affine_matrix = normalize_pairwise_tfm(
    data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size
)
```

这个 `affine_matrix` 用于把不同智能体的 BEV 特征对齐到 ego 坐标系，是后续 enhancer 和 fusion 的几何基础。

### 6. 融合模块选择

模型通过 YAML 中的：

```yaml
fusion_method: att
```

选择融合模块。

代码中支持：

- `max` -> `MaxFusion`
- `att` -> `AttFusion`
- `disconet` -> `DiscoFusion`
- `v2vnet` -> `V2VNetFusion`
- `v2xvit` -> `V2XViTFusion`
- `cobevt` -> `CoBEVT`
- `where2comm` -> `Where2commFusion`
- `who2com` -> `Who2comFusion`

当前 V2X-Real GenComm stage1 使用的是：

```yaml
fusion_method: att
att:
  feat_dim: 256
```

所以实际构建的是：

```python
self.fusion_net = AttFusion(args['att']['feat_dim'])
```

### 7. 检测头

检测头是共享的，不区分模态：

```python
self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'] * self.num_class * self.num_class, kernel_size=1)
self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'] * self.num_class, kernel_size=1)
self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1)
```

在当前 V2X-Real 配置中：

- `in_head = 256`
- `anchor_number = 2`
- `num_class = 3`
- `dir_args.num_bins = 2`

因此检测头输出包括：

- `cls_preds`：分类预测
- `reg_preds`：3D 框回归预测
- `dir_preds`：方向分类预测

### 8. enhancer

如果 YAML 里有：

```yaml
enhancer:
  in_ch: 256
```

就会构建：

```python
self.enhancer = Enhancer(self.args['enhancer']['in_ch'], [8, 8], 4)
```

它位于 `opencood/models/gencomm_modules/enhancer.py`，作用是在 GenComm 生成特征后，结合几何变换和协作关系进一步增强 BEV 特征。

### 9. compressor 可选分支

如果 YAML 中有 `compressor` 字段，会启用：

```python
self.compress = True
self.compressor = NaiveCompressor(...)
self.model_train_init()
```

`model_train_init()` 会冻结所有参数，只开放 compressor 训练：

```python
self.eval()
for p in self.parameters():
    p.requires_grad_(False)
self.compressor.train()
for p in self.compressor.parameters():
    p.requires_grad_(True)
```

当前 V2X-Real stage1 配置没有使用这个分支。

### 10. forward 输入数据

forward 的签名是：

```python
def forward(self, data_dict):
```

这里的 `data_dict` 不是原始 dataset item，而是经过 dataset collate 和 `train_utils.to_device` 后的 `batch_data['ego']`。

forward 一开始取出：

```python
agent_modality_list = data_dict['agent_modality_list']
pairwise_t_matrix = data_dict['pairwise_t_matrix']
record_len = data_dict['record_len']
```

这些字段含义：

- `agent_modality_list`：当前 batch 中每个智能体对应哪个模态，比如 `['m1', 'm1', 'm1']`。
- `pairwise_t_matrix`：智能体之间的坐标变换矩阵。
- `record_len`：每个 batch 样本中有多少个智能体，用来把扁平化的特征重新分组。

### 11. forward 第一步：逐模态提取特征和消息

代码先统计 batch 中出现了哪些模态：

```python
modality_count_dict = Counter(agent_modality_list)
```

然后逐模态处理：

```python
feature = self.encoder_m1(data_dict, 'm1')
feature = self.backbone_m1({'spatial_features': feature})['spatial_features_2d']
feature = self.shrinker_m1(feature)
message = self.message_extractor_m1(feature)
```

抽象成流程就是：

```text
点云 -> PointPillar encoder -> BEV backbone -> shrinker -> 256通道BEV特征
256通道BEV特征 -> MessageExtractorv2 -> 2通道消息条件
```

结果保存到：

```python
modality_feature_dict[modality_name] = feature
modality_message_dict[modality_name] = message
```

### 12. camera 分支兼容逻辑

模型里有一段 camera crop/padding 逻辑：

```python
if self.sensor_type_dict[modality_name] == "camera":
    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
    modality_feature_dict[modality_name] = crop_func(feature)
```

当前 V2X-Real GenComm stage1 都是 LiDAR，所以这段通常不会走。但模型保留了 camera 分支，是因为这个文件来自一个统一的 LiDAR/Camera/heterogeneous 框架。

### 13. forward 第二步：按智能体顺序重新组装特征

前面是按模态批量处理的，后面要恢复到智能体顺序：

```python
for modality_name in agent_modality_list:
    feat_idx = counting_dict[modality_name]
    heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
    heter_message_list.append(modality_message_dict[modality_name][feat_idx])
```

最后得到：

```python
heter_feature_2d = torch.stack(heter_feature_2d_list)
heter_message = torch.stack(heter_message_list)
```

可以理解为：

```text
heter_feature_2d: 所有智能体的 BEV 特征，按样本/智能体顺序排列
heter_message: 所有智能体的通信消息，按相同顺序排列
```

### 14. missing_message 推理分支

如果不是训练模式，并且 `missing_message=True`，模型会随机屏蔽一部分非 ego 消息：

```python
for i in range(1, heter_message.shape[0]):
    mask = torch.rand(...) > 0.4
    heter_message[i] = heter_message[i] * mask
```

这用于模拟协作消息缺失或通信不完整的推理场景。训练时不会触发。

### 15. forward 第三步：GenComm 生成/重建特征

核心调用是：

```python
conditions = heter_message
gt_feature = heter_feature_2d
gen_data_dict = self.gencomm(heter_feature_2d, conditions, record_len)
pred_feature = gen_data_dict['pred_feature']
```

这里的含义是：

- `gt_feature`：真实提取到的 BEV 特征，用作生成目标或监督信号。
- `conditions`：message extractor 提取出的低维条件消息。
- `pred_feature`：GenComm 根据条件消息生成/重建出来的 BEV 特征。

随后模型直接用生成特征替换原特征：

```python
heter_feature_2d = pred_feature
```

并把真实和预测特征都放进输出：

```python
output_dict.update({
    'gt_feature': gt_feature,
    'pred_feature': pred_feature
})
```

损失函数可以利用这两个字段计算 GenComm 的生成损失。

### 16. GenComm 内部做了什么

`GenComm` 位于 `opencood/models/gencomm_modules/cond_diff.py`。

它是一个条件扩散模块。核心逻辑是：

- 输入真实 BEV 特征 `spatial_features`
- 输入条件消息 `conditions`
- 对真实特征加噪
- 用条件消息引导 `DiffusionUNet` 去噪
- 输出重建后的 `pred_feature`

训练时，`GenComm.forward` 会：

1. 用 `record_len` 把 batch 按场景拆开
2. 取每个场景 ego 的特征并复制到该场景所有智能体数量
3. 对目标特征加噪
4. 从最大扩散步开始反向采样
5. 输出 `pred_feature`

它的去噪网络调用形式是：

```python
model_out = self.denoiser(torch.cat([feat, noisy_masks], dim=1), t.float())
```

这里 `feat` 是条件，`noisy_masks` 是当前带噪特征。两者在通道维拼接后送入 U-Net。

当前 YAML 中：

```yaml
gencomm:
  model:
    embed_dim: 258
    in_channels: 256
    out_ch: 256
  diffusion:
    num_diffusion_timesteps: 3
```

可以理解为：用 2 通道消息条件，引导扩散模型生成 256 通道 BEV 特征。

### 17. forward 第四步：enhancer 和 fusion

GenComm 输出后，如果特征维度退化成 3D，会补一个 batch 维：

```python
if len(heter_feature_2d.shape) == 3:
    heter_feature_2d = heter_feature_2d.unsqueeze(0)
```

然后进入 enhancer：

```python
if hasattr(self, 'enhancer'):
    heter_feature_2d = self.enhancer(heter_feature_2d, affine_matrix, record_len)
```

接着进入融合模块：

```python
fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)
```

这里 `record_len` 和 `affine_matrix` 很关键：

- `record_len` 告诉 fusion 每个 batch 样本有几个智能体。
- `affine_matrix` 告诉 fusion 如何把协作智能体特征变换到 ego 坐标。

### 18. forward 第五步：检测头输出

融合后的 ego BEV 特征进入检测头：

```python
cls_preds = self.cls_head(fused_feature)
reg_preds = self.reg_head(fused_feature)
dir_preds = self.dir_head(fused_feature)
```

最终输出字典：

```python
output_dict.update({
    'cls_preds': cls_preds,
    'reg_preds': reg_preds,
    'dir_preds': dir_preds,
    'message': conditions
})
```

训练损失会主要读取：

- `cls_preds`
- `reg_preds`
- `dir_preds`
- `gt_feature`
- `pred_feature`
- `message`

### 19. 完整数据流总结

可以把 `HeterModelBaselineWGenCommStage1` 的 forward 看成下面这条链路：

```text
batch_data['ego']
  -> 按 agent_modality_list 判断有哪些模态
  -> 每个模态走 encoder
  -> 每个模态走 BEV backbone 或 identity
  -> shrinker 统一到 256 通道 BEV 特征
  -> MessageExtractorv2 提取 2 通道消息
  -> 按智能体顺序 stack 成 heter_feature_2d / heter_message
  -> GenComm 用 heter_message 生成 pred_feature
  -> pred_feature 替换原始 heter_feature_2d
  -> Enhancer 几何增强
  -> AttFusion / 其他 fusion_net 协同融合
  -> cls/reg/dir 检测头
  -> output_dict
```

### 20. 这个模型里最容易出问题的地方

1. `model.core_method` 和类名必须匹配。

`heter_model_baseline_w_gencomm_stage1` 必须对应 `HeterModelBaselineWGenCommStage1`。否则会报 “backbone not found”，但真实原因是模型类没被分发器找到。

2. `shrink_header.input_dim` 必须匹配 backbone 输出拼接维度。

例如 `m1` 是三路 `128 + 128 + 128 = 384`，所以 `input_dim` 必须是 384。`m2` 是 256，`m3` 是 128，`m4` 是 64。

3. `message_extractor.in_ch` 必须匹配 shrinker 输出通道。

当前 shrinker 输出 256，因此 message extractor 输入也应是 256。

4. `gencomm.model.in_channels` 和检测头 `in_head` 要与 BEV 特征通道一致。

当前都是 256。

5. `record_len` 和 `agent_modality_list` 必须正确。

它们决定特征如何从“按模态处理”重新排列回“按智能体处理”。如果这里错了，融合对象和坐标变换会对不上。

## 13. 当前工作建议

如果继续做 V2X-Real 训练，建议按这个顺序：

1. 确认 `config.yaml` 中的路径已经改对
2. 确认 stage1 / stage2 的类名匹配已经修复
3. 先跑 `m1` 的 stage1，确认数据加载和前向正常
4. 再跑 `m2`、`m3`、`m4`
5. 最后进入 stage2 合并和训练
6. 训练完成后再走 V2X-Real 的专用推理脚本

## 14. V2X-Real GenComm stage1：训练完成但自动推理因 cuDNN 失败

### 14.1 本次运行的事实

用户提供的日志显示，V2X-Real GenComm stage1 的训练已经完成到 `[epoch 19][2826/2826]`，随后打印了 `Training Finished, checkpoints saved to ...`。这说明训练循环已经跑完，并且日志目录已经生成 checkpoint；它不等于后续推理或模型质量验证成功。

[train.py:131-172](opencood/tools/train.py#L131-L172) 负责 epoch 循环、保存普通 checkpoint 和验证 checkpoint。[train.py:216-224](opencood/tools/train.py#L216-L224) 中 `run_test = True`，所以训练结束后会通过 `os.system` 自动启动 V2X-Real 推理。

本次自动推理的后续行为是：

1. [inference_v2xreal.py:95-124](opencood/tools/inference_v2xreal.py#L95-L124) 创建模型、加载 checkpoint 并构建测试数据集。
2. 推理加载的是 `net_epoch_bestval_at15.pth`，即按验证损失选择的 epoch 15 checkpoint；目标目录为 `opencood/logs/GenComm/v2xreal/stage1/OPV2V_m1_att`。
3. [inference_v2xreal.py:142-179](opencood/tools/inference_v2xreal.py#L142-L179) 在第一个 batch 上调用中间融合推理。
4. [heter_model_baseline_w_gencomm_stage1.py:260-283](opencood/models/heter_model_baseline_w_gencomm_stage1.py#L260-L283) 执行 encoder 和 BEV backbone 时，在 [base_bev_backbone.py:40-58](opencood/models/sub_modules/base_bev_backbone.py#L40-L58) 的 `Conv2d` 处抛出 `RuntimeError: Unable to find a valid cuDNN algorithm to run convolution`。

因此，本次运行目前**没有完成完整 V2X-Real 推理，也没有产生可用于报告的 AP 指标**。在推理成功前，不能仅根据训练 loss 判断最终检测质量。

日志目录中可以看到 `config.yaml`、`net_epoch1.pth` 至 `net_epoch19.pth` 和 `net_epoch_bestval_at15.pth`。使用 `--model_dir` 推理时应以该日志目录中的 `config.yaml` 为准；用户日志中实际打印的数据目录为 `/data2/bohnsix/v2xreal//test`，而当前源 YAML 中的路径是 `/data/bohnsix/datasets/v2xreal/test`，两者并不完全一致。后续复现时应同时核对日志目录配置、当前源码和实际数据路径。

### 14.2 训练配置和损失解释

本次 m1 配置的关键参数见 [m1_att.yaml:9-14](opencood/hypes_yaml/v2xreal/GenComm_yamls/gencomm/stage1/m1_att.yaml#L9-L14)：`batch_size=1`、总训练轮数 `20`、每 `2` 个 epoch 验证和保存一次、`max_cav=5`。m1 的 BEV backbone 见 [m1_att.yaml:152-164](opencood/hypes_yaml/v2xreal/GenComm_yamls/gencomm/stage1/m1_att.yaml#L152-L164)：三层 block 的数量为 `[3, 5, 8]`，stride 为 `[2, 2, 2]`，通道为 `[64, 128, 256]`，三路上采样 stride 为 `[1, 2, 4]`，多尺度拼接后输入 shrinker 的通道为 `384`。

GenComm 配置见 [m1_att.yaml:172-187](opencood/hypes_yaml/v2xreal/GenComm_yamls/gencomm/stage1/m1_att.yaml#L172-L187)：输入/输出特征通道为 `256`，消息通道为 `2`，diffusion timestep 为 `3`。优化器和调度器见 [m1_att.yaml:202-220](opencood/hypes_yaml/v2xreal/GenComm_yamls/gencomm/stage1/m1_att.yaml#L202-L220)：Adam 初始学习率为 `0.002`，weight decay 为 `1e-4`，MultiStep 在 `[10, 15]` 处按 `gamma=0.1` 衰减。

损失实现见 [point_pillar_v2xreal_gencomm_loss.py:74-166](opencood/loss/point_pillar_v2xreal_gencomm_loss.py#L74-L166)。当前配置中 `reg=2.0`、`cls_weight=1.0`、`generate_weight=1`，实际关系是：

```text
total_loss = reg_loss + conf_loss + generate_weight * generate_loss
```

其中 `reg_loss` 已经乘过配置中的回归权重 `2.0`，所以日志中的 `Loc Loss` 不是未加权的原始回归损失。用户给出的 epoch 19 后段日志大致表现为：

- `Conf Loss`：约 `0.59–0.89`
- `Loc Loss`：约 `1.0–3.6`
- `Gen Loss`：约 `0.008–0.018`
- `Loss`：约 `1.8–4.3`

这些数值没有显示出训练在 epoch 19 的最后几十个 batch 发生异常发散；但单个 batch loss 波动是正常的，不能单独证明收敛或检测性能。

TensorBoard 日志目录包含多个 event 文件。读取这些文件时观察到 `Validate_Loss` 大约在 `5.55–6.47` 的范围，最近 event 文件的最后值约为 `5.50`，并且当前目录的最佳验证 checkpoint 是 `net_epoch_bestval_at15.pth`。由于多个 event 文件可能来自不同运行、续训或不同 writer，必须按 event 文件、step 和运行时间分别核对；不能把这些文件未经区分地当作一条连续实验曲线。

### 14.3 对训练结果的客观判断

可以确认的结论：

- 训练 loop 已完成 epoch 19，并且 checkpoint 已保存。
- 训练结束后的自动推理确实被启动。
- 推理在第一个样本的 BEV backbone 卷积阶段失败，尚未进入完整的 GenComm、fusion、检测头和评测流程。

当前不能确认的结论：

- 不能确认完整推理能正常运行。
- 不能确认 V2X-Real 的 AP@0.3/0.5/0.7。
- 不能仅凭 `Gen Loss` 较小就确认 GenComm 生成质量良好。
- 不能把训练结束信息解释为模型已经通过测试。

### 14.4 cuDNN 错误的分层分析

**已观察事实**：错误发生在 GPU 上为 `Conv2d` 选择 cuDNN 算法的阶段。当前证据首先指向运行环境、GPU 架构支持、显存状态或 CUDA/cuDNN 兼容性，而不是 loss 计算本身。

**环境兼容性风险**：此前检查到运行 GPU 为 NVIDIA RTX 4090，计算能力为 `sm_89`；当前环境为 PyTorch `1.13.1+cu117`，而 `torch.cuda.get_arch_list()` 观察到的最高架构为 `sm_86`。这说明当前旧版 PyTorch/cuDNN 组合对 RTX 4090 的支持存在风险，可能导致 CUDA kernel 或 cuDNN 算法选择不完整。这个信息是高优先级线索，但不是仅凭错误文本即可完全证明的唯一根因。

**显存和模型规模风险**：m1 使用四种 stage1 基座中较深的 backbone，配置的单样本最大协作智能体数为 `5`。`batch_size=1` 已经降低了 batch 维度的显存压力，但不能排除中间特征、可用显存不足、显存碎片或其他进程占用。当前错误文本没有明确写出 CUDA out-of-memory，因此不应直接把它等同于普通 OOM。

**仍需验证的因素**：还需要检查首个 batch 的 shape、dtype、device 和 finite 状态，确认 checkpoint 加载时的 missing/extra keys，并核对日志目录中的 `config.yaml`、实际运行代码和当前 checkout 是否属于同一份实验来源。当前日志已经越过了模型创建阶段，所以这些 provenance 风险不能直接被认定为本次 cuDNN 错误的原因。

### 14.5 推荐的排查和恢复顺序

1. **先记录实际运行环境**：使用 `nvidia-smi` 和 PyTorch 查询 GPU 型号、GPU index、`CUDA_VISIBLE_DEVICES` 映射、其他占用进程、总/已用/剩余显存、PyTorch/CUDA/cuDNN 版本和 `torch.cuda.get_arch_list()`。重点确认推理实际运行在哪张 GPU 上。
2. **做临时 cuDNN 诊断**：使用同一 checkpoint、同一份日志目录 `config.yaml` 和同一个首样本，临时禁用 cuDNN 后复测，并记录是否能够越过 backbone 卷积。如果禁用后可以继续，环境/cuDNN 算法选择问题的可能性会增加；但禁用 cuDNN 只是诊断或临时运行手段，**不是模型修复**。
3. **选择正式环境修复路径**：优先使用明确支持 `sm_89` 的较新 PyTorch/CUDA/cuDNN 组合，并重新核对 torchvision、spconv 和本仓库自定义 CUDA 扩展；或者改用当前旧环境明确支持的 GPU。如果确认是显存问题，先清理其他进程并重新确认可用显存。
4. **保持实验可比性**：不要仅因为 cuDNN 错误就修改 `reg`、`generate_weight`、diffusion timesteps 或 backbone 结构，这些修改会改变实验本身并使现有 checkpoint 不再可直接比较。
5. **环境修复后完成验证**：检查 `net_epoch_bestval_at15.pth` 的 key 加载情况，检查首 batch 的 shape/dtype/device/finite 值，让首样本完整通过 encoder、backbone、GenComm、fusion 和 detection head，随后完成全量 V2X-Real 推理并生成 AP。若需要比较 checkpoint，应单独比较 epoch 15 的最佳验证模型与其他 epoch，不要把 `net_epoch19.pth` 自动保存文件误当成最佳模型。
6. **重新整理 TensorBoard 结论**：按 event 文件核对写入时间、step 范围、续训关系和 `Validate_Loss` 来源，确认 epoch 15 是否确实是同一次训练中的最佳验证 epoch。

### 14.6 本次结论

本次实验的准确状态是：**stage1 训练循环完成并保存了 checkpoint，但训练结束后自动推理在第一个样本的 cuDNN 卷积阶段失败，因此当前没有完整检测评测结果。** 优先处理运行环境和 GPU/cuDNN 兼容性，再进行完整推理；在此之前不要根据训练 loss 或 `Gen Loss` 对最终 AP 和模型质量下结论。

