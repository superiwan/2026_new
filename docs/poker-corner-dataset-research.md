# 扑克碎片角标 YOLO 与圆角消歧：数据集和部署调研

> 调研日期：2026-08-04
> 范围：仅调研公开数据、训练/导出官方链路、MaixCAM Pro 部署约束，以及
> `puzzle-vision-simulator` v2.2 的圆角和 SIFT+RANSAC 实现。本文不代表已下载数据、
> 已训练模型或已完成真机验证。

## 1. 结论先行

没有发现一个公开数据集同时满足以下条件：

- 浅绿色 A4 工作面；
- 真实扑克牌被切成 2～4 块；
- 角标完整但可能只出现在某些碎片上；
- 碎片在平面内任意旋转；
- MaixCAM Pro 的实际镜头、曝光、分辨率和透视；
- 可直接用于判断同形碎片 `original` / `swapped` 的方向标签。

因此，公开数据只能用于预训练或生成数据。最终模型必须加入当前设备、当前绿色 A4、
当前牌面和真实切片的实拍数据，并把独立实拍测试集作为是否可交付的判断依据。

推荐系统不是让 YOLO 直接求解拼图，而是：

1. 模式二几何算法拼好另外两块不规则碎片，并找出唯一同形碎片对；
2. 只构造同形对的 `original` 和 `swapped` 两个候选；
3. 用物理圆角恢复作几何硬约束；
4. 用角标检测及方向证据对两个候选评分；
5. 需要固定完整牌面参考时，再用局部 SIFT+RANSAC 复核；
6. 证据差距不足则返回 `AMBIGUOUS`，不生成动作、不发送 UART。

这使 YOLO 的任务被限制为“小目标角标证据”，不会替代现有分割、模式二求解、
动作生成和串口链路。

## 2. 公开数据源审查

### 2.1 候选总表

| 数据源 | 标注对象 | 角标框 | 方向标签 | 许可/取得方式 | 对本题的结论 |
| --- | --- | --- | --- | --- | --- |
| `geaxgx/playing-card-detection` | 52 张牌类别；README 明确为两个印刷角标 bounding boxes | **有** | 无显式角度；生成器可做平面旋转 | 代码 MIT；GitHub 获取。牌面素材和背景素材仍需分别核对许可 | 最适合建立合成角标预训练和标注逻辑，但不是现成的绿色 A4 实拍碎片集 |
| Kaggle `andy8744/playing-cards-object-detection-dataset` | 52 张牌，YOLOv5/Pascal VOC，`416x416`，合成背景 | 页面只称 bounding boxes，来源受 `geaxgx` 启发；**下载前不能假定框一定是角标框，必须抽样审计** | 页面未声明方向标签 | CC0；Kaggle 页面/API，下载通常需账号和 API 凭据 | 可作为预训练候选，不能直接作为最终验证集 |
| Kaggle `hugopaigneau/playing-cards-dataset` | 多张整牌，Pascal VOC，`1080x1080`，随机背景/平移/旋转 | 页面描述的是牌类别和 bbox，未声明角标级标注 | 无 | CC0；Kaggle 页面/API | 可补充牌面、旋转和背景多样性；需重新标角标，且仍没有真实切片 |
| Kaggle `gpiosenka/cards-image-datasetclassification` | 53 类单牌裁剪，`224x224`；7624 train、265 val、265 test | 无检测框 | 无 | CC0；Kaggle 页面/API | 只适合 rank/suit 或牌面分类预训练，不可直接训练角标 detector |
| Kaggle `artemzysko/playing-cards-dataset-yolo-object-detection` | rank、suit、52 种组合类 | 页面未证明是角标框 | 无 | CC BY-NC 4.0；需署名且限非商业用途 | 仅作补充候选；许可更严格，且未覆盖绿色 A4/碎片/设备视角 |
| Roboflow-100 `poker-cards-cxcvz` | 53 类牌类别；964 train、193 val、128 test，共 1285 图 | 官方 benchmark 元数据没有角标标签说明 | 无 | 数据页条款需在下载时再次确认；benchmark 代码的 MIT 许可不等于数据许可 | 可用于整牌检测对照，不应自动并入角标训练集 |
| `okmd/playing-card-dataset` 等无许可证仓库 | 不一 | 不一 | 不一 | 无明确数据许可 | 默认排除自动下载和训练，除非作者补充授权 |

### 2.2 首选公开资源：`geaxgx` 生成器

仓库 README 明确说明：每张牌以牌名为类别，例如 `2s`、`Kh`，bounding boxes
圈出两个印刷角标。这一点比常见的“整牌 bounding box”更贴近本题。其生成流程可以加入
平面旋转、缩放、平移、纹理背景和重叠，因此适合先学会“角标长什么样”和旋转不变检测。

限制同样明确：仓库主体是生成方法，不能替代当前项目真实数据；它没有当前浅绿色 A4、
真实切割边、纸面翘曲、相机噪声、设备曝光，也未提供 `original/swapped` 方向真值。
此外，MIT 可以确认的是仓库代码；用于生成的牌面图和外部背景图必须逐项核对授权，不能把
代码许可证自动套到所有素材上。

来源（访问日期均为 2026-08-04）：

- 原始仓库及角标标注说明：<https://github.com/geaxgx/playing-card-detection>
- 仓库许可证：<https://github.com/geaxgx/playing-card-detection/blob/master/LICENSE>
- 生成 notebook：<https://github.com/geaxgx/playing-card-detection/blob/master/creating_playing_cards_dataset.ipynb>

### 2.3 Kaggle 候选

Kaggle 数据页的元数据足以确认许可、大小、格式和大致生成方式，但不能证明 bbox 是否紧贴
角标、是否包含方向，或是否覆盖切片。AI 在正式下载后必须先做“标注抽检”，再决定是否并入：

- 随机抽取 train/val/test 各 50 张；
- 可视化所有框，统计整牌框、角标框、漏标、错标和超边界框；
- 检查同一原始视频的相邻帧是否跨 split；
- 检查旋转角覆盖、角标像素尺寸分布和背景分布；
- 任何来源不明的图片或不兼容许可不得进入训练产物。

来源（访问日期均为 2026-08-04）：

- `andy8744`，CC0：<https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset>
- `hugopaigneau`，CC0：<https://www.kaggle.com/datasets/hugopaigneau/playing-cards-dataset>
- `gpiosenka`，CC0：<https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification>
- `artemzysko`，CC BY-NC 4.0：<https://www.kaggle.com/datasets/artemzysko/playing-cards-dataset-yolo-object-detection>
- Kaggle API 认证说明：<https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#authentication>

### 2.4 Roboflow-100 候选

Roboflow-100 官方 benchmark 元数据显示 `poker-cards-cxcvz` 有 1285 张图、53 类，类别是
牌的 rank+suit 组合（元数据中还出现一个异常类 `59`，下载后应核对）。这些信息不能证明存在
角标级框或方向标签。更重要的是，benchmark 仓库的 MIT 许可证只覆盖 benchmark 代码，
不能替代原始数据页的许可。因此它只能先列入审计队列，不能直接自动下载、混合或再发布。

来源（访问日期均为 2026-08-04）：

- 原始数据页：<https://universe.roboflow.com/roboflow-100/poker-cards-cxcvz>
- 官方统计：<https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv>
- 官方类别元数据：<https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/labels_names.json>
- 原始数据映射：<https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/original_datasets.csv>
- benchmark 代码许可证：<https://github.com/roboflow/roboflow-100-benchmark/blob/main/LICENSE.md>

## 3. 必须补采的真实设备数据

### 3.1 标注定义

第一版只设一个检测类 `corner_mark`，框紧贴一个完整的“点数/字母 + 花色”角标组合。
不要给中央花色、人物图案和普通文字框。负样本应刻意包含中央花色、人物牌纹理、直切边、
圆角、阴影和没有角标的碎片。

普通 YOLO detect 的轴对齐框不输出方向。把训练增强设为任意旋转只能提高检测鲁棒性，
不能让输出突然带上 `0～360°` 角度。方向证据按以下顺序增加：

1. 先由碎片几何姿态和“角标相对哪个物理圆角”推导方向；
2. 若 `0°/180°` 仍混淆，对角标 crop 训练二分类方向头；
3. 若必须直接输出有向角度，使用两个有序关键点（如 rank 中心到 suit 中心）的 pose 模型；
4. OBB 可表达旋转框，但其角度约定通常不足以天然消除 `180°` 方向歧义，不能只凭 OBB 假定方向已解决。

### 3.2 采集矩阵

建议先采 300～500 张 pilot，跑通标注、训练、导出和真机推理；然后根据失败簇扩充到
至少 1500～3000 张**去重后的真实帧**。数量不是硬指标，覆盖比相邻视频帧堆量更重要。

每组至少覆盖：

- 四个 90° 象限，并细分到约 15° 角度桶；
- A4 上方不同位置、靠近边界但不被截断的位置；
- 暗、正常、亮三档曝光和多种室内光向；
- 红/黑花色、数字牌和 J/Q/K/A；
- 两个相同长方形或正方形碎片，以及另外两块不规则碎片；
- 角标完整、无角标碎片、只有中央花色/人物纹理的 hard negatives；
- 轻微模糊、反光、阴影、透视和纸张翘曲，但不加入测评明确不会出现的角标截断正样本。

按“采集 session + 实体牌 + 切割方案”分组切分，而不是随机按帧切分。相邻视频帧不得跨
train/val/test。建议比例为 70/15/15；另外固定不少于 300 个独立实拍场景作为最终 test，
测试集在所有阈值冻结前不可查看结果细节。

### 3.3 INT8 校准集

Sipeed 官方建议 MaixCAM 的 INT8 校准使用 20～200 张代表性真实场景图，初次可从约 50 张
开始。校准图必须覆盖绿色 A4、实际曝光、角标大小和背景分布；从训练域单独建清单，不得使用
最终 test。校准图仅服务量化，不应与“模型精度测试集”混为一谈。

## 4. 训练与导出流程

### 4.1 Ultralytics 训练

Ultralytics 官方支持 Python 和 CLI 两种训练入口，例如：

```bash
yolo detect train data=poker_corner.yaml model=yolo11n.pt epochs=100 imgsz=640
```

数据 YAML 应固定 train/val/test 路径和 `corner_mark` 类名。训练阶段应记录：

- 数据来源、许可证、文件哈希和 split 清单；
- Ultralytics 版本、初始权重、随机种子、`imgsz`、batch、epochs 和全部增强参数；
- `best.pt`、`last.pt`、混淆矩阵、PR 曲线、逐类 AP（即使当前只有一类）和误检/漏检样例；
- 按角标像素尺寸、旋转角、光照、牌类、是否真实数据分桶的 recall；
- `original/swapped` 二选一的最终场景准确率，而不只报告 detector mAP。

旋转增强可以覆盖任意平面方向，但训练集仍需含真实旋转。增强图不能进入 val/test，且应避免
把角标旋转到框外。先以 YOLO11n detect 建立低成本基线；只有方向消歧证据证明不足时，再升级
到 crop 方向分类或 pose，避免一开始增加设备端后处理复杂度。

来源（访问日期均为 2026-08-04）：

- 官方 Train mode：<https://docs.ultralytics.com/modes/train/>
- 官方 detection 数据格式：<https://docs.ultralytics.com/datasets/detect/>
- 官方配置/增强参数：<https://docs.ultralytics.com/usage/cfg/>

Ultralytics 软件当前采用 AGPL-3.0 和 Enterprise 两种许可路径。项目交付或闭源分发前，
应按实际使用方式确认框架许可；数据集的 CC0/MIT 不能替代训练框架的许可证。

### 4.2 ONNX 和 CV181x

Ultralytics 官方可用 `yolo export model=best.pt format=onnx` 导出 ONNX；MaixCAM Pro
链路必须使用固定输入尺寸，不能使用 dynamic shape。Sipeed 当前文档给出的完整链路为：

```text
best.pt
  -> fixed-shape ONNX
  -> 按实际图检查并裁剪输出节点
  -> tpu-mlir / cv181x INT8 转换
  -> .cvimodel + .mud
  -> MaixPy nn.YOLO11
```

Sipeed 对 YOLO11 detect 推荐的 MaixCAM 输出节点是：

```text
/model.23/dfl/conv/Conv_output_0
/model.23/Sigmoid_output_0
```

节点名必须用 Netron 按实际 ONNX 核对，不能机械照抄。官方说明 MaixCAM 常用固定
`320x224`；但角标是小目标，项目应至少比较两个固定尺寸，记录 recall、端到端延迟、内存和
热稳定性，再冻结尺寸。高分辨率可能提高小目标 recall，但会降低速度。

来源（访问日期均为 2026-08-04）：

- Ultralytics 官方 Export mode：<https://docs.ultralytics.com/modes/export/>
- Sipeed 离线 YOLO 训练/部署：<https://wiki.sipeed.com/maixpy/doc/en/vision/customize_model_yolo.html>
- Sipeed ONNX 转 MaixCAM：<https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html>
- Sipeed MaixHub 数据采集、标注、训练和部署：<https://wiki.sipeed.com/maixpy/doc/en/vision/maixhub_train.html>
- Sipeed 官方 YOLO11 detect 示例：<https://github.com/sipeed/MaixPy/blob/main/examples/vision/ai_vision/nn_yolo11_detect.py>

## 5. `puzzle-vision-simulator` v2.2 可借鉴部分

### 5.1 圆角恢复

上游 `_merge_rounded_corners()` 把物理圆角视为两条长直边之间的 1～3 段短弦，检查短弦
长度、两侧直边长度、近似垂直关系和延长线交点距离，再用两条直线交点替换圆弧采样点，得到
供几何求解使用的“虚拟尖角”。`_rounded_vertices_by_piece()` 再按虚拟角到实测轮廓的距离，
在所有碎片上联合排序并选出完整牌面的四个物理圆角。

这套思想适合本项目，但阈值不能直接搬用。上游固定 `10 cm x 6 cm`、黑色 A4 和约 4 mm
圆角；当前项目是绿色 A4、`420x594` 透视平面和不同牌面比例。应保留“实测轮廓”和“恢复
多边形”两套几何，并用毫米尺度重标定短弦、直边、角度和切深阈值。

### 5.2 SIFT+RANSAC

上游 `solve_textured_card()` 对已知完整 Joker 参考图建立 SIFT 特征，对每块碎片的 mask 内
提特征，用 BFMatcher L2 和 Lowe ratio `0.76` 筛选，再以
`estimateAffinePartial2D(..., RANSAC)` 求配准。至少需要 4 个内点，并限制估计尺度
`0.94～1.06`；最后去掉尺度，只保留刚体旋转和平移。

本题不需要对所有几何排列运行全图特征搜索。更稳妥的用法是只在圆角与 YOLO 对
`original/swapped` 仍无法拉开差距、并且存在合法完整牌面参考图时，对两个候选做局部复核。
证据不足必须保持 `AMBIGUOUS`。

来源（访问日期均为 2026-08-04）：

- v2.2 release：<https://github.com/lvreng/puzzle-vision-simulator/releases/tag/v2.2.0-vision-rounded-card>
- 圆角恢复源码：<https://github.com/lvreng/puzzle-vision-simulator/blob/v2.2.0-vision-rounded-card/puzzle_sim.py#L517>
- 实测轮廓和虚拟轮廓关联：<https://github.com/lvreng/puzzle-vision-simulator/blob/v2.2.0-vision-rounded-card/puzzle_sim.py#L615>
- 全局四圆角：<https://github.com/lvreng/puzzle-vision-simulator/blob/v2.2.0-vision-rounded-card/puzzle_sim.py#L682>
- SIFT+RANSAC：<https://github.com/lvreng/puzzle-vision-simulator/blob/v2.2.0-vision-rounded-card/puzzle_sim.py#L1053>

GitHub 未识别到该仓库的许可证。未取得明确许可前，应把它当作算法参考并独立实现，
不要直接复制源码到交付项目。

## 6. 建议交给 AI 的执行清单

### 阶段 A：数据许可和样本审计

1. 建立 `DATA_SOURCES.md`，逐项记录 URL、版本/更新时间、许可证、下载时间、哈希和用途。
2. 只下载许可明确且与用途兼容的来源；无许可证来源直接排除。
3. 可视化抽检每个公开集，确认 bbox 是角标还是整牌；输出审计报告，不合格数据不得静默混入。
4. 检查重复图和相邻视频帧泄漏，按来源和采集 session 重建 split。

交付物：来源清单、许可证副本/链接、审计图、去重报告、冻结 split manifest。

### 阶段 B：真实采集和标注

1. 用 MaixCAM Pro 在最终机位采 pilot，覆盖采集矩阵。
2. 按统一规则标 `corner_mark`，加入无角标碎片 hard negatives。
3. 双人抽检或 AI 标注 + 人工复核；统计漏标、框松紧和类别一致性。
4. 固定 session 级 train/val/test；test 上锁。

交付物：原始图、YOLO 标签、标注规范、QA 报告、split manifest。

### 阶段 C：基线训练和错误驱动扩充

1. 先用合成/公开数据预训练，再用真实数据 fine-tune；同时训练“只用真实数据”对照组。
2. 固定随机种子做至少 3 次重复，比较均值和波动。
3. 输出漏检、误检、低置信度、角度桶和小目标尺寸桶失败簇。
4. 只针对真实失败簇补采，不用无限合成图掩盖域差异。

交付物：训练配置、日志、权重、曲线、逐桶指标、失败样例目录。

### 阶段 D：算法集成验证

1. 保持模式二几何求解和另两块不规则碎片流程不变。
2. 对唯一同形对强制生成 `original/swapped`。
3. 圆角作硬约束；角标位置/方向作评分；可选 SIFT+RANSAC 作二次复核。
4. 明确定义分数 margin；低于 margin 返回 `AMBIGUOUS`，不得强猜。
5. 验证所有失败路径都不生成动作、不触发 UART。

交付物：离线 replay 报告、候选评分明细、歧义样本和对应单元/集成测试。

### 阶段 E：导出和真机验收

1. 冻结固定输入 ONNX，并比较 PyTorch 与 ONNX 输出。
2. 用真实 calibration set 转 CV181x INT8，生成匹配的 `.cvimodel` 和 `.mud`。
3. 比较 PC FP32、ONNX、INT8 三层精度差；量化退化超限则补校准或评估 BF16。
4. 在 MaixCAM Pro 实测识别、二选一结果、端到端耗时、内存、连续运行和错误处理。
5. 最后才做真实 UART 联调；PC replay、合成图和设备文件加载不能替代真实场景验收。

交付物：ONNX、`.cvimodel`、`.mud`、转换命令和日志、真机测试原始记录、最终验收表。

## 7. 建议验收门槛

门槛应在第一次查看锁定 test 之前冻结。可先采用以下候选值，再根据测评漏检成本确认：

- `corner_mark` 实拍 test recall >= 99%，precision >= 98%；
- 小角标尺寸最低桶 recall >= 97%；
- 同形对 `original/swapped` 场景级正确率 >= 99%；
- 错误候选不得被高置信发送；不确定时 `AMBIGUOUS` 率单独报告；
- INT8 相对 FP32 的场景级正确率下降 <= 1 个百分点；
- 连续真机运行中无模型加载失败、内存持续增长或显示/推理崩溃；
- 每次发布保留相同锁定 test 回归，数据更新不得覆盖旧测试集。

这些是方案门槛，不是已经达到的结果。最终报告必须分别标注：PC 合成验证、PC 实拍 replay、
ONNX 验证、CV181x INT8 验证、MaixCAM Pro 实景验证和 UART 实物联调，不能互相替代。
