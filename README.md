# 2026 MaixCAM Pro 三模式拼图视觉

本项目将以下两个仓库整合为一个可由 MaixVision 直接打开的 MaixPy 项目：

- `superiwan/2026_new`：设备入口、黑色 A4 检测、420x594 透视校正、固定碎片查表识别和 UART 实现基础；
- `superiwan/2026_vision_v1`：题 2(1) 纯白随机多边形的边长配对、邻接图恢复、位姿图优化和矩形质量门限。

坐标平面固定为 420x594 px，对应 A4 纸 210x297 mm，即 `0.5 mm/px`。三个 Solver 都返回 `List[PieceAction]`，设备入口只处理模式切换、A4 校正、分步执行和串口发送。

## 目录

```text
main.py                       三按钮 MaixCAM 状态机入口
legacy_2026_new.py            可对照的 2026_new 原实现
core/piece_action.py          统一毫米位姿动作
core/serial_protocol.py       统一 UART + CRC-8 协议
solvers/task1_fixed.py        题1固定四片查表匹配
solvers/task2_white.py        题2(1)纯白拓扑求解
solvers/task2_config.py       白片上游求解参数
solvers/task3_poker.py        题2(2)几何候选 + 纹理接缝评分
tests/test_merged_project.py  PC 合成回归
```

## 设备运行

1. 在 MaixVision 中打开整个项目目录，运行 `main.py`。
2. 保证完整黑色 A4 纸可见，纸外留出亮色边界。
3. 点击屏幕顶部 `[TASK1 FIXED]`、`[TASK2 WHITE]` 或 `[TASK2 POKER]`。设备屏幕只显示 ASCII 英文，避免固件字体缺失导致乱码；中文诊断仍输出到 MaixVision 终端。
4. 程序按相机帧依次执行 A4 定位、透视校正、对应 Solver 和 UART 发送；正常预览不持续运行重算法。
5. 相机或 A4 移动后重新点击当前模式，重新获取透视矩阵。

题1首次运行时，A4 下半区必须放置拼好的 4 块固定碎片，上半区放置对应散片。程序会生成 `task1_layout.json`；之后始终使用该查表模板。更换固定碎片时，应先把旧模板另行备份，再手动移走该文件并重新初始化。

## Solver 契约

每个 Solver 实现：

```python
actions, diagnostics = solver.solve(rectified_rgb)
```

其中每个 `PieceAction` 包含：

```text
piece_id
pick_x, pick_y, pick_angle
place_x, place_y, place_angle
confidence
```

坐标单位为毫米，原点为矫正后 A4 左上角，X 向右、Y 向下；角度沿图像坐标正方向计算并归一化至 `[-180, 180)`。

## UART

使用 MaixCAM Pro `UART1`，`115200` 波特率：A18 为 RX，A19 为 TX，两块板必须共地并使用 3.3 V 电平。

每个动作发送一帧：

```text
$A,piece_id,pick_x,pick_y,pick_angle,place_x,place_y,place_angle*CRC8\r\n
```

CRC-8 对 `A,...` payload 计算，多项式 `0x07`、初值 `0x00`。只有 Solver 完整通过质量门限后才发送动作。

## 扑克算法边界

`2026_vision_v1` 当前版本虽然能渲染大鬼扑克牌，但上游 README 和代码都明确使用几何拼接，没有已经实现的 ORB/SIFT 或纹理求解器。`task3_poker.py` 因此保留上游拓扑候选搜索，并由本整合项目新增切割缝两侧的颜色/梯度连续性评分。它不是从上游直接提取的现成纹理算法，仍需用现场牌面图片和真机帧验证阈值、候选规模与性能。

## PC 验证

```powershell
python main.py --self-test
```

测试覆盖三阶段状态机、题1四片查表、题2毫米动作、扑克纹理接缝排序和 UART CRC。PC 合成测试不等同于 MaixCAM Pro 真实相机、触控、中文字体、UART 电气链路或真实 FPS 验收。

设备端可使用有界启动回归，不需要点击按钮，也不会持续运行：

```text
python3 main.py --device-smoke=120
```

该模式执行真实相机、Display、Touch、OpenCV/Maix 图像桥接和三按钮绘制路径，
达到指定帧数后正常退出。`image.cv2image(..., copy=False)` 会借用 NumPy 内存，
因此代码必须保留 `canvas` 引用直到 `display.show()` 完成，不能直接传入临时数组。

## 上游追踪

当前 Git 配置保留两个 remote：

```text
origin  https://github.com/superiwan/2026_new.git
vision  https://github.com/superiwan/2026_vision_v1.git
```

`task2_white.py` 与 `task2_config.py` 的算法主体来自 `vision/main` 的 `39c94d8`。`legacy_2026_new.py` 对应 `origin/main` 的 `c68ffd9`，便于后续对照更新。
