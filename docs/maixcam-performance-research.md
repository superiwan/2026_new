# MaixCAM Pro 实时拼图路径性能调研

## 范围与结论

本文针对 `main.py` 的实时路径做静态分析，资料只引用 Sipeed/MaixPy/MaixCDK 和 OpenCV 官方文档或源码。随后已通过 USB SSH 在 MaixCAM Pro 真机上完成有界合成场景基准。

真机验证结果支持下文的优先级判断：原实现完整 CPU 帧约 `466 ms`；改为灰度半区 fixed-point `remap`、单次多边形近似、bitmask DP 和原图 `LINE_8` 叠加后，完整检测帧约 `65.5 ms`，非检测显示帧约 `8.6 ms`。这些数字尚未包含真实相机和 `Display.show` 的端到端对比，后者仍需区分 SSH 独立运行与 MaixVision 预览模式。

最应优先验证的不是 OpenCV 单个函数，而是下面四件事：

1. **不要在 MaixVision 模式下测比赛帧率。** MaixPy 文档明确说明 `Display.show()` 在 MaixVision 中会自动压缩并发送图像；MaixCDK 源码也显示 `Display::show` 会先调用 `img_trans->send_image(img)`，之后才提交到本机显示。也就是说，每帧可能额外承担压缩和传输。[MaixPy display 文档](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/display.md#displaying-on-maixvision)；[MaixCDK `Display::show` 源码](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/src/maix_display.cpp#L186-L203)
2. **实时检测不要做整张三通道透视。** 当前先把 `640x480 RGB` 透视为 `420x594 RGB`，然后才转灰度，而真正检测只使用上半区。应先转灰度，再仅生成上半区透视结果；保存下半区只在按下 `SAVE` 时计算。其输出字节量理论上从约 `420*594*3` 降至约 `420*285`，减少约 84%，但最终 FPS 增益必须实测。
3. **缓存透视采样表。** A4 单次定位后单应矩阵不变，可以一次生成逆向坐标表，后续用 `remap`。OpenCV 支持复用固定映射，并可用 `convertMaps` 转为更紧凑、更快的 fixed-point 表。[OpenCV 几何变换接口与说明](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp#L2519-L2596)
4. **去掉每帧多次整图复制/混合。** 当前 `draw_piece_overlay` 先 `warped.copy()`，随后每个保存碎片又执行一次 `overlay.copy()` 和一次全图 `addWeighted`。6 个碎片时，仅这些临时 RGB 图约搬运 `6 * 420 * 594 * 3 = 4.49 MB/frame`，尚未计初始副本、输出和其他预览。应该用一个 mask/layer 收集所有填充，最后只 blend 一次，或比赛模式只画轮廓。

## 当前每帧路径的主要开销

识别过 A4 后，当前循环每帧依次执行：

- `cam.read()` 和零拷贝 `image2cv`
- `warpPerspective`：`640x480 RGB -> 420x594 RGB`
- `cvtColor`、`threshold`、`morphologyEx`、`findContours`
- 每个候选轮廓最多测试 6 个 `approxPolyDP` epsilon；每次还可能执行边拟合、两个局部 mask 的分配/填充和 IoU
- 最多 6 个碎片的全排列匹配（最坏 `6! = 720` 个 assignment）
- 构建 `640x480` UI，缩放 raw 和 warped preview，重复 alpha blend，绘制抗锯齿文字/线条
- `cv2image(copy=False)` 和 `Display.show()`

因此 `warp`、`binary_morph`、`contours`、`approx_poly`、`total` 这几个已有计时还不完整：它们没有覆盖 `match`、`build_result_view`、`cv2image`、`display.show`、`camera.read`。在改算法前应先补全这些分段计时，并每 30 帧打印一次 p50/max 或累计平均值；不要逐帧打印，以免串口/终端输出本身干扰。

## Camera、格式与缓冲

### 已确认行为

- MaixPy 官方建议图像处理尽量降低分辨率；`640x480` 已对 MaixCAM 有一定负担，`320x240` 更容易跑算法。[MaixPy camera 文档](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/camera.md#choosing-resolution-size)
- `Camera` 的默认格式是 `RGB888`；也支持直接输出 `GRAYSCALE` 和 `YVU420SP`。[MaixPy camera 文档](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/camera.md#getting-image-from-the-camera)
- `fps` 可以在构造器中指定；官方同时提醒 60/80 FPS 相对 30 FPS 可能发生几个像素的画面偏移，精确对位场景要校正。[MaixPy camera 文档](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/camera.md#L147-L179)
- 文档称不高于 `1280x720` 时默认 80 FPS，但当前 MaixCAM 实现中 `fps=-1` 在不高于 720p 时先落到 60 FPS，满足显式请求等条件才可能配置 80 FPS。目标固件也可能不是该源码版本，因此必须在设备上打印 `cam.fps()` 和固件/MaixPy 版本，不能假定实际采集率。[MaixCDK MaixCAM FPS 选择源码](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/port/maixcam/maix_camera_mmf.cpp#L78-L126)
- `buff_num` 越多通常更利于读取吞吐，但耗费更多内存；`buff_num=1` 可降低采集延迟，但可能丢帧。MaixCAM 内部仍至少有双缓冲，官方测试最小采集延迟约 30+ ms。[MaixPy camera 文档](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/camera.md#setting-lower-capture-latency)；[MaixCDK Camera API](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/include/maix_camera.hpp)
- MaixCAM 的 `read` 路径会把 MMF frame 搬到 `Image`：`RGB888` 为整帧复制，`GRAYSCALE` 只拷贝 Y 平面，`YVU420SP` 搬运约 1.5 bytes/pixel。请求 `BGR888` 时底层仍配置 `RGB888`，读取时再逐像素换 R/B；所以当前直接请求 `RGB888` 并以 RGB 语义处理是合理选择，不应为了迎合 OpenCV 默认 BGR 改成摄像头 BGR 输出。[MaixCDK MaixCAM camera read 源码](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/port/maixcam/maix_camera_mmf.cpp#L996-L1096)

### 本项目建议

- 比赛目标是低延迟时，对比 `buff_num=1` 与 `2`。`1` 可能更“跟手”，但不能假定 FPS 更高；分别记录 `camera.read` 时间、总 FPS 和是否丢帧。
- 当前 MaixCAM 源码在 `buff_num != 1` 时还会按 `1/fps` 做 pacing 等待，而 `buff_num=1` 跳过该等待。这进一步说明 `1` 值得用于低延迟 A/B 测试，但仍有官方提示的丢帧风险。[MaixCDK `Camera::read`](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/port/maixcam/maix_camera_mmf.cpp#L1163-L1192)
- 显式设置一个传感器稳定支持的 `fps`（先试 30 或 60），不要把自动 80 FPS 当作算法必须跟上的目标。若分析只跑 10~15 Hz，显示仍可维持更高刷新率。
- 不建议把唯一摄像头通道直接改为灰度，因为 UI 仍需彩色；可进一步实验 `add_channel(...FMT_GRAYSCALE)` 作为分析通道，但要验证两通道同步、额外 VPSS 成本以及坐标一致性后再采用。
- 若比赛模式可以接受灰度画面和灰度标记，应直接 A/B 测试唯一通道 `FMT_GRAYSCALE`：官方格式定义为 1 byte/pixel，而 RGB 为 3 bytes/pixel，且 MaixCAM read 对灰度只复制 Y 平面。它能同时省掉 RGB read 带宽和 `cvtColor`，但这是 UI 取舍，不能在未看实机交互前直接定案。[MaixCDK image formats](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/include/maix_image_def.hpp#L24-L38)

## `image2cv` / `cv2image` 是否复制

当前调用：

```python
rgb = image.image2cv(frame, ensure_bgr=False, copy=False)
shown = image.cv2image(live_view, bgr=False, copy=False)
```

已经是官方给出的极限速度写法：这两个“格式桥接”步骤均借用原内存，不产生转换副本（不代表前面的 `Camera.read` 没有搬运）。官方同时强调生命周期要求：`frame` 必须活到 NumPy/OpenCV 使用结束，`live_view` 必须活到 `shown` 使用结束；修改借用数组也会修改原图。[MaixPy OpenCV 文档](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/opencv.md#L20-L57)；[MaixCDK 转换接口](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/include/maix_image_cv.hpp)

结论：这里不是当前首要瓶颈，也没有更快且同样通用的替代 API。不要改回 `ensure_bgr=True` 或 `copy=True`。真正的复制来自后续 `warpPerspective`、`cvtColor`、多个 `.copy()`、`resize` 和 blend。

## `warpPerspective` 的替代

### 方案 A：灰度 + 半区 `warpPerspective`（先做）

OpenCV 明确说明 `warpPerspective` 产生给定 `dsize` 的目标图，且不能原地执行。[OpenCV `warpPerspective`](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp#L2491-L2517)

实时 check 只需要上半区，所以可将处理顺序改成：

1. `gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)`
2. 使用调整后的 homography 只输出 `WARP_W x upper_height`
3. 在这张小灰度图上 threshold/morph/contours
4. 仅低频生成彩色透视预览，或比赛模式完全不生成

这项改动简单，且同时减少透视、阈值、形态学和轮廓输入像素数。

### 方案 B：预计算 `remap`（第二步基准）

`remap` 使用 `map_x/map_y` 指定每个目标像素从源图的采样位置；固定 A4 后这些位置不会改变。可由 homography 的逆矩阵一次生成上半区 float map，再执行：

```python
map1, map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
upper_gray = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
```

OpenCV 官方说明 fixed-point map 更紧凑、更快，且有利于重复使用同一 map；但当前 `4.x` 头文件也明确提示实际收益依硬件而异，应测量后决定。[OpenCV `remap` / `convertMaps`](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp#L2519-L2596)

注意：`remap` 不是零拷贝，也不能原地执行；它优化的是每帧重复计算坐标映射及 map 的内存访问，不会消除插值本身。

## ROI、灰度、阈值和轮廓

- 当前先处理完整 `420x594`，再把非目标半区写 0。应直接把上半区作为检测输入，使 `threshold`、`morphologyEx`、`findContours` 都只处理实际 ROI。
- `CHAIN_APPROX_SIMPLE` 已经是合理设置：OpenCV 会压缩水平、垂直、对角线段，只保留端点。[OpenCV contour approximation modes](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp#L440-L452)
- `MORPH_KERNEL` 的 3x3 kernel 不应每帧 `np.ones`；初始化一次复用。收益可能不大，但修改低风险。
- 若现场背景稳定，可测试取消 `MORPH_CLOSE`，或降为只在轮廓数量异常时执行；正确率优先，不能未经样片验证直接删除。

## 轮廓拟合与匹配逻辑

当前最值得怀疑的 CPU 逻辑不是 6 个碎片的全排列，而是 `approximate_piece`：每个候选轮廓最多尝试 6 个 epsilon，每个候选都可能做边拟合、mask 分配、`fillPoly` 和 raster IoU。

建议拆成两条路径：

- `SAVE`（只执行一次）：保留高精度多 epsilon + 顶点 refinement，记录模板。
- 实时 `CHECK`：直接使用原轮廓的 `area`、周长/顶点粗略特征和 `cv2.matchShapes` 做槽位代价；只有当状态从“不匹配”接近“匹配”时，才做一次精细拟合确认。

OpenCV 的 `matchShapes` 三种模式均基于 Hu invariants；Hu moments 在理想条件下对尺度、旋转和反射不变（第七项反射时变号），栅格图像会有少量误差。这恰好适合“碎片位置和旋转在上半区任意，但形状身份不变”的粗匹配。[OpenCV `HuMoments`](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp#L3889-L3919)；[OpenCV `matchShapes`](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp#L4316-L4327)

全排列在 6 个碎片时只有 720 项，通常不是第一优先级。如果设备计时证明 `match` 仍明显，可换成 bitmask DP，复杂度从 `O(n! * n)` 降到 `O(n^2 * 2^n)`；但这属于项目算法优化，不是 OpenCV 替代 API。

## 绘制、缩放与显示

### 当前可直接削减的开销

- 将每个 saved polygon 的“复制整图 + 填充 + 全图 blend”改为：单个 overlay/mask 画完所有 polygon，只执行一次 `addWeighted`。
- 轮廓若满足凸或单调条件，可使用 `fillConvexPoly`；OpenCV 官方说明它比 `fillPoly` 快得多。但不满足条件的拼图块不能强用，否则填充结果错误。[OpenCV drawing API](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp#L4820-L4841)
- 比赛模式把 `LINE_AA` 改为 `LINE_8`，只保留必要轮廓/状态文字。官方未承诺具体加速倍数，因此要以设备基准为准。
- `raw_small`、`warp_small` 不必每帧都更新；视觉反馈可 5~10 Hz 更新，触摸、检测状态和最终指引独立刷新。
- 更激进的方案是直接以 `640x480` 摄像头画面为 canvas 原地画少量 UI，取消 raw preview resize、warped preview resize 和保存布局缩略图的每帧重绘。

### `Display.show` 的边界

MaixCAM 主显示层直接支持 `RGB888`、`GRAYSCALE`、`YVU420SP`，底层通过 `mmf_vo_frame_push_with_fit` 提交；`FIT_CONTAIN` 等 fit 行为由显示管线处理。若传入 BGR，`cv2image(copy=False)` 本身虽不复制，`Display.show` wrapper 仍会先新建 RGB 图并转换；当前以 `bgr=False` 标记真正的 RGB buffer 是正确路径。[MaixCDK display wrapper](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/src/maix_display.cpp#L204-L221)；[MaixCDK MaixCAM display 源码](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/port/maixcam/maix_display_mmf.hpp#L328-L403)

因此当前 `640x480 RGB` canvas 与 MaixCAM Pro `640x480` 屏幕尺寸一致，是合理的。要重点比较：

- 单独 SSH/独立 app 运行时的 `display.show` 时间
- MaixVision 模式下的 `display.show` 时间
- 暂时不调用 `display.show` 时的纯算法 FPS

若后两者差异大，瓶颈是显示/远程预览链路，不应继续微调轮廓算法来“补偿”。

## 推荐实施顺序与验收

1. **完整计时**：补 `read / convert-in / analysis / match / render / convert-out / show / loop`，以 30 帧窗口输出。
2. **运行方式 A/B**：SSH 独立运行 vs MaixVision，另测临时禁用 `show`；先确认卡顿是否主要来自远程预览。
3. **渲染瘦身**：single blend、减少 AA/文字、preview 降频；目标是 `render` 明显低于检测耗时。
4. **分析瘦身**：先灰度、只 warp 上半 ROI、复用 kernel；保存下半区仍按键触发。
5. **检测降频**：相机/显示 30 FPS，分析 10~15 Hz，复用最近一次匹配结果。对人手摆拼图仍属于实时反馈，但需要现场确认响应延迟。
6. **轮廓快路径**：实时用 contour + `matchShapes`，精细 polygon 仅保存/确认时执行。
7. **最后再测 remap**：对比半区灰度 `warpPerspective`、float-map `remap`、fixed-map `remap`，以板端测量决定保留哪一个。
8. **缓冲实验**：固定算法后比较 `buff_num=1/2` 和 `fps=30/60`，同时观察延迟、丢帧、坐标偏移和温度。

建议验收目标不是只看平均 FPS，而是同时记录：

- 显示 FPS 与分析 Hz
- 从移动碎片到屏幕更新的端到端延迟
- `loop` 的平均值、p95/max（识别“偶发卡死”）
- 连续 2 分钟是否内存增长或崩溃
- 6 个碎片场景下匹配准确率是否下降

## 官方来源索引

- [MaixPy: Use OpenCV](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/opencv.md)
- [MaixPy: Camera](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/camera.md)
- [MaixPy: Display](https://github.com/sipeed/MaixPy/blob/7337fc84f1487d807fc41a63e5bf9ee84b4d9b6e/docs/doc/en/vision/display.md)
- [MaixCDK: OpenCV conversion API](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/include/maix_image_cv.hpp)
- [MaixCDK: Camera API](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/include/maix_camera.hpp)
- [MaixCDK: MaixCAM camera implementation](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/port/maixcam/maix_camera_mmf.cpp)
- [MaixCDK: MaixCAM display implementation](https://github.com/sipeed/MaixCDK/blob/2a0502ecb20e5695b28580b3689492b7a228f9e4/components/vision/port/maixcam/maix_display_mmf.hpp)
- [OpenCV: imgproc transform/shape/drawing declarations and documentation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp)
