# 模式三扑克拼接完整性优化

> 当前状态：本页主体记录的是旧版扑克专用求解策略。当前模式三仍调用模式二
> `Task2WhiteSolver.solve_detected()` 的默认有界流程，并保留 `5:7` 比例门限；2026-08-05 起仅
> 恢复最多 12 个 finalist 的轻量接缝连续性排序，并让 `80 s` 超时输出复用同一排序，不恢复
> 本页旧版的独立搜索流程。四块扑克固定存在的同形二元组必须比较 `original` / `swapped`；
> 严格矩形门限为空时使用最近形状对，最终以对角牌角和无间距接缝纹理共同排序。

## 问题描述

在模式三（扑克拼接任务）中，当检测到多个碎片时，如果只有部分碎片能拼成高填充率的局部解，系统可能会错误地接受这个不完整的拼接结果。

### 具体场景

- **检测到**：4个扑克碎片
- **实际拼接**：只使用了2-3个碎片
- **问题**：虽然这2-3个碎片能拼成高填充率的局部矩形，但这不是一个完整的扑克牌

### 原因分析

原有的质量判断标准包括：
1. `fill_ratio`（填充率）：拼接面积 / 外接矩形面积
2. `overlap_ratio`（重叠率）：碎片是否过度重叠
3. `aspect`（长宽比）：是否接近扑克牌标准比例
4. `disconnected_area`（断连面积）：是否有分离区域

但**缺少对"所有检测到的碎片是否都被使用"的检查**。

## 优化方案

### 核心思路

添加轻量级的"碎片使用率检查"，要求当检测到多个碎片时（2-4个），所有碎片都必须参与拼接。

### 实现细节

在 `Task3PokerSolver.solve_detected()` 方法中：

1. **添加碎片计数方法** `_count_used_pieces()`
   - 通过遍历匹配集合（matches），统计参与拼接的唯一碎片数量
   - Match格式: `(error, i, ei, j, ej, ...)`，其中i和j是碎片索引

2. **在每个求解阶段验证**
   - 标准求解（standard）
   - 多片部分拼接（multi_partial）
   - 回退模式（fallback）
3. **拒绝不完整解**
   - 当 `used_pieces < piece_count` 时，抛出错误或拒绝该解

### 代码修改

```python
class Task3PokerSolver:
    def _count_used_pieces(self, matches):
        """统计匹配集合中使用的唯一碎片数量"""
        if not matches:
            return 0
        used = set()
        for match in matches:
            used.add(match[1])  # piece i
            used.add(match[3])  # piece j
        return len(used)

    def solve_detected(self, rectified_rgb, pieces, mask=None):
        # 当检测到多个碎片时，要求全部使用
        piece_count = len(pieces)
        require_all_pieces = piece_count >= 2
        # 在每个求解分支中验证
        if require_all_pieces and result is not None:
            used_pieces = self._count_used_pieces(result[1].get("matches", []))
            if used_pieces < piece_count:
                # 拒绝不完整的解
                raise RuntimeError(
                    "拼接仅使用了 %d/%d 块碎片，不是完整扑克牌"
                    % (used_pieces, piece_count))
```

## 性能影响

### 时间复杂度
- `_count_used_pieces()`: O(m)，其中m是匹配数量
- 典型情况：m ≤ 3（4个碎片需要3个匹配）
- **额外开销：可忽略不计**（<0.1ms）

### 不影响推理速度
- 检查只在每个候选解生成后进行，不影响主求解循环
- 不增加搜索空间或候选数量
- 纯粹是最终验证步骤

## 效果预期

### 优点
1. **准确性提升**：拒绝不完整的局部解
2. **性能无损**：几乎零额外开销
3. **错误提示清晰**：明确告知"仅使用X/Y块碎片"

### 边界情况处理
- **单碎片（piece_count=1）**：不检查（已经是完整的）
- **检测错误（检测到噪声碎片）**：会拒绝所有解 → 需要重新检测或调整检测阈值
- **真实缺失碎片**：正确拒绝（确实不完整）

## 未来改进方向

如果需要进一步提升准确性，可以考虑：

1. **面积一致性检查**
   ```python
   # 检查拼接后的总面积是否接近标准扑克牌面积
   POKER_STANDARD_AREA = 420 * 297  # A4的某个比例
   if abs(union_area - POKER_STANDARD_AREA) > threshold:
       reject
   ```

2. **外形轮廓验证**
   ```python
   # 检查外接矩形的长宽比是否接近扑克牌标准
   POKER_ASPECT = 88.0 / 63.0  # 扑克牌标准尺寸
   ```

3. **边缘完整性检查**
   ```python
   # 检查外边缘是否都是"原始边缘"而不是"切割边缘"
   # 通过纹理特征区分扑克牌外边缘和内部切割边
   ```

但这些会显著增加计算开销，当前的"碎片使用率"检查是最佳权衡。

## 测试

运行 `test_poker_completeness.py` 验证逻辑正确性：
```bash
python test_poker_completeness.py
```

## 总结

这次优化通过**轻量级的碎片使用率检查**，在几乎不增加计算开销的前提下，有效防止了系统接受不完整的扑克拼接解决方案。这是速度和准确性之间的最佳平衡点。
