"""测试扑克模式仅带比例门限委托给模式二求解器。"""
import numpy as np
from unittest.mock import patch

from solvers import task2_white as geometry
from solvers.task3_poker import (
    Task3PokerSolver,
    _poker_target_aspect_valid,
)


def test_delegates_to_task2():
    image = np.zeros((594, 420, 3), np.uint8)
    pieces = [np.float64(((10, 10), (80, 10), (40, 70)))]
    mask = np.zeros(image.shape[:2], np.uint8)
    expected = (["actions"], {"topology_path": "standard"})
    with patch.object(
            geometry.Task2WhiteSolver, "solve_detected",
            return_value=expected) as solve_detected:
        actual = Task3PokerSolver().solve_detected(image, pieces, mask)
    assert actual is expected
    args, kwargs = solve_detected.call_args
    assert args == (image, pieces, mask)
    assert kwargs["candidate_validator"] is _poker_target_aspect_valid
    assert kwargs["use_default_search"] is True
    assert kwargs["solve_options"]["defer_fast_accept"] is True
    assert kwargs["solve_options"]["finalist_count"] == 12
    assert "return_best_candidate_on_failure" not in kwargs


if __name__ == "__main__":
    test_delegates_to_task2()
    print("所有测试通过")
