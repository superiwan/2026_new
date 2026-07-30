"""Mode-specific solvers sharing the PieceAction output contract."""

from .task1_fixed import Task1FixedSolver
from .task2_white import Task2WhiteSolver
from .task3_poker import Task3PokerSolver

__all__ = ("Task1FixedSolver", "Task2WhiteSolver", "Task3PokerSolver")
