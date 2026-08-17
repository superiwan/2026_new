"""Mode-specific solvers sharing the PieceAction output contract."""

from .task2_white import Task2WhiteSolver
from .task3_poker import Task3PokerSolver

__all__ = ("Task2WhiteSolver", "Task3PokerSolver")
