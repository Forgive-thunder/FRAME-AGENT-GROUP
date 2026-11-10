"""FRAME agents — re-exported for clean imports from ``main.py``."""
from agents.base import BaseAgent
from agents.extractor import ExtractorAgent
from agents.checker import CheckerAgent
from agents.reflector import ReflectorAgent
from agents.filter import FilterAgent
from agents.generator import GeneratorAgent
from agents.evaluator import EvaluatorAgent
from agents.integrator import IntegratorAgent

__all__ = [
    "BaseAgent",
    "ExtractorAgent",
    "CheckerAgent",
    "ReflectorAgent",
    "FilterAgent",
    "GeneratorAgent",
    "EvaluatorAgent",
    "IntegratorAgent",
]
