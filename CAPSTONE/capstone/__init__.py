"""CAPSTONE: Composable Attribute-Prompted Scene Translation
for Zero-Shot Vision-Language Reasoning.

A modular, pixel-free framework that turns the outputs of off-the-shelf vision
models into structured text prompts for a frozen LLM. The three stages mirror
the paper:

    V : I -> F_v          perception   (capstone.perception)
    T : F_v -> D          translation  (capstone.translation)
    L : D x Q -> A        reasoning    (capstone.reasoning)
"""

__version__ = "0.1.0"

__all__ = ["perception", "translation", "reasoning", "eval", "utils"]
