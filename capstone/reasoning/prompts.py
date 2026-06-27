"""Prompt construction P = C(D, q) = [instruction; D; q].

CAPSTONE's accuracy hinges on how the symbolic description ``D`` is wrapped
into a prompt. We expose two styles:

* ``chain_of_thought`` (default) — elicits step-by-step reasoning before a
  final ``Answer: [YES/NO]``. This is the variant that drives CAPSTONE's large
  recall / F1 gains on VSR (paper Table 8: CoT vs Direct).
* ``direct`` — the compact POPE template from the paper's Appendix A.2, which
  asks the model to return only the final answer.
"""

from functools import lru_cache


@lru_cache(maxsize=4096)
def chain_of_thought_prompt(detection_results: str, question: str) -> str:
    """Verbose CoT prompt that exposes the model's reasoning chain."""
    return f"""You are an AI assistant tasked with analyzing image detection results.

I have detected the following objects and their details in an image: {detection_results}.

Based on this information, please answer the following question:
Question: {question}

Instructions:
1. Analyze the detected objects and their properties carefully
2. Consider the context implied by the question
3. Provide step-by-step reasoning that explains your analysis
4. End with a clear structured answer in the format: "Answer: [YES/NO]"

Remember to:
- Connect the detected objects to the question being asked
- Consider spatial relationships between objects when relevant
- Use logical deduction based only on the available information
- Provide your best assessment even if the information is incomplete
- Format your final answer in capital letters (YES or NO) for clarity
"""


@lru_cache(maxsize=4096)
def pope_direct_prompt(detection_results: str, question: str) -> str:
    """Compact POPE template (paper Appendix A.2) — final answer only."""
    return f"""You are a visual reasoning assistant tasked with answering real-world \
questions about an image using detected objects and scene context.

Image Analysis: {detection_results}

Question: {question}

Instructions:
- Use object names, locations, and relationships to reason about the scene.
- Apply commonsense and spatial reasoning to answer correctly.
- Think step-by-step if needed, but keep the final response short.
- Return only the final answer.
- Format the output as: Answer: [YES/NO]
"""


_PROMPT_BUILDERS = {
    "chain_of_thought": chain_of_thought_prompt,
    "cot": chain_of_thought_prompt,
    "direct": pope_direct_prompt,
    "pope": pope_direct_prompt,
}


def create_prompt(detection_results: str, question: str, style: str = "chain_of_thought") -> str:
    """Build a prompt in the requested ``style`` (default chain-of-thought)."""
    try:
        builder = _PROMPT_BUILDERS[style.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt style '{style}'. "
            f"Choose from: {sorted(set(_PROMPT_BUILDERS))}"
        ) from exc
    return builder(detection_results, question)
