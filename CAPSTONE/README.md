<div align="center">

# CAPSTONE

### Composable Attribute-Prompted Scene Translation for Zero-Shot Vision–Language Reasoning

[![EMNLP 2025](https://img.shields.io/badge/EMNLP_2025-Industry_Track-blue)](https://aclanthology.org/venues/emnlp/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code](https://img.shields.io/badge/code-github-black.svg)](https://github.com/ismail31416/CAPSTONE)

*A lightweight, modular, **pixel-free** framework that turns off-the-shelf vision
model outputs into structured text prompts for a **frozen LLM** — enabling
zero-shot visual reasoning with no multimodal training.*

</div>

---

## Overview

Training and maintaining Vision–Language Models (VLMs) is expensive and opaque.
**CAPSTONE** takes a different route: instead of fusing pixels and text inside a
trained multimodal model, it (1) runs cheap, frozen classical vision modules,
(2) translates their outputs into a structured natural-language description of
the scene, and (3) hands that description to a **frozen** Large Language Model
that reasons over it in text.

```
a_hat = L( T( V(I) ), q )
```

The result is a plug-and-play system that is **cheaper, faster, and fully
interpretable**, yet competitive with — and on some benchmarks better than —
end-to-end VLMs. Using a 7B LLM, CAPSTONE outperforms several fully-trained
VLMs on **POPE**; with a 4B model it sets a new state of the art in F1 on
**VSR**.

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline
diagram.

## Key ideas

- **Decouple vision from reasoning.** Classic CV modules emit *symbolic
  attributes*; a frozen LLM consumes them directly — no Q-formers, no
  multimodal encoders, no paired image–text training.
- **Symbolic scene translation.** Labels, boxes, depth, pose, colors and OCR
  become *declarative, metric and relational* sentences the LLM can reason over.
- **Hot-swappable backends.** Upgrade YOLOv8 → YOLOv11 or Qwen2.5 → Qwen3
  without retraining anything else.
- **Built-in interpretability.** Every answer traces back to a specific
  detection or attribute, so errors can be attributed to the CV module or the
  LLM.

## Repository structure

```
CAPSTONE/
├── capstone/                  # installable package
│   ├── perception/            # V : I → F_v   (frozen vision stack)
│   │   ├── models.py          #   load YOLO / DPT / EasyOCR / pose
│   │   ├── detector.py        #   detect_objects: the full perception pass
│   │   ├── color.py           #   K-means dominant colors
│   │   ├── pose.py            #   pose → activity label
│   │   ├── geometry.py        #   quadrant / confidence banding
│   │   └── visualize.py       #   optional qualitative plots
│   ├── translation/           # T : F_v → D   (symbolic description)
│   │   ├── formatter.py       #   display_detections / prepare_for_json
│   │   └── extractors.py      #   pull OCR / depth / pose back out of JSON
│   ├── reasoning/             # L : D × Q → A (frozen LLM)
│   │   ├── llm.py             #   load_llm_model
│   │   ├── prompts.py         #   CoT + POPE prompt templates
│   │   └── inference.py       #   batched, checkpointed generation
│   ├── eval/                  # answer parsing + accuracy/P/R/F1
│   └── utils/                 # serialization helpers
├── scripts/
│   ├── run_perception.py      # Stage 1 → enhanced CSV
│   ├── run_reasoning.py       # Stage 2 → final_results.csv
│   └── evaluate.py            # Stage 3 → metrics
├── configs/default.yaml       # paper hyperparameters (reference)
├── docs/architecture.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
git clone https://github.com/ismail31416/CAPSTONE.git
cd CAPSTONE

python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# install the package itself (enables the capstone-* CLI entry points)
pip install -e .
```

> **GPU note.** The frozen CV modules are CPU-friendly and run at 100+ FPS. The
> LLM backend benefits from a GPU; the reasoning script auto-selects a batch
> size from available GPU memory and falls back to CPU (batch size 4) otherwise.

## Quickstart

The pipeline runs in three stages. Each stage writes a file the next one reads,
so you can cache, resume, or swap components freely.

### 1 — Perception + Translation

```bash
python scripts/run_perception.py \
    --dataset cambridgeltl/vsr_zeroshot --split test \
    --output vsr_detection_results_enhanced.csv \
    --detector yolov8l.pt
```

Produces an enhanced CSV where each row carries a `formatted_results` column
(the structured scene description) plus `ocr_text`, `depth_info`, `pose_info`
and `object_count`.

Run on a local folder of images instead:

```bash
python scripts/run_perception.py \
    --from-images ./my_images \
    --question "Is there a person in the image?" \
    --output my_detection_results_enhanced.csv
```

### 2 — Frozen-LLM Reasoning

```bash
python scripts/run_reasoning.py \
    --input vsr_detection_results_enhanced.csv \
    --model Qwen/Qwen2.5-7B-Instruct \
    --prompt-style chain_of_thought \
    --output-dir outputs
```

Writes `outputs/final_results.csv` with an `llm_caption` column containing the
model's step-by-step reasoning and a final `Answer: [YES/NO]`. Progress is
checkpointed to `outputs/processing_checkpoint.txt`, so interrupted runs resume.

### 3 — Evaluation

```bash
python scripts/evaluate.py \
    --input outputs/final_results.csv \
    --pred-col llm_caption --label-col answer \
    --by-relation
```

Parses YES/NO predictions and reports accuracy / precision / recall / F1
(with an optional per-relation-type breakdown for VSR).

## Choosing a backend

| Benchmark | Best backend (paper)   | Notes                                   |
|-----------|------------------------|-----------------------------------------|
| POPE      | `Qwen2.5-7B`           | best object-presence accuracy           |
| VSR       | `Qwen3-4B-Thinking`    | best recall / F1 with a modest 4B model |

Smaller backends (LLaMA-3.2-1B, Qwen2.5-1.5B, DeepSeek-R1-1.5B) are supported
but reason less reliably — performance scales with the LLM's reasoning ability,
not just parameter count.

## Prompting

Two prompt styles are exposed via `--prompt-style`:

- **`chain_of_thought`** *(default)* — elicits explicit reasoning before the
  answer. This is what drives the large recall / F1 gains on VSR.
- **`direct`** — the compact POPE template (final answer only).

The ablation in the paper shows CoT lifts VSR recall from ~52.7% to ~89.5% and
F1 from ~53.4% to ~67.3%, while accuracy and precision stay comparable — i.e.
CoT mainly strengthens *relational* reasoning over the symbolic outputs.

## Results

### POPE (accuracy, %)

| Method | LLM | Random | Popular | Adversarial |
|---|---|---|---|---|
| mPLUG-Owl2 | LLaMA2-7B | 86.70 | 83.66 | 81.73 |
| Ovis2-8B | 8B | 86.60 | 86.00 | 85.90 |
| Ovis2-4B | 4B | 86.20 | 86.10 | 85.50 |
| **CAPSTONE (Ours)** | **Qwen2.5-7B** | **87.47** | **87.17** | **85.93** |

### VSR (Visual Spatial Reasoning, %)

| Method | LLM | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Spatial-LLaVA-7b | Vicuna-7B | 53.60 | 54.77 | 52.61 | 47.08 |
| Qwen-VL-Chat | Qwen-7B | 53.44 | 58.30 | 52.17 | 41.98 |
| **CAPSTONE (Ours)** | **Qwen3-4B-Thinking** | **55.24** | 53.92 | **89.51** | **67.30** |

### Efficiency (per 1B images, GPU @ \$1.00/hr)

| Pipeline | Latency | Throughput | Params | Cost (USD) |
|---|---|---|---|---|
| 3B end-to-end VLM | 66.3 ms | 15.1 FPS | 3.75B | 18,395.9 |
| 8B end-to-end VLM | ~95.0 ms | ~10.5 FPS | 8.29B | 26,455.0 |
| **CAPSTONE (CV + 3B LLM)** | **60.9 ms** | **16.1 FPS** | **3.15B** | **17,274.7** |

CAPSTONE cuts cost by ~6% vs a comparable 3B VLM and ~35% vs an 8B VLM (with
~62% fewer parameters), while keeping throughput competitive.

## Datasets

- **POPE** — binary object-presence detection for hallucination resistance,
  evaluated under Random / Popular / Adversarial regimes.
- **VSR** — Visual Spatial Reasoning; measures relational understanding with
  accuracy, precision, recall and F1.

The perception script loads VSR directly from the Hugging Face Hub
(`cambridgeltl/vsr_zeroshot`). Point `--dataset` / `--split` at your POPE source
or use `--from-images` for arbitrary image folders.

## Interpretability

Because reasoning happens over text, CAPSTONE can disentangle perception
failures from reasoning failures. In the paper's study, ~6% of errors trace to
the CV module (missed detections), ~3% to LLM reasoning (ignoring valid
detections), and ~1% to mixed cases — actionable debugging signal that a
black-box VLM cannot provide.

## Limitations

CAPSTONE currently relies on a standard object detector and a limited set of
basic attributes, which can constrain capture of nuanced or abstract scene
content. Richer detectors with larger class vocabularies and more semantic
attributes would give the LLM more to reason over. The framework's modularity
is designed to make exactly these swaps painless.

## Citation

If you use this code or build on CAPSTONE, please cite:

```bibtex
@inproceedings{hossain2025capstone,
  title     = {CAPSTONE: Composable Attribute-Prompted Scene Translation
               for Zero-Shot Vision--Language Reasoning},
  author    = {Hossain, Md. Ismail and Ridoy, Shahriyar Zaman and
               Farazi, Moshiur and Mohammed, Nabeel and Rahman, Shafin},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in
               Natural Language Processing: Industry Track},
  pages     = {2840--2851},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}
```

## Acknowledgements

Built on excellent open-source projects: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
[EasyOCR](https://github.com/JaidedAI/EasyOCR), Intel
[DPT](https://huggingface.co/Intel/dpt-large), and the
[Qwen](https://github.com/QwenLM) / LLaMA / DeepSeek model families via
🤗 [Transformers](https://github.com/huggingface/transformers).

## License

Released under the [MIT License](LICENSE).
