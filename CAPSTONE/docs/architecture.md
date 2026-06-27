# CAPSTONE Architecture

CAPSTONE decouples *perception* from *reasoning*. No module is ever trained or
fine-tuned — everything is frozen and hot-swappable. The pipeline is a function
composition:

```
a_hat = L( T( V(I) ), q )
```

where `V` is the vision stack, `T` is the symbolic translator, `L` is the
frozen LLM, `I` is the image and `q` is the question.

```
┌───────────────────────────────────────────────────────────────────────────┐
│  (a) Input: image I + question q                                            │
└───────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  (b) V : I → F_v   —  capstone/perception   (all frozen)                    │
│      • YOLO detection  → labels, boxes, confidence, quadrant                │
│      • K-means colors  → dominant colors per object (HSV, K=3)              │
│      • Pose estimation → standing / sitting / profile / raised-arms        │
│      • Depth (DPT)     → near/mid/far distribution per object               │
│      • OCR (EasyOCR)   → detected text + location                          │
└───────────────────────────────────────────────────────────────────────────┘
                  │ raw structured outputs
                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  (c) T : F_v → D   —  capstone/translation                                  │
│      Compiles d_objects, d_relations, d_scene into one declarative,         │
│      metric, relational text description. Serialized to JSON/CSV so the     │
│      two stages can be cached and run independently.                        │
└───────────────────────────────────────────────────────────────────────────┘
                  │ prompt P = C(D, q) = [instruction; D; q]
                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  (d) L : D × Q → A   —  capstone/reasoning   (frozen LLM)                   │
│      Premise Extraction → Relational Encoding → Chain-of-Thought →          │
│      Answer Synthesis  →  "Answer: [YES/NO]"                                │
└───────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
              capstone/eval  →  accuracy / precision / recall / F1
```

## Why decouple?

* **Cost & speed** — classic CV modules are tiny, CPU-friendly, and run at
  100+ FPS; only the LLM is heavy. No multimodal paired training is needed.
* **Interpretability** — because the LLM reasons over *text*, every decision is
  traceable to a specific detection / attribute, which lets you attribute an
  error to either the CV module or the LLM (paper Tables 5–6).
* **Modularity** — swap YOLOv8 → YOLOv11, or Qwen2.5 → Qwen3, without touching
  any other component or retraining anything.

## Module map

| Paper concept            | Package                       | Key entry points                        |
|--------------------------|-------------------------------|-----------------------------------------|
| `V` (perception)         | `capstone.perception`         | `load_models`, `detect_objects`         |
| `T` (translation)        | `capstone.translation`        | `display_detections`, `prepare_for_json`|
| `L` (reasoning)          | `capstone.reasoning`          | `load_llm_model`, `process_data_in_batches` |
| `C` (prompt construction)| `capstone.reasoning.prompts`  | `create_prompt`                         |
| evaluation               | `capstone.eval`               | `parse_yes_no`, `compute_metrics`       |
