# Convenience targets for the CAPSTONE pipeline.
# Override variables on the command line, e.g.:
#   make reason MODEL=Qwen/Qwen3-4B  PROMPT=direct

DATASET ?= cambridgeltl/vsr_zeroshot
SPLIT   ?= test
DETECTOR ?= yolov8l.pt
ENHANCED ?= vsr_detection_results_enhanced.csv
MODEL   ?= Qwen/Qwen2.5-7B-Instruct
PROMPT  ?= chain_of_thought
OUTDIR  ?= outputs

.PHONY: install perceive reason eval all clean

install:
	pip install -r requirements.txt && pip install -e .

perceive:
	python scripts/run_perception.py --dataset $(DATASET) --split $(SPLIT) \
		--detector $(DETECTOR) --output $(ENHANCED)

reason:
	python scripts/run_reasoning.py --input $(ENHANCED) --model $(MODEL) \
		--prompt-style $(PROMPT) --output-dir $(OUTDIR)

eval:
	python scripts/evaluate.py --input $(OUTDIR)/final_results.csv \
		--pred-col llm_caption --label-col answer --by-relation

all: perceive reason eval

clean:
	rm -rf $(OUTDIR) *.log __pycache__ */__pycache__ */*/__pycache__ \
		*_raw.csv processing_checkpoint.txt intermediate_results_*.csv
