# Pythia LoRA code-generation lab

A reproducible fine-tuning and evaluation pipeline for `EleutherAI/pythia-410m`.
The original notebooks and synthetic code datasets remain available as historical
experiments. The supported workflow below adds instruction-disjoint splits,
response-only training, validation-based checkpoint selection, and held-out
completion evaluation.

## Why evaluation changed

The old generation notebooks scored the first 100 examples from
`datascience_1000_multistep.jsonl`, also used as a training source. The original
ROUGE-1 result of 0.8763 is therefore **not evidence of held-out functional code
correctness**. No replacement model-quality score is claimed until a fresh model
is trained and evaluated with the new split manifest. Previously trained adapters
must not be used to claim a clean held-out result on these new splits.

ROUGE measures text overlap, not whether a Python program works. The new evaluator
reports syntax validity and exact match, and optionally test-backed pass@1 inside
a resource-limited, network-disabled Docker container. The supplied datasets lack
curated execution tests; pass@1 remains null until tests are supplied.

## Reproduce

Python 3.10+; install a PyTorch build appropriate to your CPU/CUDA platform first:

```bash
python -m venv .venv
# Activate .venv, then:
pip install -r requirements.txt
python prepare_data.py data/datascience_1000_multistep.jsonl data/datascience_4000_multistep.jsonl --output splits
python train.py --splits splits --output outputs/adapter --epochs 1 --max-length 512
# On a supported CUDA/bitsandbytes installation, optionally add --four-bit.
# Install bitsandbytes separately for that hardware; it is not required for CPU.
python generate.py --splits splits --output outputs/base.jsonl
python generate.py --splits splits --adapter outputs/adapter --output outputs/tuned.jsonl
python evaluate.py outputs/base.jsonl --splits splits
python evaluate.py outputs/tuned.jsonl --splits splits
```

Long examples fail explicitly rather than silently truncating code. Increase
`--max-length` within your memory budget or curate the source data before creating
splits. Pythia's fused `query_key_value` modules are the LoRA targets; another
model architecture may require different targets. CPU training is supported but
can be slow. Four-bit mode requires compatible CUDA and bitsandbytes.

The splitter groups exact whitespace/case-normalized instructions, keeps the
first occurrence, and creates seeded approximately 80/10/10 train/validation/test
partitions. File hashes, source hashes, counts, and the seed are saved in
`splits/manifest.json`; training/generation verify them. Only train and validation
are used by the Trainer. Prompt tokens and padding receive label -100; real EOS
remains supervised. Adapter output includes the split manifest.

Grouping does not detect paraphrases, semantic duplicates, or upstream model
pretraining contamination. Review near-duplicates before making generalization
claims. Source ordering determines the representative of duplicate instructions.

## Optional functional evaluation

Add reviewed `tests` strings (Python assertions with no external dependencies) to
source JSONL records **before preparing splits**. Each test must actually assert
correct behavior; blank/no-op tests do not make a useful benchmark. Use a trusted,
already pulled Python Docker image, preferably pinned by digest:

```bash
python evaluate.py outputs/tuned.jsonl --splits splits --docker-image python:3.11-slim
```

This explicit option executes generated code only in Docker, never in the host
Python process. Containers have no network, a read-only root, an unprivileged
user, resource/pid limits, a read-only case mount, and timeout cleanup. Execution
results apply only to examples with supplied tests. These controls are for a
local experiment, not a hostile multi-tenant code-execution service. The default
static evaluator executes no completions.

Generation uses greedy decoding, identical token budgets for base/tuned models,
and decodes only newly generated tokens. Do not tune generation settings or
select checkpoints against the test split. The evaluator rejects missing,
duplicate, or unknown prediction IDs. Syntax validity uses compilation without
execution, so module-level return/break are rejected. Execution requires a
harness receipt emitted only after tests finish; a zero exit status alone is
insufficient. The local harness is not an adversarially tamper-proof judge.

## Validation

```bash
pip install -r requirements-test.txt
python -m pytest -q
```

Tests verify deterministic grouping, tamper detection, complete prediction
coverage, code scoring, and response/padding masks (the tensor check needs
PyTorch). Full model training, CUDA four-bit compatibility, and Docker execution
require their own environment and are not established by these unit tests.

## Repository map

- `prepare_data.py`: grouped splits and provenance verification.
- `train.py`: LoRA training and validation checkpoint selection.
- `generate.py`: held-out base/adapter completions.
- `evaluate.py`: static metrics and optional Docker execution.
- `data/`: original synthetic datasets.
- `*.ipynb`, `imgs/`: archived experiments and outputs.

References: [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer),
[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
