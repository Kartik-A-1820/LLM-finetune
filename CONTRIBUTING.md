# Contributing to the Pythia LoRA lab

Begin with the offline test path in [README.md](README.md). For a larger training or evaluation change, open an issue with the hypothesis, baseline, and comparison protocol.

## A focused contribution

1. Fork the repository and create a branch.
2. Use an isolated Python environment. Reproduce the problem with a small fixture before downloading or training a model.
3. Add a regression test when behavior changes. Run `python -m pytest -q`; use PyTorch for the tensor-specific test.
4. Explain the change, validation, and any remaining limitations in the pull request.

## Experiment reports

Record the commit, package versions, model revision, hardware, split-manifest hashes, seed, training configuration, and generation settings. Compare base and adapter on the same held-out IDs and token budget. Do not select checkpoints or prompts using the test split.

The splitter groups normalized exact instructions; it does not guarantee semantic deduplication. Report paraphrase overlap and data provenance separately. Execution metrics require reviewed tests that assert the intended behavior; a successful exit is insufficient.

Do not commit adapter weights, generated environments, credentials, or private data. Changes to generated-code execution must preserve the default static-only behavior and explicit Docker opt-in.

## Useful starting points

- Add small, reviewed functional test cases to a redistributable dataset.
- Analyze near-duplicates before preparing the split.
- Report dependency installation failures with an exact command and traceback.
- Reproduce base-versus-adapter evaluation with a fresh training run.

There is no standalone LICENSE file. Identify third-party sources and terms in proposed additions; discuss repository reuse terms with the maintainer.
