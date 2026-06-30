# Failure Mode ID
Evaluation of open models (Qwen family) against Complex Constraints Benchmark Set (Surge AI) using LLM-as-a-judge (Opus 4.8). Incudes failure-mode analysis and bias audit.


## Data & Attribution
This project evaluates models against the **Complex Constraints Benchmark Set**,
released by Surge AI under CC-BY-4.0.
- Source: <https://huggingface.co/datasets/surgeai/ComplexConstraints>
- License: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)

- Modifications: generated model responses, derived per-criterion pass/fail
  grades via an LLM judge, and classified criteria for verifiability and
  reward-hackability. These derived artifacts and all code are MIT-licensed.

