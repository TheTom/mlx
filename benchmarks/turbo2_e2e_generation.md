# Item 9: turbo2 E2E Generation Coherence Test

## Test Setup
- Model: Qwen3.5-2B-8bit
- Prompt: "Explain the theory of relativity in simple terms."
- Max tokens: 256
- Configs: turbo2 asymmetric, turbo2 symmetric, FP16 baseline

## Results

| Config         | Speed (tok/s) | Coherent? | Notes                          |
|----------------|--------------|-----------|--------------------------------|
| FP16 baseline  | 200.9        | Yes       | Clean structured output        |
| turbo2 asym    | 170.1        | Yes       | Full reasoning trace, coherent |
| turbo2 sym     | 167.6        | Yes       | Full reasoning trace, coherent |

## Output Quality
All three outputs are fully coherent, well-structured, and factually accurate.
turbo2 outputs produce detailed thinking/reasoning traces with proper categorization
and logical flow. No degeneration, repetition, or garbage detected.

## Speed
turbo2: ~170 tok/s (85% of FP16 baseline at 201 tok/s).
Memory: identical peak (2.17 GB) — compression benefit not visible at this short context.

## Verdict
turbo2 is viable for generation on Qwen3.5-2B. Output quality is indistinguishable
from FP16 at 256 tokens. Would need longer context (4K+) to see quality degradation.
