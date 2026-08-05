---
license: mit
---
This is a high-fidelity reasoning dataset synthesized using Claude Opus 4.6. The dataset is designed to capture the model's internal "Chain of Thought" and reasoning traces, specifically focusing on mathematical accuracy and structured logical deduction.
The dataset is intended for Supervised Fine-Tuning (SFT) and Distillation, allowing smaller open-source models to inherit the sophisticated reasoning patterns of Claude Opus 4.6.
Dataset Description
This collection combines high-difficulty math problems (GSM8K, MATH) with general-purpose logic puzzles and multi-step instructions. Each row includes a hidden reasoning trace where the model "thinks" through the problem before providing the final answer.
By exposing the fine-tuned model to these internal monologues, the resulting model learns process-oriented thinking rather than just pattern-matching for answers.
Why Simple Logic & Math Improves Reasoning
Fine-tuning on "Simple Logic and Math" serves as a cognitive foundation for LLMs for several reasons:
Rule Adherence: Math requires strict following of operations. Training on these paths reduces "hallucinations" in non-math tasks.
Step-by-Step Verification: These examples force the model to break down complex problems into smaller, verifiable units.
Cross-Domain Generalization: The ability to solve a "simple" logic puzzle translates into better coding, legal analysis, and structured writing, as all these tasks rely on the same underlying cognitive architecture of premise → deduction → conclusion.
Stats
## Teacher Model: [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
**Total Cost: $ 87.20 (USD)**

**Total Tokens (Input + Output): 27.2 M**

**Format: JSONL (Conversational with Reasoning Traces)**

**Primary Categories: Mathematics, Symbolic Logic, General Purpose Problem Solving**

### Usage
This dataset is optimized for fine-tuning models such as Qwen3.5 27b,25b a3b, 9b, 4b, 2b, 0.8b to increase their performance on benchmarks like BigBench Hard and GSM8K without increasing their parameter count.