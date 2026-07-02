# pytorch-summer-2026

Self-paced PyTorch curriculum: tensors → autograd → training loops → models, working from a quant/finance background rather than a general ML one. Examples lean on finance-flavored data (returns, portfolios, factor models) instead of generic toy datasets.

## Setup

Notebooks run against the `ptlearn` conda environment.

```bash
conda activate ptlearn
# or point Jupyter/VSCode at /opt/anaconda3/envs/ptlearn/bin/python
```

If the `ptlearn` Jupyter kernel isn't registered yet:

```bash
/opt/anaconda3/envs/ptlearn/bin/python -m ipykernel install --user --name ptlearn --display-name "Python (ptlearn)"
```

Core deps: `torch`, `numpy`, `matplotlib`.

## Structure

```
week1_basics/
  pytorch_intro.ipynb                    tensors, numpy interop, reshape/matmul/broadcasting,
                                          autograd basics, a full nn.Module training loop
  week1_day2_tensor_ops.ipynb / .py      reshape, matmul, broadcasting deep dive
  week1_day3_autograd.ipynb              requires_grad, backward(), gradient accumulation
                                          and zero_grad, no_grad/detach, manual gradient descent
  week1_day4_linear_regression_scratch.ipynb
                                          exercise: linear regression with hand-derived
                                          gradients (no autograd, no nn.Linear, no optim),
                                          checked against autograd, plus closed-form OLS
```

Each day's notebook builds on the previous one's mechanics. Exercise notebooks (like day 4) are scaffolds — TODO stubs with hints and self-check cells, not filled-in solutions; the point is deriving the math yourself and using autograd/closed-form solutions as a check, not a shortcut.

## Conventions

- One notebook per day, `week{N}_day{D}_{topic}.ipynb`.
- Exercise notebooks end with a "Try yourself" section for further practice beyond the core exercise.
- Concept notebooks (day 3 style) end with a summary table of what was covered.
