# torchbcsoftmax
An implementation of Box-Constrained Softmax function in PyTorch.

## Installation
```bash
pip install git+https://github.com/neonnnnn/torchbcsoftmax
```
## Usage
This `bcsoftmax` package exposes following three functions: `bcsoftmax` (Box-Constrained Softmax), `lbsoftmax` (Lower-Bounded), and `ubsoftmax` (Upper-Bounded).
They require a logit tensor `x`, lower bound vector `lb` (for `bcsoftmax` and `lbsoftmax` functions), and upper bound tensor `ub` (for `bcsoftmax` and `ubsoftmax` functions).
```python
import torch
import bcsoftmax

logit = torch.tensor([-1.5, 1.0, -0.5])
torch.nn.functional.softmax(logit)
# tensor([0.0629, 0.7662, 0.1710])

ub = torch.tensor([1.0, 0.6, 0.5])
bcsoftmax.ubsoftmax(logit, ub)
# tensor([0.1076, 0.6000, 0.2924])

lb = torch.tensor([0.15, 0.2, 0.10])
bcsoftmax.lbsoftmax(logit, lb)
# tensor([0.1500, 0.6949, 0.1551])

bcsoftmax.bcsoftmax(logit, lb, ub)
# tensor([0.1500, 0.6000, 0.2500])
```

## Reference
```bibtex
@misc{atarashi2025box,
    title={Box-Constrained Softmax Function and Its Application for Post-Hoc Calibration},
    author={Kyohei Atarashi and Satoshi Oyama and Hiromi Arai and Hisashi Kashima},
    year={2025},
    eprint={2506.10572},
    archivePrefix={arXiv},
    primaryClass={stat.ML},
    url={https://arxiv.org/abs/2506.10572},
}
```
