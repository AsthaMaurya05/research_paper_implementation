# Research Paper Implementations: Federated Learning and Its Privacy Risks

From-scratch PyTorch reproductions of two papers that tell one story: **federated learning keeps raw data on the device, but sharing gradients can still leak it.**

| # | Project | Paper | What it shows |
|---|---------|-------|---------------|
| 1 | [FedAvg](FedAvg/) | McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (AISTATS 2017) | How federated training works, and how non-IID data causes client drift |
| 2 | [DLG](DLG/) | Zhu, Liu & Han, *Deep Leakage from Gradients* (NeurIPS 2019) | How a shared gradient can be inverted to recover a private training image |

**Blog post:** [I built federated learning from scratch, then broke its privacy promise](https://medium.com/@asthamaurya8115/i-built-federated-learning-from-scratch-then-broke-its-privacy-promise-3a93c95b8c09?postPublishedType=initial)

---

## 1. FedAvg (`FedAvg/`)

Reproduces the FederatedAveraging algorithm on MNIST with 100 simulated clients.

**Setup**
- IID partition (600 examples per client) and the paper's non-IID partition (sort by label, 200 shards of 300, 2 shards per client)
- Two models, both matching the paper's parameter counts: 2NN (199,210 parameters) and CNN (1,663,370 parameters)
- Weighted averaging over the selected clients only, following the paper's erratum

**Results on non-IID data**

| Model | Rounds | Local epochs (E) | Final test accuracy |
|-------|--------|------------------|---------------------|
| 2NN | 50 | 1 | 88.18% |
| 2NN | 50 | 5 | 93.79% |
| 2NN | 50 | 20 | 95.40% |
| CNN | 25 | 1 | 90.88% |
| CNN | 25 | 5 | 94.62% |

(C = 0.1, B = 10; learning rate 0.05 for the 2NN and 0.01 for the CNN.)

**Findings**
- More local epochs gave a better final model, supporting the paper's computation-for-communication trade-off.
- On non-IID data, high E slowed early convergence and made accuracy oscillate (at E = 20 it moved between about 77% and 95% across rounds). This is **client drift**, the problem later methods such as SCAFFOLD target.
- The paper does not state the CNN padding. Working back from its parameter count shows `padding=2` is needed to reach 1,663,370 parameters.

## 2. DLG (`DLG/`)

Implements the gradient-inversion attack against the same 2NN model used in the FedAvg project.

**How the attack works:** start from a random dummy image and label, compute the gradient they would produce, and update the dummy so its gradient matches the one the client shared. This needs second-order derivatives, using `torch.autograd.grad(..., create_graph=True)`.

**Results (single MNIST image, 2NN trained for 5 epochs on 5,000 examples, about 93% test accuracy)**
- Gradient distance fell from about 1454 to about 0.14.
- The private label was recovered correctly (predicted 4, true label 4).
- The noise image sharpened into a recognizable digit over about 300 iterations. The recovery is clearly recognizable but not pixel-perfect.

**Scope and differences from the paper**
- Single image only, not batches.
- Uses the Adam optimizer; the paper uses L-BFGS.
- The defenses discussed in the paper (gradient noise, low precision, pruning) are described in the DLG README but **not implemented here**.

---

## Repository structure

```
research_paper_implementation/
├── FedAvg/
│   ├── FedAvg_MNIST.ipynb      # implementation, experiments, plots
│   ├── README.md
│   └── (paper PDF)
└── DLG/
    ├── DLG_Implementation_.ipynb   # attack code and visualization
    ├── Deep Leakage from Gradients.pdf
    └── README.md
```

## How to run

1. Open a notebook in Google Colab or Jupyter.
2. Run all cells. MNIST downloads automatically.
3. No GPU is needed. Rough CPU times: FedAvg 2NN about 15 minutes, FedAvg CNN about 10 minutes, DLG training about 30 seconds plus about 30-60 seconds for the attack.

**Requirements:** Python 3, PyTorch, torchvision, numpy, matplotlib.

## Possible next steps

- Implement the gradient-pruning defense and measure when DLG stops working.
- Attack a batch of images, and attack the FedAvg CNN.
- Run DLG against a client inside a full FedAvg training run.
- Implement SCAFFOLD to correct the client drift seen in the non-IID experiments.

## References

- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Aguera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017. arXiv:1602.05629
- Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. NeurIPS 32.
