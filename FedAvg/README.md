# FedAvg Implementation — Communication-Efficient Learning of Deep Networks from Decentralized Data

A from-scratch PyTorch reproduction of the FederatedAveraging (FedAvg) algorithm from McMahan et al. (2017). This project was built as part of a research application to the Visual Computing Lab at IISc, which works on Privacy-Preserving Federated Learning.

---

## What is Federated Learning? 

Normally, to train a machine learning model, you collect all the data in one place (a server) and train there. But a lot of useful data is private — the photos on your phone, the messages you type, your medical records. People don't want that data uploaded to a company's server.

**Federated Learning** flips the setup. Instead of bringing data to the model, it brings the model to the data. Your phone downloads the current model, trains it locally on your data, and sends back only the *updated weights* — never the raw data. A central server averages the updates from many devices to improve the shared model.

The key paper that started this field is McMahan et al. (2017), which introduced the **FederatedAveraging (FedAvg)** algorithm. This project reproduces it.

---

## The Paper in 60 Seconds

**Problem:** Train a shared model across many clients (phones), each holding private data, without the data ever leaving the device. Communication is the bottleneck — phones have slow connections and only talk occasionally.

**Solution (FedAvg):** Each round, the server sends the current model to a random subset of clients. Each client runs several epochs of local SGD on its own data, then sends back the updated weights. The server averages all the weights (weighted by how much data each client has). Repeat.

**Key result:** By doing more computation locally (more local epochs = E), you need far fewer communication rounds to reach a target accuracy — 10-100x fewer than naive federated SGD.

**Three knobs control everything:**
- **C** = fraction of clients selected each round (e.g. 0.1 = 10 out of 100)
- **E** = number of local training epochs per client per round
- **B** = local minibatch size

---

## Files

| File | What it is |
|------|-----------|
| `FedAvg_MNIST.ipynb` | Complete implementation — code, experiments, and plots. Run top to bottom. |
| `README.md` | This file. |

---

## What This Project Reproduces

| Component | Paper specification | This implementation |
|-----------|---------------------|---------------------|
| Dataset | MNIST | 60,000 train / 10,000 test |
| Clients | K = 100 | 100 clients |
| IID partition | 600 examples per client, shuffled | Yes |
| Non-IID partition | Sort by label, 200 shards of 300, 2 shards per client | Yes |
| Model 1 (2NN) | MLP, 2 hidden layers x 200 ReLU units | 199,210 params (matches paper) |
| Model 2 (CNN) | 2 conv layers (32, 64 channels, 5x5) + FC 512 + softmax | 1,663,370 params (matches paper) |
| Algorithm | FederatedAveraging with weighted averaging | Yes, with erratum correction |
| Core experiment | Varying E shows fewer rounds needed | Yes |

### A note on the CNN architecture

The paper states 1,663,370 parameters but does not explicitly mention padding. Working backwards from the parameter count, the convolution layers must use `padding=2` (same-padding) to produce 7x7 feature maps after two max-pools. Without padding, the parameter count would not match. This was verified by tracing the spatial dimensions through the forward pass.

---

## Key Results

### 2NN on Non-IID data (50 rounds, C=0.1, B=10, lr=0.05)

| E (local epochs) | Final accuracy | Observation |
|-------------------|----------------|-------------|
| 1 | 88.18% | Fast early, low ceiling |
| 5 | 93.79% | Balanced |
| 20 | 95.40% | Slow start, best final accuracy |

**Finding:** More local epochs (E) gives a better final model, but on non-IID data, high E causes slower early convergence. This is the signature of **client drift** — each client overfits its narrow data slice (only 2 digit classes) before sending weights back, pulling the global model in conflicting directions.

### CNN on Non-IID data (25 rounds, C=0.1, B=10, lr=0.01)

| E (local epochs) | Final accuracy |
|-------------------|----------------|
| 1 | 90.88% |
| 5 | 94.62% |

**Finding:** The CNN outperforms the 2NN in both settings, and E=5 beats E=1, confirming the result generalizes across model architectures.

### The client drift observation

The accuracy curves on non-IID data oscillate rather than climbing smoothly. At E=20, accuracy bounces between 77% and 95% across rounds. This is not a bug — it is the phenomenon that motivates later algorithms like **SCAFFOLD**, which uses control variates to correct for client drift and stabilize convergence.

---

## How the Code is Organized

The notebook `FedAvg_MNIST.ipynb` is structured in clear modules:

```
Module 1: Data loading and partitions
  - Load MNIST with normalization
  - IID partition (shuffle, split into 100 equal chunks)
  - Non-IID partition (sort by label, 200 shards of 300, 2 per client)
  - DataLoader helper (turn client indices into batched iterator)

Module 2: Models
  - TwoNN: MLP with 2 hidden layers (200 units, ReLU), 199,210 params
  - CNN: 2 conv layers + FC, 1,663,370 params
  - Both verified against paper's stated parameter counts

Module 3: Client (client_update)
  - Load global weights into a fresh local model
  - Run E epochs of local SGD with batch size B
  - Return trained weights (not gradients) and data size n_k

Module 4: Server (fedavg_round)
  - Sample C-fraction of clients
  - Each client trains locally
  - Weighted-average weights by n_k (with erratum correction)

Module 5: Training loop (run_fedavg + evaluate)
  - Initialize fresh global model
  - Run T rounds, evaluate on test set every 5 rounds
  - Track and plot accuracy

Module 6: Experiments
  - 2NN: E in {1, 5, 20}, Non-IID, 50 rounds
  - CNN: E in {1, 5}, Non-IID, 25 rounds
  - Plots showing convergence speed vs. local epochs
```

---

## How to Run

1. Open `FedAvg_MNIST.ipynb` in Google Colab (or Jupyter locally)
2. Runtime -> Run all
3. MNIST downloads automatically on first run
4. The 2NN experiment takes ~15 minutes on free Colab CPU
5. The CNN experiment takes ~10 minutes on free Colab CPU

No GPU is required. The models are small enough that CPU is sufficient.

**Requirements:** Python 3, PyTorch, torchvision, numpy, matplotlib (all pre-installed in Colab)

---

## Key Concepts Demonstrated

1. **Federated Averaging** — local SGD + weighted model averaging, the foundation of FL
2. **Non-IID data** — clients hold data from different distributions (here: only 2 of 10 digit classes), the hard case for FL
3. **Client drift** — over-training locally causes weights to diverge, visible as oscillation in non-IID accuracy curves
4. **Communication-computation tradeoff** — more local work (higher E) reduces communication rounds needed
5. **Weighted averaging with erratum** — the aggregation normalizes over selected clients only (per the paper's erratum), not all clients

---

## Reference

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Aguera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017. arXiv:1602.05629

---

## Medium Blog:
https://medium.com/@asthamaurya8115/i-built-federated-learning-from-scratch-then-broke-its-privacy-promise-3a93c95b8c09?postPublishedType=initial

## What's Next

This implementation serves as the foundation for two follow-up papers:
- **Deep Leakage from Gradients (DLG)** — using this FedAvg client as the attack target to reconstruct private data from shared gradients
- **SCAFFOLD** — correcting the client drift observed in the non-IID experiments using control variates
