# DLG Implementation — Deep Leakage from Gradients

A from-scratch PyTorch reproduction of the Deep Leakage from Gradients (DLG) attack from Zhu et al. (NeurIPS 2019). This project was built as part of a research application to the Visual Computing Lab at IISc, which works on privacy-preserving federated learning and investigating vulnerabilities in FL frameworks.

This is the second project in a series. The first project implements FedAvg (federated learning). This project attacks it.

---

## The Problem (Plain English)

In federated learning, clients train models on their private data and share only the gradients with a central server. The raw data never leaves the device. Everyone assumed this was safe — gradients are just numbers, right?

This paper shows that assumption is wrong. Given the gradient, the model architecture, and the current weights, an attacker can reconstruct the exact private training data. Not a similar-looking image — the actual image, pixel by pixel.

---

## The Attack (Intuition)

The idea is surprisingly simple: turn gradient descent against itself.

In normal training, you:
1. Feed real data through the model
2. Compute the gradient of the loss
3. Update the model weights to reduce the loss

In DLG, the attacker does the opposite:
1. Start with a random noise image and random label
2. Feed the noise through the model and compute what the gradient would be
3. Compare this dummy gradient to the real gradient (the one the client shared)
4. Update the noise image to make the gradients match
5. Repeat until the noise becomes the real image

When the dummy gradient matches the real gradient, the dummy data has converged to the real data. The attack is done.

The clever part is that this requires second-order derivatives: you backpropagate through a backpropagation step. PyTorch handles this with `create_graph=True`.

---

## How the Code is Organized

```
Module 1: Setup
  - Load MNIST
  - Define the TwoNN model (same architecture as FedAvg project)

Module 2: Train the model
  - Train 2NN for 5 epochs on 5000 MNIST examples
  - The model needs to have learned something for the gradients to carry
    meaningful information about the input
  - Reaches ~93% test accuracy

Module 3: The victim
  - Pick one private image from the test set
  - Compute the gradient of the loss w.r.t. all model parameters
  - This gradient (199,210 numbers) is the ONLY thing the attacker sees
  - The raw image stays private (we show it for comparison, but the
    attacker never sees it)

Module 4: The DLG attack
  - Initialize a random noise image and random label
  - Feed the dummy through the model, compute dummy gradients
  - Measure the distance between dummy and real gradients
  - Use gradient descent to update the dummy image until the
    gradients match
  - Uses torch.autograd.grad with create_graph=True for second-order
    derivatives
  - Uses Adam optimizer (the paper uses L-BFGS; Adam also works)

Module 5: Visualization
  - Show snapshots of the dummy image at different iterations
  - Show side-by-side comparison of recovered vs ground truth
```

---

## Key Results

The attack successfully recovers the private training image:

- Starting gradient distance: ~1454 (random noise, completely wrong)
- Final gradient distance: ~0.14 (gradients essentially match)
- Label recovery: correct (predicted 4, true label 4)
- Visual recovery: the noise image gradually sharpens into a
  recognizable digit over ~300 iterations

The recovered image is not pixel-perfect, but it is clearly recognizable and the label is correctly recovered. This is achieved using only:
- The gradient (199,210 numbers)
- The model architecture
- The current model weights

The attacker never sees the raw training data at any point.

---

## The Key Technical Detail: Second-Order Derivatives

The core challenge is computing the gradient of the gradient distance with respect to the dummy input. This requires differentiating through a backpropagation step.

In PyTorch:
1. First backward pass: compute dummy gradients using
   `torch.autograd.grad(loss, model.parameters(), create_graph=True)`
2. Compute the distance between dummy and real gradients
3. Second backward pass: compute the gradient of this distance
   w.r.t. the dummy image and dummy label

The `create_graph=True` flag tells PyTorch to keep the computation graph
alive after the first backward pass, so the second backward pass can
differentiate through it.

---

## Connection to FedAvg

This attack directly targets the federated learning system implemented in the companion project (FedAvg_MNIST.ipynb). In FedAvg, each client:
1. Downloads the global model
2. Trains locally on private data
3. Sends the updated weights (or gradients) to the server

The DLG attack shows that step 3 leaks the private training data from step 2. The privacy guarantee of federated learning is not as strong as assumed.

---

## Defenses (from the paper)

The paper discusses three defense strategies:

1. Gradient perturbation — add noise (Gaussian or Laplacian) to gradients.
   Works if noise scale is above 1e-2.

2. Low precision — use half-precision (fp16) gradients. Does not help much.

3. Gradient compression/pruning — zero out small gradient values.
   Works if more than 20% of gradients are pruned. This is the most
   effective defense that does not change the training setup.

---

## How to Run

1. Open the DLG notebook in Google Colab (or Jupyter locally)
2. Runtime -> Run all
3. MNIST downloads automatically
4. Training takes ~30 seconds
5. The attack takes ~30-60 seconds on CPU

No GPU required. The attack operates on a single image, so computation
is minimal.

**Requirements:** Python 3, PyTorch, torchvision, matplotlib
(all pre-installed in Colab)

---

## Files

| File | What it is |
|------|-----------|
| `DLG_MNIST.ipynb` | Complete implementation — attack code and visualization |
| `DLG_README.md` | This file |

---

## Reference

Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients.
Advances in Neural Information Processing Systems (NeurIPS), 32.

---

## What's Next

This implementation attacks a single image. Potential extensions:
- Attack a batch of images simultaneously (the paper covers this)
- Implement gradient pruning defense and show it stops the attack
- Attack the CNN model from the FedAvg project
- Apply the attack in a full federated learning setting (attack a
  real FedAvg client mid-training)
