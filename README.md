# Physics-Informed Recurrent Neural Networks (PRNN) for Homogenization

[![Python](https://img.shields.io/badge/Python-3.10%252B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%252B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, domain-driven Python library designed for multiscale material homogenization using **Physics-Informed Recurrent Neural Networks (PRNN)**. This framework bridges high-fidelity Representative Volume Element (RVE) simulations with efficient surrogate constitutive models, ensuring strict thermodynamic and physical consistency.

---

## 🚀 Key Features

- **Domain-Driven Architecture**: Clean separation of concerns featuring custom abstract interfaces, modular factories, and explicit data pipelines.
- **Physical Constraint Enforcement**: 
  - *Strict Positivity*: Enforced via Softplus layers to prevent non-physical negative volume stiffness.
  - *Sparsity & Conservation*: Decoupled topologies with weight normalization ($\sum V_f = 1.0$) ensuring partition of unity for strain-stress integration.
- **Robust Recurrent Evolution**: Unrolled temporal pathways designed to predict stress states from complex strain paths under cyclic or monotonic loading.
- **Advanced Training Control**: Built-in early stopping, gradient clipping (to prevent exploding gradients in recurrent unrolling), and GPU device-agnostic execution.
- **Specialized Loss Functions**: Custom relative error metrics with stabilization barriers for near-zero stress unloading phases.
- **C++/LibTorch Ready**: Designed with serializable states for seamless integration into finite element solvers.

---

## 🛠️ Project Structure

```text
prnn_project/
├── prnn/
│   ├── __init__.py
│   ├── interfaces.py       # Abstract base classes and structural contracts
│   ├── materials.py        # Constitutive models and vector equations
│   ├── models.py           # PRNNCell and sequential time-unrolling architectures
│   ├── layers.py           # Physics-constrained homogenization layers (Soft, Sparse, etc.)
│   └── utils.py            # Datasets, normalizers, loss functions, and Trainer loop
├── tests/                  # Unit and integration tests
├── scripts/                # Training and evaluation runner scripts
└── README.md