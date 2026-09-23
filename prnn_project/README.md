# PRNN: Physically Recurrent Neural Networks for Surrogate Constitutive Modeling

## Project Overview
This repository implements a Physically Recurrent Neural Network (PRNN) designed for surrogate constitutive modeling in computational mechanics. The architecture maps macroscopic strain sequences to macroscopic stresses by embedding a fully vectorized physical material layer (e.g., J2 Plasticity) between an encoder (localization) and a decoder (homogenization) network. 

The codebase follows strict Object-Oriented Programming (OOP) principles and PEP 8 guidelines, completely decoupling the PyTorch neural network unrolling logic from the internal return-mapping physics. This modular design allows for seamless integration of custom material models and customized topological constraints, such as the `SparseNormalizedLayer`.

## Repository Structure
The project is organized into a standard Python package structure to separate core logic from executable scripts and datasets.

```text
prnn_project/
├── prnn/                 # Core Python package (OOP Library)
│   ├── interfaces.py     # Abstract contracts (AbstractMaterial, AbstractPRNN)
│   ├── models.py         # Network logic (PRNNCell, PRNNSequence)
│   ├── layers.py         # Custom network layers (SparseNormalizedLayer)
│   ├── materials.py      # Physical constitutive models (J2Material3D)
│   ├── utils.py          # Data loaders and training routines
│   └── visualization.py  # Academic plotting tools
├── scripts/              # Executable routines for training and evaluation
├── notebooks/            # Jupyter/Colab environments for prototyping
├── data/                 # Raw structural analysis datasets (.out files)
└── trained_models/       # Saved PyTorch weights (.pth) and convergence logs


## Installation 
"""
git clone [https://github.com/your-username/prnn_project.git](https://github.com/your-username/prnn_project.git)
cd prnn_project
pip install -r requirements.txt
pip install -e .
"""


