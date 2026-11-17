"""
prnn/__init__.py

Public API exports for the Physically Recurrent Neural Network (PRNN) package.
Visualization utilities (`prnn.visualize`) are intentionally excluded from top-level
eager imports to keep embedded Jem/Jive execution free of GUI dependencies.
"""

from .checkpoint import load_model
from .constitutive import (
    CompositeMaterial,
    Elastic3DVectorized,
    J2Material3DVectorized,
    PythonMaterialFactory,
    compile_math_expression,
)
from .interfaces import ConstitutiveModel, Decoder, Encoder, PRNN
from .layers import (
    ConsistentDecoder,
    ConsistentEncoder,
    DecoderFactory,
    DenseLayer,
    EncoderFactory,
    LinearEncoder,
    PointLayer,
    SparseLayer,
)
from .material import MaterialLayer, MaterialLayerContext
from .prnn import PRNNCell, PRNNSequence
from .utils import (
    Normalizer,
    RelativeError,
    RobustRelativeError,
    StressStrainDataset,
    Trainer,
    inspect_human_readable_weights,
    train_and_save_model,
)

__all__ = [
    # Core PRNN architecture
    "PRNN",
    "PRNNCell",
    "PRNNSequence",
    "load_model",
    # Localization & Homogenization layers
    "Encoder",
    "Decoder",
    "LinearEncoder",
    "ConsistentEncoder",
    "DenseLayer",
    "SparseLayer",
    "PointLayer",
    "ConsistentDecoder",
    "EncoderFactory",
    "DecoderFactory",
    # Material layers & pure-Python constitutive models
    "ConstitutiveModel",
    "MaterialLayer",
    "MaterialLayerContext",
    "Elastic3DVectorized",
    "J2Material3DVectorized",
    "CompositeMaterial",
    "PythonMaterialFactory",
    "compile_math_expression",
    # Training & evaluation utilities
    "Normalizer",
    "StressStrainDataset",
    "RelativeError",
    "RobustRelativeError",
    "Trainer",
    "train_and_save_model",
    "inspect_human_readable_weights",
]