"""
Centralized dataclass configurations for PRNN architecture, multi-phase constitutive
material definitions, training hyperparameters, and Jem/Jive runtime constraints.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import torch


@dataclass
class MaterialPhaseConfig:
    """
    Configuration for a single constitutive phase inside the PRNN `MaterialLayer`.

    Attributes:
        material_type: Constitutive law identifier ('j2', 'elastic', or 'jive_native').
        n_points: Number of fictitious material points allocated to this phase.
        young_modulus: Young's modulus E (in MPa).
        poisson_ratio: Poisson's ratio nu.
        yield_expr: Hardening/yield expression string in terms of equivalent plastic
            strain `x` (used when `material_type == 'j2'`).
        jive_script: Optional raw Jem/Jive material block string (used when
            `material_type == 'jive_native'`).
    """

    material_type: str = "j2"
    n_points: int = 5
    young_modulus: float = 79500.0
    poisson_ratio: float = 0.33
    yield_expr: str = "90.0740 * (1. - exp(-113.0268 * x)) + 444.1184"
    jive_script: Optional[str] = None


@dataclass
class SimulationConfig:
    """
    Master configuration container for PRNN dataset loading, network topology,
    constitutive material phases, and training hyperparameters.
    """

    # 1. Configurações de Dados e I/O
    data_path: str
    weight_path: str
    csv_path: str
    features: List[int] = field(default_factory=lambda: list(range(0, 6)))
    targets: List[int] = field(default_factory=lambda: list(range(6, 12)))

    # 2. Arquitetura da PRNN
    dim: Optional[int] = 3
    num_material_points: int = 5
    tensor_components: int = 6
    encoder_type: str = "linear"          # 'linear', 'consistent' ('tc')
    decoder_type: str = "dense"           # 'dense', 'sparse', 'point', 'consistent' ('tc')
    normalize_weights: Optional[bool] = None  # Partition-of-unity flag for decoder weights
    normalize_features: bool = False      # Min-Max [-1, 1] scaling for input strains

    # 3. Camada Material (Python puro ou ponte C++ Jem/Jive)
    material_backend: str = "jive"        # 'jive' (CPU C++ bridge) or 'python' (pure PyTorch)
    phases: List[MaterialPhaseConfig] = field(default_factory=list)

    # 4. Hiperparâmetros de Treinamento
    sequence_length: int = 61
    batch_size_train: int = 3
    batch_size_val: int = 10
    epochs: int = 100000
    learning_rate: float = 1e-3
    patience: int = 1000
    random_seed: int = 42

    # 5. Precisão e Hardware
    dtype: torch.dtype = torch.float64
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        # Resolve default device safely: Jive C++ bridge strictly requires CPU
        if self.device is None:
            if self.material_backend.lower() == "jive":
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.material_backend.lower() == "jive" and self.device.type != "cpu":
            raise ValueError(
                "The Jem/Jive native material bridge requires `device=torch.device('cpu')`."
            )

        # Keep num_material_points synchronized if explicit phases were supplied
        if self.phases:
            total_phase_pts = sum(p.n_points for p in self.phases)
            if total_phase_pts != self.num_material_points:
                self.num_material_points = total_phase_pts

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the configuration into a dictionary with JSON/pickle-safe primitives."""
        raw = asdict(self)
        raw["dtype"] = str(self.dtype)
        raw["device"] = str(self.device)
        return raw