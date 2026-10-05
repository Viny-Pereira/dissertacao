from dataclasses import dataclass, field
import torch
from typing import List

@dataclass
class SimulationConfig:
    # 1. Configurações de Dados e I/O
    data_path: str
    weight_path: str
    csv_path: str
    features: List[int] = field(default_factory=lambda: list(range(0, 6)))
    targets: List[int] = field(default_factory=lambda: list(range(6, 12)))
    
    # 2. Arquitetura da PRNN
    num_material_points: int = 5
    tensor_components: int = 6
    decoder_type: str = 'soft'
    normalize_features: bool = False
    
    # 3. Hiperparâmetros de Treinamento
    sequence_length: int = 61
    batch_size_train: int = 3
    batch_size_val: int = 10
    epochs: int = 100000
    learning_rate: float = 1e-3
    patience: int = 1000
    random_seed: int = 42
    
    # 4. Precisão e Hardware
    dtype: torch.dtype = torch.float64
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))