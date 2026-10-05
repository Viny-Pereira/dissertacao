import os
import torch
import numpy as np
import random

from config import SimulationConfig
from prnn.materials import J2Material3DVectorized
from prnn.models import PRNNCell, PRNNSequence
from prnn.utils import StressStrainDataset, Trainer


def run_experiment(cfg: SimulationConfig) -> None:
    """Orquestra a inicialização, treino e gravação de um único ensaio."""
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.set_default_dtype(cfg.dtype)

    print(f"A executar: {cfg.num_material_points} pontos | Decoder: {cfg.decoder_type}")

    # 1. Carregamento de dados
    full_dataset = StressStrainDataset(
        filename=cfg.data_path,
        features=cfg.features,
        targets=cfg.targets,
        seq_length=cfg.sequence_length,
        normalize_features=cfg.normalize_features,
        dtype=cfg.dtype
    )

    train_dataset = torch.utils.data.Subset(full_dataset, range(30))
    val_dataset = torch.utils.data.Subset(full_dataset, range(30, len(full_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size_val, shuffle=False)

    # 2. Construção do modelo PRNN
    material = J2Material3DVectorized(device=cfg.device, dtype=cfg.dtype)
    cell = PRNNCell(
        input_size=len(cfg.features),
        output_size=len(cfg.targets),
        num_material_points=cfg.num_material_points,
        tensor_components=cfg.tensor_components,
        material_instance=material,
        decoder_type=cfg.decoder_type,
        dtype=cfg.dtype
    )
    model = PRNNSequence(cell=cell).to(device=cfg.device, dtype=cfg.dtype)

    # 3. Otimizador e Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=200, min_lr=1e-7
    )

    # 4. Treino
    trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler)
    trainer.train(train_loader, val_loader, epochs=cfg.epochs, patience=cfg.patience)
    trainer.save(cfg.weight_path)


def main():
    """Função principal que orquestra a comparação paramétrica."""
    output_dir = "benchmark_points"
    os.makedirs(output_dir, exist_ok=True)

    pontos_para_testar = [1, 5]
    decoder = "soft"

    for n_pts in pontos_para_testar:
        cfg = SimulationConfig(
            data_path="data/monotonic_loading.out",
            weight_path=os.path.join(output_dir, f"prnn_pts{n_pts}_{decoder}.pth"),
            csv_path=os.path.join(output_dir, f"history_pts{n_pts}_{decoder}.csv"),
            num_material_points=n_pts,
            decoder_type=decoder,
            batch_size_train=3,    
            batch_size_val=10,    
            learning_rate=1e-3,
            epochs=10000,
            patience=500,          
            random_seed=42,
            device=torch.device("cpu"),
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()