import os
import torch
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
import time

from config import SimulationConfig
from prnn.materials import J2Material3DVectorized
from prnn.models import PRNNCell, PRNNSequence
from prnn.utils import StressStrainDataset, Trainer


def run_experiment(cfg: SimulationConfig) -> None:
    """Orquestra a inicialização, treino, cronometragem e gravação de um ensaio."""
    num_threads_per_worker = 4
    torch.set_num_threads(num_threads_per_worker)
    torch.set_num_interop_threads(1)

    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.set_default_dtype(cfg.dtype)

    print(
        f"[PID {os.getpid()}] Iniciando: {cfg.num_material_points} pts | Seed: {cfg.random_seed}"
    )

    # 1. Carregamento de dados com Split fixo
    full_dataset = StressStrainDataset(
        filename=cfg.data_path,
        features=cfg.features,
        targets=cfg.targets,
        seq_length=cfg.sequence_length,
        normalize_features=cfg.normalize_features,
        dtype=cfg.dtype,
    )

    total_trajectories = len(full_dataset)
    n_train = 100
    n_val = 150

    if total_trajectories < (n_train + n_val):
        raise ValueError(
            f"Dataset insuficiente: o arquivo possui {total_trajectories} curvas, "
            f"mas são necessárias pelo menos {n_train + n_val} (100 treino + 150 teste)."
        )

    generator = torch.Generator().manual_seed(cfg.random_seed)
    n_unused = total_trajectories - (n_train + n_val)

    if n_unused > 0:
        train_dataset, val_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val, n_unused],
            generator=generator
        )
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val],
            generator=generator
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size_train, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size_val, shuffle=False
    )

    # 2. Construção do modelo PRNN (Agora recebe o encoder_type explicitamente)
    material = J2Material3DVectorized(device=cfg.device, dtype=cfg.dtype)
    cell = PRNNCell(
        input_size=len(cfg.features),
        output_size=len(cfg.targets),
        num_material_points=cfg.num_material_points,
        tensor_components=cfg.tensor_components,
        material_instance=material,
        encoder_type=getattr(cfg, 'encoder_type', 'linear'),  # <-- AJUSTE AQUI
        decoder_type=cfg.decoder_type,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    model = PRNNSequence(cell=cell).to(device=cfg.device, dtype=cfg.dtype)

    # 3. Otimizador sem scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 4. Treino com Cronometragem
    trainer = Trainer(model, optimizer=optimizer, scheduler=None)

    t_start = time.perf_counter()
    trainer.train(
        train_loader,
        val_loader,
        epochs=cfg.epochs,
        patience=cfg.patience,
        verbose=False,
    )
    total_time_sec = time.perf_counter() - t_start

    # Salva pesos (.pth)
    trainer.save(cfg.weight_path)

    # Métricas de execução
    epochs_run = len(trainer.train_losses)
    sec_per_epoch = total_time_sec / epochs_run if epochs_run > 0 else 0.0

    # 5. Salva o Histórico (.csv)
    os.makedirs(os.path.dirname(cfg.csv_path), exist_ok=True)
    history_data = np.column_stack((trainer.train_losses, trainer.val_losses))

    header = (
        f"Config - N_pts: {cfg.num_material_points} | Decoder: {cfg.decoder_type} | Seed: {cfg.random_seed}\n"
        f"Total_Trajectories_File: {total_trajectories} | Train_Trajectories: {n_train} | Val_Trajectories: {n_val}\n"
        f"Total_Time_sec: {total_time_sec:.2f}\n"
        f"Epochs_run: {epochs_run}\n"
        f"Time_per_epoch_sec: {sec_per_epoch:.5f}\n"
        f"Train_Loss_MSE,Val_Loss_MSE"
    )
    np.savetxt(
        cfg.csv_path, history_data, delimiter=",", header=header, comments="# "
    )

    mins, secs = divmod(total_time_sec, 60)
    print(
        f"[PID {os.getpid()}] Concluído: {cfg.num_material_points} pts | Seed: {cfg.random_seed}"
    )
    print(
        f"       Tempo: {int(mins)}m {secs:.1f}s ({total_time_sec:.2f} s total"
        f" | {sec_per_epoch*1000:.1f} ms/época | {epochs_run} épocas)"
    )
    print(f"       CSV  : {cfg.csv_path}")


def main():
    """Função principal que orquestra a comparação paramétrica."""
    output_dir = "benchmark_points"
    os.makedirs(output_dir, exist_ok=True)

    pontos_para_testar = [x + 1 for x in range(10)]
    seeds = [1, 7, 12, 24, 30, 42, 51, 68, 77, 100]
    
    
    encoder = "linear"
    decoder = "point"
   
    """ 
    encoder = "consistent"
    decoder = "consistent"
    """
    
    max_workers = len(pontos_para_testar)
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"A DISPARAR LOTE EM PARALELO PARA A SEED: {seed} ({max_workers} PROCESSOS SIMULTÂNEOS)")
        print(f"{'='*70}")

        configs = []
        for n_pts in pontos_para_testar:
            cfg = SimulationConfig(
                data_path="data/GP_loading.out",
                weight_path=os.path.join(output_dir, f"GP-prnn_pts{n_pts}_{decoder}_seed{seed}.pth"),
                csv_path=os.path.join(output_dir, f"GP-history_pts{n_pts}_{decoder}_seed{seed}.csv"),
                num_material_points=n_pts,
                encoder_type=encoder, 
                decoder_type=decoder,
                batch_size_train=10,
                batch_size_val=25,
                learning_rate=1e-3,
                epochs=10000,
                patience=500,
                random_seed=seed,
                device=torch.device("cpu"),
            )
            configs.append(cfg)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(run_experiment, configs))

        print(f"\n[OK] Todas as configurações da seed {seed} foram finalizadas com sucesso.")


if __name__ == "__main__":
    main()