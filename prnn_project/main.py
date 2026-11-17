import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch

from config import MaterialPhaseConfig, SimulationConfig
from prnn.constitutive import Elastic3DVectorized, J2Material3DVectorized
from prnn.material import MaterialSpec
from prnn.prnn import PRNNCell, PRNNSequence
from prnn.utils import StressStrainDataset, Trainer


def build_materials(cfg: SimulationConfig) -> MaterialSpec:
    """
    Constructs the material specification (single-phase or multi-phase) for
    PRNNCell from SimulationConfig, supporting both pure-Python models and
    Jem/Jive native MaterialFactory instances.
    """
    backend = cfg.material_backend.lower()

    # Default single-phase J2 specification when `cfg.phases` is empty
    phases = cfg.phases or [
        MaterialPhaseConfig(material_type="j2", n_points=cfg.num_material_points)
    ]

    if backend == "python":
        built_phases = []
        for phase in phases:
            mtype = phase.material_type.lower()
            if mtype == "j2":
                model = J2Material3DVectorized(
                    young_modulus=phase.young_modulus,
                    poisson_ratio=phase.poisson_ratio,
                    yield_expr=phase.yield_expr,
                    device=cfg.device,
                    dtype=cfg.dtype,
                )
            elif mtype == "elastic":
                model = Elastic3DVectorized(
                    young_modulus=phase.young_modulus,
                    poisson_ratio=phase.poisson_ratio,
                    device=cfg.device,
                    dtype=cfg.dtype,
                )
            else:
                raise ValueError(f"Unsupported Python material_type: '{phase.material_type}'.")
            built_phases.append((model, phase.n_points))
        return built_phases

    if backend == "jive":
        # Lazy import so pure-Python environments do not require compiled Jive bindings
        from _fe2_material import MaterialFactory

        built_phases = []
        for phase in phases:
            if not phase.jive_script:
                raise ValueError(
                    "When `material_backend='jive'`, each MaterialPhaseConfig must define `jive_script`."
                )
            built_phases.append((MaterialFactory(phase.jive_script), phase.n_points))
        return built_phases

    raise ValueError(f"Unknown material_backend '{cfg.material_backend}'. Expected 'python' or 'jive'.")


def run_experiment(cfg: SimulationConfig) -> None:
    """Orquestra a inicialização, treino, cronometragem e gravação de um ensaio."""
    # 1 thread per worker prevents CPU oversubscription across ProcessPoolExecutor workers
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.set_default_dtype(cfg.dtype)

    print(
        f"[PID {os.getpid()}] Iniciando: {cfg.num_material_points} pts | "
        f"Enc: {cfg.encoder_type} | Dec: {cfg.decoder_type} | Seed: {cfg.random_seed}"
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
            generator=generator,
        )
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val],
            generator=generator,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size_train, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size_val, shuffle=False
    )

    # 2. Construção do modelo PRNN (Multi-phase MaterialLayer + Encoder/Decoder Factories)
    materials = build_materials(cfg)
    cell = PRNNCell(
        dim=cfg.dim,
        input_size=len(cfg.features),
        output_size=len(cfg.targets),
        num_material_points=cfg.num_material_points,
        tensor_components=cfg.tensor_components,
        materials=materials,
        encoder_type=cfg.encoder_type,
        decoder_type=cfg.decoder_type,
        normalize_weights=cfg.normalize_weights,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    model = PRNNSequence(cell=cell).to(device=cfg.device, dtype=cfg.dtype)

    # 3. Otimizador
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

    # Salva pesos + 'config' auto-descritivo (.pth compatível com Jive)
    trainer.save(cfg.weight_path)

    # Métricas de execução
    epochs_run = len(trainer.train_losses)
    sec_per_epoch = total_time_sec / epochs_run if epochs_run > 0 else 0.0

    # 5. Salva o Histórico (.csv)
    os.makedirs(os.path.dirname(cfg.csv_path), exist_ok=True)
    history_data = np.column_stack((trainer.train_losses, trainer.val_losses))

    header = (
        f"Config - N_pts: {cfg.num_material_points} | Encoder: {cfg.encoder_type} | "
        f"Decoder: {cfg.decoder_type} | NormWeights: {cfg.normalize_weights} | "
        f"Backend: {cfg.material_backend} | Seed: {cfg.random_seed}\n"
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
        f" | {sec_per_epoch * 1000:.1f} ms/época | {epochs_run} épocas)"
    )
    print(f"       CSV  : {cfg.csv_path}")


def main() -> None:
    """Função principal que orquestra a comparação paramétrica."""
    output_dir = "benchmark_points"
    os.makedirs(output_dir, exist_ok=True)

    pontos_para_testar = [x + 1 for x in range(10)]
    seeds = [1, 7, 12, 24, 30, 42, 51, 68, 77, 100]

    # Alterne facilmente entre as topologias:
    # encoder, decoder, norm_w = "linear", "sparse", True
    # encoder, decoder, norm_w = "linear", "point", False
    # encoder, decoder, norm_w = "linear", "dense", False
    encoder, decoder, norm_w = "consistent", "consistent", True

    max_workers = len(pontos_para_testar)
    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"A DISPARAR LOTE EM PARALELO PARA A SEED: {seed} ({max_workers} PROCESSOS SIMULTÂNEOS)")
        print(f"{'=' * 70}")

        configs = []
        for n_pts in pontos_para_testar:
            cfg = SimulationConfig(
                data_path="data/GP_loading.out",
                weight_path=os.path.join(output_dir, f"GP-prnn_pts{n_pts}_{decoder}_seed{seed}.pth"),
                csv_path=os.path.join(output_dir, f"GP-history_pts{n_pts}_{decoder}_seed{seed}.csv"),
                dim=3,
                num_material_points=n_pts,
                encoder_type=encoder,
                decoder_type=decoder,
                normalize_weights=norm_w,
                material_backend="python",  # ou "jive"
                batch_size_train=10,
                batch_size_val=25,
                learning_rate=1e-3,
                epochs=10000,
                patience=1000,
                random_seed=seed,
                device=torch.device("cpu"),
            )
            configs.append(cfg)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(run_experiment, configs))

        print(f"\n[OK] Todas as configurações da seed {seed} foram finalizadas com sucesso.")


if __name__ == "__main__":
    main()