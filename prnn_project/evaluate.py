import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from prnn.materials import J2Material3DVectorized
from prnn.models import PRNNCell, PRNNSequence
from prnn.utils import StressStrainDataset, Trainer, inspect_human_readable_weights
from prnn.visualization import plot_publication_prnn


def main():
    num_material_points = 5
    input_file_name = "data/monotonic_loading.out"
    weight_file_name = f"benchmark_points/prnn_pts{num_material_points}_soft.pth"
    decoder_type = "soft"
    encoder_type = "linear"
    curvas_para_plotar = [2, 20, 45]  # Curvas garantidas dentro dos limites usuais

    dtype = torch.float64
    device = torch.device("cpu")  # CPU para avaliação rápida sem overhead
    torch.set_default_dtype(dtype)

    # 1. Dataset com separação correta de validação
    dataset = StressStrainDataset(
        filename=input_file_name,
        features=list(range(6)),
        targets=list(range(6, 12)),
        seq_length=61,
        normalize_features=False,
        dtype=dtype
    )
    
    # Validação estrita (curvas 30 em diante)
    val_dataset = Subset(dataset, range(30, len(dataset)))
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # 2. Construção do Modelo
    material = J2Material3DVectorized(device=device, dtype=dtype)
    cell = PRNNCell(
        num_material_points=num_material_points,
        material_instance=material,
        dim=3,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        dtype=dtype
    )
    model = PRNNSequence(cell=cell).to(device=device, dtype=dtype)

    # 3. Carregamento e Avaliação
    trainer = Trainer(model)
    trainer.load(weight_file_name)
    trainer.eval(val_loader, verbose=True)

    # 4. Leitura Humana dos Pesos Treinados
    inspect_human_readable_weights(cell, num_material_points)

    # 5. Plotagem com Criação Segura do Diretório
    output_img_dir = "plots_artigo"
    os.makedirs(output_img_dir, exist_ok=True)
    model.eval()

    for curve_id in curvas_para_plotar:
        if curve_id >= len(dataset):
            print(f"[Aviso] Curva {curve_id} fora dos limites do dataset ({len(dataset)} amostras). Ignorando.")
            continue

        strain_tensor, true_stress_tensor = dataset[curve_id]
        strain_batch = strain_tensor.unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            pred_stress_tensor = model(strain_batch)

        hf_strain = strain_tensor.cpu().numpy()
        hf_stress = true_stress_tensor.cpu().numpy()
        nn_strain = hf_strain
        nn_stress = pred_stress_tensor.squeeze(0).cpu().numpy()

        save_file = os.path.join(output_img_dir, f"curva_{curve_id}_painel.png")
        plot_publication_prnn(
            hf_strain=hf_strain,
            hf_stress=hf_stress,
            nn_strain=nn_strain,
            nn_stress=nn_stress,
            curve_id=curve_id,
            custom_labels=["Micro FE (Referência)", f"PRNN ({num_material_points} pts)"],
            save_path=save_file,
            contexto='artigo',
            orientacao='horizontal'
        )
        print(f"Curva {curve_id} salva em: {save_file}")


if __name__ == "__main__":
    main()