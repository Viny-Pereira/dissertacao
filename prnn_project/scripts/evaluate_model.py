import os
import sys
import copy
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Garante acesso ao pacote prnn adicionando a raiz do projeto ao sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prnn.materials import J2Material3DVectorized
from prnn.models import PRNNCell, PRNNSequence
from prnn.utils import (
    StressStrainDataset,
    Trainer,
    RelativeError,
    RobustRelativeError
)
from prnn.visualization import plot_publication_prnn
class Evaluator:
    """
    Classe dedicada à avaliação de modelos PRNN treinados.
    Executa inferências sem gradiente, computa métricas físicas e gera gráficos científicos.
    """
    def __init__(
        self,
        weight_path: str,
        num_material_points: int = 1,
        decoder_type: str = "soft",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        num_threads: int = 4
    ):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.weight_path = weight_path
        self.num_material_points = num_material_points
        self.decoder_type = decoder_type

        if self.device.type == "cpu":
            torch.set_num_threads(num_threads)

        torch.set_default_dtype(self.dtype)

        # 1. Instancia o Modelo Modular
        material = J2Material3DVectorized(device=self.device, dtype=self.dtype)
        cell = PRNNCell(
            num_material_points=self.num_material_points,
            material_instance=material,
            dim=3,
            decoder_type=self.decoder_type,
            dtype=self.dtype
        )
        self.model = PRNNSequence(cell=cell).to(device=self.device, dtype=self.dtype)

        # 2. CARREGAR OS PESOS PRIMEIRO!
        self._load_weights()

        # 3. Agora instancia o Trainer com o modelo já calibrado com os pesos certos
        self.trainer = Trainer(self.model)
        self.trainer._best_state_dict = copy.deepcopy(self.model.state_dict())
        self.model.eval()

    def _load_weights(self) -> None:
        """Carrega os pesos compatibilizando modelos novos e legados com validação estrita."""
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Arquivo de pesos não encontrado: {self.weight_path}")

        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("best_state_dict", checkpoint.get("model_state_dict", checkpoint))

        model_keys = list(self.model.state_dict().keys())

        if "fc1.weight" in state_dict:
            print("[INFO] Detectado formato legado (fc1/fc2). Mapeando para o modelo modular...")
            
            enc_key = "cell.encoder.linear.weight" if "cell.encoder.linear.weight" in model_keys else "cell.encoder.weight"
            dec_key = "cell.decoder.weight"
            
            mapped_state = {
                enc_key: state_dict["fc1.weight"],
                dec_key: state_dict["fc2.weight"]
            }
            
            dec_bias_key = "cell.decoder.bias"
            if dec_bias_key in model_keys:
                if "fc2.bias" in state_dict and state_dict["fc2.bias"] is not None:
                    mapped_state[dec_bias_key] = state_dict["fc2.bias"]
                else:
                    mapped_state[dec_bias_key] = torch.zeros_like(self.model.state_dict()[dec_bias_key])

            # Carrega no modelo
            self.model.load_state_dict(mapped_state, strict=False)
            
            # Verificação de Sanidade: confere se os valores do encoder batem com o checkpoint
            diff = torch.max(torch.abs(self.model.cell.encoder.weight - state_dict["fc1.weight"])).item()
            if diff > 1e-9:
                raise RuntimeError("ERRO CRÍTICO: Os pesos do encoder continuam diferentes do checkpoint!")
            print(f"[OK] Validação de integridade aprovada (Discrepância dos pesos: {diff:.2e})")

        else:
            self.model.load_state_dict(state_dict)

        print(f"[OK] Pesos carregados e validados com sucesso a partir de: {self.weight_path}\n")

    def evaluate_metrics(
        self, 
        data_loaders: Dict[str, DataLoader], 
        criterion: Optional[nn.Module] = None,
        verbose_batches: bool = True
    ) -> Dict[str, float]:
        """
        Avalia os conjuntos de dados reproduzindo a saída por lote do código original.
        """
        # Se nenhum critério for especificado, utiliza MSELoss como no notebook
        eval_criterion = criterion if criterion is not None else nn.MSELoss()
        results = {}

        print("\n" + "=" * 70)
        print("INÍCIO DA AVALIAÇÃO DE TESTE / VALIDAÇÃO")
        print("=" * 70)

        with torch.no_grad():
            for dataset_name, loader in data_loaders.items():
                print(f"\n--- Avaliando: {dataset_name} ({len(loader)} lotes) ---")
                
                combined_loss = 0.0
                total_batches = len(loader)

                for j, (x, t) in enumerate(loader):
                    x = x.to(self.device, non_blocking=True)
                    t = t.to(self.device, non_blocking=True)
                    
                    y = self.model(x)
                    loss = eval_criterion(y, t)
                    loss_val = loss.item()
                    combined_loss += loss_val

                    # Impressão exatamente idêntica ao notebook original
                    if verbose_batches:
                        print(f"Loss for test batch  {j + 1} / {total_batches} : {loss_val}")

                avg_loss = combined_loss / total_batches
                results[dataset_name] = avg_loss

                print(f"Aggregated test set loss: {avg_loss}")
                print(f"{avg_loss}\n")

        print("=" * 70)
        return results
    
    def plot_curves(
        self,
        dataset: StressStrainDataset,
        curve_ids: List[int],
        output_dir: str = "plots_eval",
        dataset_label: str = "Micro FE"
    ) -> None:
        """
        Extrai as trajetórias selecionadas e salva o painel de 6 componentes.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGerando gráficos para as curvas {curve_ids}...")

        with torch.no_grad():
            for c_id in curve_ids:
                if c_id >= len(dataset):
                    print(f"Aviso: Curva {c_id} está fora dos limites do dataset.")
                    continue

                strain_tensor, true_stress_tensor = dataset[c_id]
                strain_batch = strain_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

                pred_stress_tensor = self.model(strain_batch)

                hf_strain = strain_tensor.cpu().numpy()
                hf_stress = true_stress_tensor.cpu().numpy()
                nn_stress = pred_stress_tensor.squeeze(0).cpu().numpy()

                save_path = os.path.join(output_dir, f"curva_{c_id}_painel_6comp.png")
                plot_publication_prnn(
                    hf_strain=hf_strain,
                    hf_stress=hf_stress,
                    nn_strain=hf_strain,
                    nn_stress=nn_stress,
                    curve_id=c_id,
                    custom_labels=[dataset_label, f"PRNN ({self.num_material_points} pts)"],
                    save_path=save_path,
                    contexto="artigo",
                    orientacao="horizontal"
                )
                print(f" -> Salvo: {save_path}")

def main():
    # -------------------------------------------------------------------------
    # CONFIGURAÇÕES DA AVALIAÇÃO
    # -------------------------------------------------------------------------
    # Caminho do modelo a testar
    #weight_file = "prnn/benchmark_comparison/prnn_composite_loading_1_dense.pth"
    weight_file = "dissertacao/prnn_project/benchmark_points/prnn_pts1_soft.pth"
    
    #weight_file = "prnn/benchmark_comparison/prnn_modular_1pt_soft.pth"

    num_points = 1
    decoder_type = "soft"
    
    device = torch.device("cpu")
    dtype = torch.float64

    # 1. Instancia o Avaliador
    evaluator = Evaluator(
        weight_path=weight_file,
        num_material_points=num_points,
        decoder_type=decoder_type,
        device=device,
        dtype=dtype,
        num_threads=4
    )

    # 2. Definição do Dataset de Carregamento Monotônico
    features = list(range(6))
    targets = list(range(6, 12))
    seq_len = 61

    # Carrega o arquivo de carregamento monotônico
    monotonic_ds = StressStrainDataset("data/monotonic_loading.out", features, targets, seq_len, dtype=dtype)
    
    # Separa a validação exatamente como no treino (curvas 30 em diante)
    val_monotonic = Subset(monotonic_ds, range(30, len(monotonic_ds)))

    test_loaders = {
        "Monotonic Loading (Validação)": DataLoader(val_monotonic, batch_size=10, shuffle=False)
    }

    # 3. Execução das Métricas com L1Loss (Erro Absoluto Médio em MPa)
    evaluator.evaluate_metrics(test_loaders, criterion=nn.MSELoss(), verbose_batches=True)

    # 4. Geração dos Gráficos das Curvas no Próprio Monotonic Loading
    # Exemplo: Curva 30 e Curva 50 (que pertencem ao conjunto de validação)
    evaluator.plot_curves(
        dataset=monotonic_ds,
        curve_ids=[30, 50],
        output_dir="plots_eval/monotonic",
        dataset_label="Micro FE (Monotonic)"
    )


if __name__ == "__main__":
    main()