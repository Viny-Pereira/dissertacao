"""
prnn/visualize.py

Scientific visualization and publication-ready figure generation module for PRNNs.
Follows the Voigt tensor convention: [11, 22, 33, 12, 23, 13].
"""

import os
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np


def _apply_publication_style(contexto: str = "artigo") -> dict:
    """Applies STIX academic typography settings and returns font sizes."""
    if contexto == "apresentacao":
        tamanhos = {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    else:  # 'artigo' / dissertação
        tamanhos = {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        }

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            **tamanhos,
        }
    )
    return tamanhos


def plot_publication_prnn(
    hf_strain: np.ndarray,
    hf_stress: np.ndarray,
    nn_strain: np.ndarray,
    nn_stress: np.ndarray,
    curve_id: Optional[int] = None,
    custom_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    contexto: str = "artigo",
    orientacao: str = "horizontal",
    show: bool = True,
) -> None:
    """
    Generates a 6-panel stress-strain comparison figure following the Voigt order
    [11, 22, 33, 12, 23, 13] (xx, yy, zz, xy, yz, xz).

    Args:
        hf_strain: Reference strain trajectory [seq_len, 6].
        hf_stress: Reference stress trajectory [seq_len, 6].
        nn_strain: PRNN input strain trajectory [seq_len, 6].
        nn_stress: PRNN predicted stress trajectory [seq_len, 6].
        curve_id: Optional trajectory identifier for title/metadata.
        custom_labels: Legend labels `[ref_label, pred_label]`.
        save_path: Optional file path to export high-resolution figure (300 DPI).
        contexto: Typography preset ('artigo' or 'apresentacao').
        orientacao: Subplot grid layout ('horizontal' -> 2x3, 'vertical' -> 3x2).
        show: Whether to display the interactive figure window. Defaults to True.
    """
    tamanhos = _apply_publication_style(contexto)

    if orientacao == "horizontal":
        fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    else:
        fig, axs = plt.subplots(3, 2, figsize=(8.5, 9.5))

    axs_flat = axs.flatten()

    # Voigt convention: [xx, yy, zz, xy, yz, xz] -> [11, 22, 33, 12, 23, 13]
    xlabels = [
        r"$\varepsilon_{11}$ [-]",
        r"$\varepsilon_{22}$ [-]",
        r"$\varepsilon_{33}$ [-]",
        r"$\gamma_{12}$ [-]",
        r"$\gamma_{23}$ [-]",
        r"$\gamma_{13}$ [-]",
    ]
    ylabels = [
        r"$\sigma_{11}$ [MPa]",
        r"$\sigma_{22}$ [MPa]",
        r"$\sigma_{33}$ [MPa]",
        r"$\tau_{12}$ [MPa]",
        r"$\tau_{23}$ [MPa]",
        r"$\tau_{13}$ [MPa]",
    ]

    if custom_labels is None:
        custom_labels = ["Referência (Micro FE)", "PRNN"]

    color_ref = "#4A4A4A"
    color_pred = "#D95F02"

    for j in range(6):
        ax = axs_flat[j]

        ax.plot(
            hf_strain[:, j],
            hf_stress[:, j],
            color=color_ref,
            linewidth=2.0,
            linestyle="--",
            label=custom_labels[0],
            zorder=1,
            alpha=0.85,
        )

        ax.plot(
            nn_strain[:, j],
            nn_stress[:, j],
            color=color_pred,
            linewidth=1.6,
            linestyle="-",
            label=custom_labels[1],
            zorder=2,
        )

        ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))
        ax.xaxis.get_offset_text().set_fontsize(tamanhos["xtick.labelsize"])

        ax.set_xlabel(xlabels[j])
        ax.set_ylabel(ylabels[j])

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", length=5, direction="in")
        ax.tick_params(which="minor", length=3, direction="in")

        ax.grid(True, which="major", linestyle=":", alpha=0.6)

        min_x, max_x = ax.get_xlim()
        if abs(max_x - min_x) < 1e-6:
            ax.set_xlim(min_x - 1e-3, max_x + 1e-3)

    if curve_id is not None:
        fig.suptitle(f"Trajetória {curve_id}", y=1.01, fontsize=tamanhos["axes.labelsize"])

    fig.tight_layout()
    fig.subplots_adjust(top=0.90 if orientacao == "horizontal" else 0.93)

    handles, labels = axs_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_convergence_from_csv(
    csv_path: str,
    ignore_initial: int = 100,
    save_path: Optional[str] = None,
    contexto: str = "artigo",
    show: bool = True,
) -> None:
    """
    Reads convergence history from a CSV file and plots the full training/validation
    loss history alongside a zoomed-in view omitting the initial noisy epochs.
    """
    _apply_publication_style(contexto)

    data = np.loadtxt(csv_path, delimiter=",", comments="#")
    train_losses = data[:, 0]
    val_losses = data[:, 1]
    total_epochs = len(train_losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(train_losses, label="Treino", color="#1f77b4", linewidth=1.5)
    ax1.plot(val_losses, label="Validação", color="#D95F02", linewidth=1.5)
    ax1.set_yscale("log")
    ax1.set_title("Convergência (Histórico Completo)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel(r"MSE [MPa$^2$]")
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    if total_epochs > ignore_initial:
        zoom_epochs = range(ignore_initial, total_epochs)
        ax2.plot(zoom_epochs, train_losses[ignore_initial:], label="Treino", color="#1f77b4", linewidth=1.5)
        ax2.plot(zoom_epochs, val_losses[ignore_initial:], label="Validação", color="#D95F02", linewidth=1.5)
        ax2.set_yscale("log")
        ax2.set_title(f"Convergência (A partir da Época {ignore_initial})")
        ax2.set_xlabel("Época")
        ax2.set_ylabel(r"MSE [MPa$^2$]")
        ax2.legend(frameon=True, facecolor="white", edgecolor="none")
        ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "Treino curto para recorte", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)