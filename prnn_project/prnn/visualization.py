"""
prnn/visualize.py
Módulo de visualização científica e geração de figuras para publicação.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from typing import Optional, List


def plot_publication_prnn(
    hf_strain: np.ndarray,
    hf_stress: np.ndarray,
    nn_strain: np.ndarray,
    nn_stress: np.ndarray,
    curve_id: Optional[int] = None,
    custom_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    contexto: str = 'artigo',
    orientacao: str = 'horizontal'
) -> None:
    """
    Gera um painel comparando as 6 componentes do tensor de Voigt (11, 22, 33, 12, 13, 23).
    """
    if contexto == 'apresentacao':
        tamanhos = {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14
        }
    else:  # 'artigo' / dissertação
        tamanhos = {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11
        }

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        **tamanhos
    })
    plt.rcParams['text.usetex'] = False

    if orientacao == 'horizontal':
        fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    else:
        fig, axs = plt.subplots(3, 2, figsize=(8.5, 9.5))

    axs_flat = axs.flatten()

    xlabels = [
        r'$\varepsilon_{11}$ [-]', r'$\varepsilon_{22}$ [-]', r'$\varepsilon_{33}$ [-]',
        r'$\gamma_{12}$ [-]', r'$\gamma_{13}$ [-]', r'$\gamma_{23}$ [-]'
    ]
    ylabels = [
        r'$\sigma_{11}$ [MPa]', r'$\sigma_{22}$ [MPa]', r'$\sigma_{33}$ [MPa]',
        r'$\tau_{12}$ [MPa]', r'$\tau_{13}$ [MPa]', r'$\tau_{23}$ [MPa]'
    ]

    if custom_labels is None:
        custom_labels = ["Referência (Micro FE)", "PRNN"]

    color_ref = '#4A4A4A'
    color_pred = '#D95F02'

    for j in range(6):
        ax = axs_flat[j]

        ax.plot(
            hf_strain[:, j], hf_stress[:, j],
            color=color_ref, linewidth=2.0, linestyle="--",
            label=custom_labels[0], zorder=1, alpha=0.85
        )

        ax.plot(
            nn_strain[:, j], nn_stress[:, j],
            color=color_pred, linewidth=1.6, linestyle="-",
            label=custom_labels[1], zorder=2
        )

        ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
        ax.xaxis.get_offset_text().set_fontsize(tamanhos["xtick.labelsize"])

        ax.set_xlabel(xlabels[j])
        ax.set_ylabel(ylabels[j])

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', length=5, direction='in')
        ax.tick_params(which='minor', length=3, direction='in')

        ax.grid(True, which='major', linestyle=':', alpha=0.6)

        min_x, max_x = ax.get_xlim()
        if abs(max_x - min_x) < 1e-6:
            ax.set_xlim(min_x - 1e-3, max_x + 1e-3)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90 if orientacao == 'horizontal' else 0.93)

    handles, labels = axs_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98 if orientacao == 'horizontal' else 0.98),
        ncol=2,
        frameon=True,
        facecolor='white',
        edgecolor='none'
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)