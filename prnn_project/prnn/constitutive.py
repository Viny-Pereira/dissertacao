"""
constitutive.py

Pure-Python physical constitutive models for the PRNN architecture.
Features Jive-compatible string expression compilation, automatic autograd
hardening derivatives, implicit differentiation through radial return mapping,
vectorized algorithmic tangent computation, and full trial-commit-cancel
lifecycle support for Jem/Jive integration.
"""

import time
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor

from .interfaces import ConstitutiveModel


def compile_math_expression(
    expr: Union[str, Callable[[Tensor], Tensor]],
    var_name: str = "x",
) -> Callable[[Tensor], Tensor]:
    """
    Compiles a mathematical expression string (e.g., Jive yield curve syntax)
    into a vectorized PyTorch callable once at initialization.

    Args:
        expr: Mathematical expression string in terms of `var_name` (e.g.,
            '90.0740 * (1. - exp(-113.0268 * x)) + 444.1184') or a PyTorch callable.
        var_name: Independent variable name inside the expression. Defaults to 'x'.

    Returns:
        A vectorized callable accepting and returning a PyTorch Tensor.

    Raises:
        TypeError: If `expr` is neither a string nor a callable.
        ValueError: If `expr` is an empty string or evaluates to an invalid expression.
    """
    if callable(expr):
        return expr

    if not isinstance(expr, str):
        raise TypeError(
            "Expression must be a string (e.g. '444.1 + 90.0 * (1 - exp(-113 * x))') or a callable."
        )

    sanitized = expr.replace("^", "**").strip()
    if not sanitized:
        raise ValueError("Mathematical expression string cannot be empty.")

    safe_env = {
        "exp": torch.exp,
        "sqrt": torch.sqrt,
        "log": torch.log,
        "sin": torch.sin,
        "cos": torch.cos,
        "abs": torch.abs,
        "pow": torch.pow,
        "clamp": torch.clamp,
    }

    code_obj = compile(sanitized, "<constitutive_expression>", "eval")

    def compiled_fn(x: Tensor) -> Tensor:
        res = eval(code_obj, {"__builtins__": {}}, {**safe_env, var_name: x})
        if not isinstance(res, Tensor):
            return torch.full_like(x, float(res))
        return res

    # Sanity check at initialization to catch syntax or domain errors immediately
    with torch.no_grad():
        _ = compiled_fn(torch.zeros(1, dtype=torch.float64))

    return compiled_fn


class PythonMaterialFactory:
    """
    Adapter wrapping a pure-Python ConstitutiveModel builder so it exposes the exact
    `factory(count)` callable signature expected by MaterialLayer and PRNN.

    Attributes:
        builder: Zero-argument callable returning a fresh ConstitutiveModel instance.
        components: Number of Voigt stress/strain components (e.g., 3 or 6).
    """

    def __init__(self, builder: Callable[[], ConstitutiveModel], components: int = 6):
        self.builder = builder
        self.components = components

    def __call__(self, count: int) -> ConstitutiveModel:
        """
        Instantiates and configures a constitutive model batch for `count` material points.

        Args:
            count: Total number of fictitious integration points (batch_size * n_matpts).

        Returns:
            Configured ConstitutiveModel instance.
        """
        model = self.builder()
        model.configure(total_points=count, tensor_components=self.components)
        return model


class CompositeMaterial(ConstitutiveModel):
    """
    Composite material manager orchestrating multiple constitutive models in parallel.
    Partitions input strain tensors across distinct material phases (e.g., elastic
    inclusions + elastoplastic matrix) and recombines resulting stresses, algorithmic
    tangents, and internal history states.

    Attributes:
        materials_with_counts: List of (ConstitutiveModel, n_points_per_sample) tuples.
        materials: Registered ModuleList of sub-material instances.
        counts: Number of material points assigned to each phase per sample.
        total_points_per_sample: Total fictitious material points per RVE/sample.
    """

    def __init__(self, materials_with_counts: List[Tuple[ConstitutiveModel, int]]):
        """
        Initializes the composite material orchestrator.

        Args:
            materials_with_counts: List of tuples containing constitutive model instances
                and their assigned material point counts per sample.
                Example: [(Elastic3DVectorized(), 2), (J2Material3DVectorized(), 3)].

        Raises:
            ValueError: If the sum of material points per sample is not strictly positive.
        """
        super().__init__()
        self.materials_with_counts = materials_with_counts
        self.materials = torch.nn.ModuleList([m for m, _ in materials_with_counts])
        self.counts = [c for _, c in materials_with_counts]
        self.total_points_per_sample = sum(self.counts)
        self.batch_size = 0
        self.tensor_components = 6

        if self.total_points_per_sample <= 0:
            raise ValueError("The total number of material points per sample must be strictly positive.")

    def configure(self, total_points: int, tensor_components: int = 6) -> None:
        """
        Allocates memory and configures state variables for each sub-material phase
        based on the global batch size.

        Args:
            total_points: Flattened number of integration points
                (batch_size * total_points_per_sample).
            tensor_components: Voigt stress/strain tensor dimension. Defaults to 6.

        Raises:
            ValueError: If `total_points` is not divisible by `total_points_per_sample`.
        """
        if total_points % self.total_points_per_sample != 0:
            raise ValueError(
                f"total_points ({total_points}) must be a multiple of "
                f"points per sample ({self.total_points_per_sample})."
            )

        self.batch_size = total_points // self.total_points_per_sample
        self.tensor_components = tensor_components

        for material, count in self.materials_with_counts:
            sub_total_points = self.batch_size * count
            material.configure(total_points=sub_total_points, tensor_components=tensor_components)

    def stress(self, micro_strain: Tensor) -> Tensor:
        """
        Partitions the flattened micro-strain tensor across sub-materials, evaluates
        their individual constitutive laws, and recombines the trial stresses.

        Args:
            micro_strain: Micro-strain tensor of shape
                [batch_size * total_points_per_sample, tensor_components].

        Returns:
            Combined trial micro-stress tensor of shape
            [batch_size * total_points_per_sample, tensor_components].
        """
        strain_reshaped = micro_strain.reshape(
            self.batch_size, self.total_points_per_sample, self.tensor_components
        )

        stress_blocks = []
        start_idx = 0

        for material, count in self.materials_with_counts:
            end_idx = start_idx + count
            sub_strain = strain_reshaped[:, start_idx:end_idx, :].reshape(-1, self.tensor_components)
            sub_stress = material.stress(sub_strain)
            stress_blocks.append(sub_stress.reshape(self.batch_size, count, self.tensor_components))
            start_idx = end_idx

        return torch.cat(stress_blocks, dim=1).reshape(-1, self.tensor_components)

    def tangent(self, micro_strain: Tensor) -> Tensor:
        """
        Partitions the flattened micro-strain tensor across sub-materials and aggregates
        their individual consistent algorithmic tangent stiffness matrices.

        Args:
            micro_strain: Micro-strain tensor of shape
                [batch_size * total_points_per_sample, tensor_components].

        Returns:
            Combined algorithmic tangent stiffness tensor of shape
            [batch_size * total_points_per_sample, tensor_components, tensor_components].
        """
        strain_reshaped = micro_strain.reshape(
            self.batch_size, self.total_points_per_sample, self.tensor_components
        )

        tangent_blocks = []
        start_idx = 0

        for material, count in self.materials_with_counts:
            end_idx = start_idx + count
            sub_strain = strain_reshaped[:, start_idx:end_idx, :].reshape(-1, self.tensor_components)
            sub_tangent = material.tangent(sub_strain)
            tangent_blocks.append(
                sub_tangent.reshape(self.batch_size, count, self.tensor_components, self.tensor_components)
            )
            start_idx = end_idx

        return torch.cat(tangent_blocks, dim=1).reshape(
            -1, self.tensor_components, self.tensor_components
        )

    def commit(self, keep_graph: bool = False) -> None:
        """
        Propagates state commitment to all managed sub-materials upon step convergence.

        Args:
            keep_graph: If True, retains autograd history across time steps for BPTT
                training. If False, detaches state tensors for online inference.
        """
        for material in self.materials:
            material.commit(keep_graph=keep_graph)

    def cancel(self) -> None:
        """
        Reverts trial internal states back to the last committed states across all phases.
        """
        for material in self.materials:
            material.cancel()

    def get_history(self) -> Tensor:
        """
        Collects, unifies, and concatenates committed internal state variables from
        each sub-material phase.

        Returns:
            Aggregated history tensor of shape [batch_size * total_points_per_sample, history_dim].
        """
        histories = []
        for material, count in self.materials_with_counts:
            h = material.get_history()
            if h.ndim == 1:
                h = h.unsqueeze(1)
            histories.append(h.reshape(self.batch_size, count, -1))
        return torch.cat(histories, dim=1).reshape(-1, histories[0].size(-1))


class Elastic3DVectorized(ConstitutiveModel):
    """
    Linear isotropic elastic material model (Hooke's Law) in 3D Voigt notation.
    Evaluates elastic stresses and constant elastic stiffness tensors without
    requiring iterative plastic corrections.
    """

    def __init__(
        self,
        young_modulus: float = 70e3,
        poisson_ratio: float = 0.3,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Initializes elastic material constants (Bulk modulus K and Shear modulus G).

        Args:
            young_modulus: Young's modulus E. Defaults to 70e3.
            poisson_ratio: Poisson's ratio nu. Defaults to 0.3.
            device: PyTorch computation device. Defaults to CPU.
            dtype: Floating-point precision. Defaults to torch.float64.
        """
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.bulk = young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
        self.shear = young_modulus / (2.0 * (1.0 + poisson_ratio))

        self.total_points = 0
        self.tensor_components = 6
        self.elastic_batch = torch.empty(0, device=self.device, dtype=self.dtype)

    def configure(self, total_points: int, tensor_components: int = 6) -> None:
        """
        Pre-computes and expands the isotropic 3D elasticity matrix D_e for the batch.

        Args:
            total_points: Number of integration points allocated to this material.
            tensor_components: Voigt stress/strain dimensions (must be 6).

        Raises:
            ValueError: If `tensor_components` is not 6.
        """
        if tensor_components != 6:
            raise ValueError("Elastic3DVectorized requires 6 Voigt tensor components.")

        self.total_points = total_points
        self.tensor_components = tensor_components

        i_voigt = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        i_vol = torch.outer(i_voigt, i_voigt)
        i_sym = torch.diag(torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], device=self.device, dtype=self.dtype))
        i_dev = i_sym - (1.0 / 3.0) * i_vol

        d_elastic = self.bulk * i_vol + 2.0 * self.shear * i_dev
        self.elastic_batch = d_elastic.unsqueeze(0).repeat(total_points, 1, 1)

    def stress(self, micro_strain: Tensor) -> Tensor:
        """
        Computes elastic Cauchy stress via batched tensor contraction (sigma = D_e : epsilon).

        Args:
            micro_strain: Strain tensor of shape [total_points, 6].

        Returns:
            Elastic stress tensor of shape [total_points, 6].
        """
        return torch.bmm(self.elastic_batch, micro_strain.unsqueeze(2)).squeeze(2)

    def tangent(self, micro_strain: Tensor) -> Tensor:
        """
        Returns the constant 3D isotropic elastic stiffness tensor D_e for all points.

        Args:
            micro_strain: Strain tensor of shape [total_points, 6] (unused, kept for API uniformity).

        Returns:
            Elastic stiffness tensor of shape [total_points, 6, 6].
        """
        return self.elastic_batch

    def commit(self, keep_graph: bool = False) -> None:
        """No internal state variables to commit in pure linear elasticity."""
        pass

    def cancel(self) -> None:
        """No trial state variables to revert in pure linear elasticity."""
        pass

    def get_history(self) -> Tensor:
        """
        Returns a zero-valued scalar tensor since linear elasticity is history-independent.

        Returns:
            Zero tensor of shape [total_points, 1].
        """
        return torch.zeros((self.total_points, 1), device=self.device, dtype=self.dtype)


class J2Material3DVectorized(ConstitutiveModel):
    """
    Fully vectorized 3D rate-independent J2 (von Mises) plasticity model with
    nonlinear isotropic hardening.

    Features:
      - User-defined hardening curve via Jive-style string expression or PyTorch callable.
      - Automatic hardening modulus H(ep) differentiation via `torch.autograd.grad`.
      - Implicit differentiation through the scalar return-mapping root so PyTorch's
        autograd computes the exact consistent tangent during training and Jive callbacks.
      - Separate vectorized `stress()` and `tangent()` methods for optimal performance.
    """

    def __init__(
        self,
        young_modulus: float = 79.5e3,
        poisson_ratio: float = 0.33,
        yield_expr: Union[str, Callable[[Tensor], Tensor]] = "90.0740 * (1. - exp(-113.0268 * x)) + 444.1184",
        hardening_expr: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        tolerance: float = 1e-12,
        max_iterations: int = 100,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Initializes the 3D J2 elastoplastic constitutive model.

        Args:
            young_modulus: Young's modulus E. Defaults to 79.5e3.
            poisson_ratio: Poisson's ratio nu. Defaults to 0.33.
            yield_expr: Yield stress law sigma_y(x) as a function of equivalent plastic
                strain `x`. Accepts a string expression or a PyTorch callable.
            hardening_expr: Optional explicit hardening modulus law H(x) = d(sigma_y)/dx.
                If None, H(x) is computed automatically via `torch.autograd.grad`.
            tolerance: Yield criterion and Newton-Raphson convergence tolerance.
            max_iterations: Maximum scalar Newton-Raphson iterations for radial return.
            device: PyTorch computation device. Defaults to CPU.
            dtype: Floating-point precision. Defaults to torch.float64.
        """
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.shear_modulus = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
        self.bulk_modulus = self.young_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))

        self.yield_spec = yield_expr if isinstance(yield_expr, str) else "<custom_callable>"
        self.yield_func = compile_math_expression(yield_expr, var_name="x")
        self.hardening_func = (
            compile_math_expression(hardening_expr, var_name="x")
            if hardening_expr is not None
            else None
        )

        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.total_points = 0
        self.tensor_components = 6

    def configure(self, total_points: int, tensor_components: int = 6) -> None:
        """
        Allocates converged and trial history variables (plastic strain tensor and
        accumulated equivalent plastic strain) and pre-computes Voigt projection operators.

        Args:
            total_points: Number of integration points allocated to this material.
            tensor_components: Voigt stress/strain dimensions (must be 6).

        Raises:
            ValueError: If `tensor_components` is not 6.
        """
        if tensor_components != 6:
            raise ValueError("J2Material3DVectorized requires 6 Voigt tensor components.")

        self.total_points = total_points
        self.tensor_components = tensor_components

        self.plastic_strain = torch.zeros(
            (total_points, self.tensor_components), device=self.device, dtype=self.dtype
        )
        self.eq_plastic_strain = torch.zeros(total_points, device=self.device, dtype=self.dtype)

        self.new_plastic_strain = self.plastic_strain.clone()
        self.new_eq_plastic_strain = self.eq_plastic_strain.clone()

        self.i_voigt = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        self.i_vol = torch.outer(self.i_voigt, self.i_voigt)
        self.i_sym = torch.diag(
            torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], device=self.device, dtype=self.dtype)
        )
        self.i_dev = self.i_sym - (1.0 / 3.0) * self.i_vol

        self.d_elastic = self.bulk_modulus * self.i_vol + 2.0 * self.shear_modulus * self.i_dev
        self.elastic_stiffness_batch = self.d_elastic.unsqueeze(0).repeat(total_points, 1, 1)

    def _yield_stress(self, eq_p_strain: Tensor) -> Tensor:
        """Evaluates the isotropic yield stress sigma_y(eq_p_strain)."""
        return self.yield_func(eq_p_strain)

    def _hardening_modulus(self, eq_p_strain: Tensor) -> Tensor:
        """Evaluates the isotropic hardening modulus H = d(sigma_y)/d(eq_p_strain)."""
        if self.hardening_func is not None:
            return self.hardening_func(eq_p_strain)

        with torch.enable_grad():
            ep = eq_p_strain.detach().requires_grad_(True)
            sig_y = self.yield_func(ep)
            (h_prime,) = torch.autograd.grad(
                sig_y, ep, grad_outputs=torch.ones_like(sig_y), create_graph=False
            )
        return h_prime

    def _solve_plastic_multiplier(self, q_tr: Tensor, eq_p_strain_old: Tensor) -> Tensor:
        """
        Solves the scalar radial-return equation f(delta_gamma) = 0 via vectorized
        Newton-Raphson and attaches exact first-order implicit derivatives so autograd
        recovers the consistent algorithmic tangent.

        Args:
            q_tr: Trial von Mises equivalent stress tensor of shape [total_points].
            eq_p_strain_old: Last committed equivalent plastic strain of shape [total_points].

        Returns:
            Differentiable plastic multiplier increment delta_gamma of shape [total_points].
        """
        with torch.no_grad():
            delta_gamma = torch.zeros_like(q_tr)
            for _ in range(self.max_iterations):
                f_val = (
                    q_tr
                    - 3.0 * self.shear_modulus * delta_gamma
                    - self._yield_stress(eq_p_strain_old + delta_gamma)
                )
                converged = torch.abs(f_val) < self.tolerance
                if torch.all(converged):
                    break
                h_prime = self._hardening_modulus(eq_p_strain_old + delta_gamma)
                f_der = -3.0 * self.shear_modulus - h_prime
                delta_gamma = delta_gamma - (f_val / f_der) * (~converged)

        # Implicit function theorem re-attachment:
        # At root f(dg, q_tr, ep_old) = 0 => d(dg) = (dq_tr - H * dep_old) / (3G + H)
        h_converged = self._hardening_modulus(eq_p_strain_old + delta_gamma).detach()
        denom = 3.0 * self.shear_modulus + h_converged
        residual = (
            q_tr
            - 3.0 * self.shear_modulus * delta_gamma
            - self._yield_stress(eq_p_strain_old + delta_gamma)
        )
        return delta_gamma + residual / denom

    def _elastic_predictor(self, strain_new: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes trial elastic stress, deviatoric stress s_tr, and von Mises equivalent
        stress q_tr from the last committed plastic strain.

        Args:
            strain_new: Total trial strain tensor of shape [total_points, 6].

        Returns:
            Tuple (stress_tr, s_tr, q_tr).
        """
        elastic_strain_tr = strain_new - self.plastic_strain
        stress_tr = torch.bmm(self.elastic_stiffness_batch, elastic_strain_tr.unsqueeze(2)).squeeze(2)

        pressure_tr = (stress_tr[:, 0] + stress_tr[:, 1] + stress_tr[:, 2]) / 3.0
        s_tr = stress_tr - pressure_tr.unsqueeze(1) * self.i_voigt

        j2_tr = 0.5 * (
            s_tr[:, 0] ** 2
            + s_tr[:, 1] ** 2
            + s_tr[:, 2] ** 2
            + 2.0 * (s_tr[:, 3] ** 2 + s_tr[:, 4] ** 2 + s_tr[:, 5] ** 2)
        )
        q_tr = torch.sqrt(3.0 * j2_tr + 1e-16)
        return stress_tr, s_tr, q_tr

    def stress(self, strain_new: Tensor) -> Tensor:
        """
        Integrates the J2 elastoplastic constitutive equations using radial return mapping
        and updates trial internal history variables without mutating committed states.

        Args:
            strain_new: Trial total strain tensor in Voigt notation [total_points, 6].

        Returns:
            Updated trial Cauchy stress tensor in Voigt notation [total_points, 6].
        """
        stress_tr, s_tr, q_tr = self._elastic_predictor(strain_new)
        yield_stress_old = self._yield_stress(self.eq_plastic_strain)

        f_tr = q_tr - yield_stress_old
        plasticity_mask = f_tr > self.tolerance

        if torch.any(plasticity_mask):
            delta_gamma_raw = self._solve_plastic_multiplier(q_tr, self.eq_plastic_strain)
            delta_gamma = torch.where(
                plasticity_mask, delta_gamma_raw, torch.zeros_like(delta_gamma_raw)
            )

            n_dir = 1.5 * s_tr / q_tr.unsqueeze(1)

            self.new_eq_plastic_strain = self.eq_plastic_strain + delta_gamma
            d_eps_p = delta_gamma.unsqueeze(1) * n_dir
            d_eps_p = torch.cat([d_eps_p[:, :3], 2.0 * d_eps_p[:, 3:]], dim=1)
            self.new_plastic_strain = self.plastic_strain + d_eps_p

            stress_plastic = stress_tr - 2.0 * self.shear_modulus * delta_gamma.unsqueeze(1) * n_dir
            stress_updated = torch.where(plasticity_mask.unsqueeze(1), stress_plastic, stress_tr)
        else:
            stress_updated = stress_tr
            self.new_plastic_strain = self.plastic_strain
            self.new_eq_plastic_strain = self.eq_plastic_strain

        return stress_updated

    def tangent(self, strain_new: Tensor) -> Tensor:
        """
        Computes the vectorized consistent algorithmic elastoplastic tangent modulus
        C_alg = d(sigma_{n+1}) / d(epsilon_{n+1}) in 3D Voigt notation:
            C_alg = K (1 (x) 1) + 2G * theta_1 * I_dev - (4/3)G * theta_2 * (n (x) n)

        Args:
            strain_new: Trial total strain tensor in Voigt notation [total_points, 6].

        Returns:
            Consistent algorithmic tangent stiffness tensor [total_points, 6, 6].
        """
        _, s_tr, q_tr = self._elastic_predictor(strain_new)
        yield_stress_old = self._yield_stress(self.eq_plastic_strain)

        f_tr = q_tr - yield_stress_old
        plasticity_mask = f_tr > self.tolerance

        if not torch.any(plasticity_mask):
            return self.elastic_stiffness_batch

        delta_gamma_raw = self._solve_plastic_multiplier(q_tr, self.eq_plastic_strain)
        delta_gamma = torch.where(
            plasticity_mask, delta_gamma_raw, torch.zeros_like(delta_gamma_raw)
        )

        h_modulus = self._hardening_modulus(self.eq_plastic_strain + delta_gamma)
        n_dir = 1.5 * s_tr / q_tr.unsqueeze(1)  # [total_points, 6]

        theta_1 = 1.0 - (3.0 * self.shear_modulus * delta_gamma) / q_tr
        theta_2 = (3.0 * self.shear_modulus) / (3.0 * self.shear_modulus + h_modulus) - (
            3.0 * self.shear_modulus * delta_gamma
        ) / q_tr

        # Outer product n (x) n -> [total_points, 6, 6]
        n_outer_n = torch.bmm(n_dir.unsqueeze(2), n_dir.unsqueeze(1))

        c_alg = (
            self.bulk_modulus * self.i_vol.unsqueeze(0)
            + 2.0 * self.shear_modulus * theta_1.view(-1, 1, 1) * self.i_dev.unsqueeze(0)
            - (4.0 / 3.0) * self.shear_modulus * theta_2.view(-1, 1, 1) * n_outer_n
        )

        return torch.where(
            plasticity_mask.view(-1, 1, 1), c_alg, self.elastic_stiffness_batch
        )

    def commit(self, keep_graph: bool = False) -> None:
        """
        Commits converged trial plastic strains to the permanent state history.

        Args:
            keep_graph: If True, preserves autograd history across time steps for
                BPTT sequence training. If False, detaches tensors for online inference.
        """
        if keep_graph:
            self.plastic_strain = self.new_plastic_strain
            self.eq_plastic_strain = self.new_eq_plastic_strain
        else:
            self.plastic_strain = self.new_plastic_strain.detach()
            self.eq_plastic_strain = self.new_eq_plastic_strain.detach()

    def cancel(self) -> None:
        """
        Discards unconverged trial plastic states and restores the last committed state.
        """
        self.new_plastic_strain = self.plastic_strain
        self.new_eq_plastic_strain = self.eq_plastic_strain

    def get_history(self) -> Tensor:
        """
        Retrieves the last committed accumulated equivalent plastic strain.

        Returns:
            Equivalent plastic strain tensor of shape [total_points].
        """
        return self.eq_plastic_strain


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    num_points = 5000
    strain_test = torch.rand((num_points, 6), device=device, dtype=torch.float64) * 0.05
    strain_test[:, 3:] = 0.0

    mat_vec = J2Material3DVectorized(
        young_modulus=79500.0,
        poisson_ratio=0.33,
        yield_expr="444.1184 + 90.0740 * (1. - exp(-113.0268 * x))",
        device=device,
    )
    mat_vec.configure(total_points=num_points)

    t0 = time.perf_counter()
    stress_out = mat_vec.stress(strain_test)
    tangent_out = mat_vec.tangent(strain_test)
    elapsed = time.perf_counter() - t0

    # Validate analytical tangent against autograd Jacobian on a single point
    mat_single = J2Material3DVectorized(device=device)
    mat_single.configure(total_points=1)
    eps_1 = torch.tensor([[0.02, -0.005, 0.0, 0.01, 0.0, 0.0]], requires_grad=True)
    sig_1 = mat_single.stress(eps_1)
    jac_autograd = torch.stack(
        [torch.autograd.grad(sig_1[0, i], eps_1, retain_graph=True)[0][0] for i in range(6)]
    )
    jac_analytic = mat_single.tangent(eps_1.detach())[0]

    print(f"Vectorized stress + tangent ({num_points} pts): {elapsed:.6f}s")
    print(f"Tangent vs Autograd max discrepancy: {torch.max(torch.abs(jac_analytic - jac_autograd)):.2e}")