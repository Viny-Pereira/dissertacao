"""
Physical constitutive models for the PRNN architecture.
Includes both vectorized and looped implementations for performance benchmarking.
"""

import time
import torch
from .interfaces import Material
from typing import List, Tuple, Optional


class CompositeMaterial(Material):
    """
    Composite material manager that orchestrates multiple constitutive models in parallel.
    Partitions input strain tensors across different material phases and aggregates 
    resulting stresses and algorithmic tangent moduli.
    """
    def __init__(self, materials_with_counts: List[Tuple[Material, int]]):
        """
        Args:
            materials_with_counts: List of tuples containing material instances and their 
                                   respective material point count.
                                   Example: [(Elastic3DVectorized(), 3), (J2Material3DVectorized(), 2)]
        """
        super().__init__()
        self.materials_with_counts = materials_with_counts
        self.materials = [m for m, _ in materials_with_counts]
        self.counts = [c for _, c in materials_with_counts]
        self.total_points_per_sample = sum(self.counts)
        self.batch_size = 0
        self.tensor_components = 6

        if self.total_points_per_sample <= 0:
            raise ValueError("The total number of material points per sample must be strictly positive.")

    def configure(self, total_points: int, tensor_components: int = 6) -> None:
        """
        Allocates memory and configures state variables for each sub-material 
        based on the global batch size.

        Args:
            total_points (int): Flattened number of integration points (batch_size * total_points_per_sample).
            tensor_components (int, optional): Voigt stress/strain tensor dimension. Defaults to 6.
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

    def update(self, micro_strain: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Partitions the flattened micro-strain tensor across sub-materials, evaluates 
        their individual constitutive laws, and recombines the results.

        Args:
            micro_strain (torch.Tensor): Micro-strain tensor of shape 
                                         [batch_size * total_points_per_sample, tensor_components].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Combined micro-stress tensor [batch_size * total_points_per_sample, tensor_components].
                - Combined algorithmic tangent matrix [batch_size * total_points_per_sample, 6, 6].
        """
        strain_reshaped = micro_strain.view(self.batch_size, self.total_points_per_sample, self.tensor_components)

        stress_blocks = []
        tangent_blocks = []
        start_idx = 0

        for material, count in self.materials_with_counts:
            end_idx = start_idx + count
            
            # Slice strain for this specific material phase
            sub_strain = strain_reshaped[:, start_idx:end_idx, :].reshape(-1, self.tensor_components)
            sub_stress, sub_tangent = material.update(sub_strain)
            
            stress_blocks.append(sub_stress.view(self.batch_size, count, self.tensor_components))
            tangent_blocks.append(sub_tangent.view(self.batch_size, count, self.tensor_components, self.tensor_components))
            
            start_idx = end_idx

        # Recombine across all material points
        macro_stress_combined = torch.cat(stress_blocks, dim=1).view(-1, self.tensor_components)
        macro_tangent_combined = torch.cat(tangent_blocks, dim=1).view(
            -1, self.tensor_components, self.tensor_components
        )

        return macro_stress_combined, macro_tangent_combined

    def commit(self) -> None:
        """Propagates state commitment to all managed sub-materials upon step convergence."""
        for material in self.materials:
            material.commit()

    def get_history(self) -> torch.Tensor:
        """
        Collects, unifies, and concatenates internal state variables from each sub-material.

        Returns:
            torch.Tensor: Aggregated history tensor of shape [batch_size * total_points_per_sample, history_dim].
        """
        histories = []
        for material, count in self.materials_with_counts:
            h = material.get_history()
            # Ensure consistent 2D shape [batch_size * count, history_dim]
            if h.ndim == 1:
                h = h.unsqueeze(1)
            histories.append(h.view(self.batch_size, count, -1))
        
        return torch.cat(histories, dim=1).view(-1, histories[0].size(-1))


class Elastic3DVectorized(Material):
    """
    Linear isotropic elastic material model (Hooke's Law) in Voigt notation.
    Evaluates stress states without requiring iterative plastic corrections.
    """
    def __init__(self, young_modulus: float = 70e3, poisson_ratio: float = 0.3, 
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64):
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
        Pre-computes and expands the isotropic 3D elasticity matrix for the batch.

        Args:
            total_points (int): Number of integration points for this material phase.
            tensor_components (int, optional): Voigt stress/strain dimensions. Defaults to 6.
        """
        self.total_points = total_points
        self.tensor_components = tensor_components
        
        i_voigt = torch.tensor([1., 1., 1., 0., 0., 0.], device=self.device, dtype=self.dtype)
        i_vol = torch.outer(i_voigt, i_voigt)
        i_sym = torch.diag(torch.tensor([1., 1., 1., 0.5, 0.5, 0.5], device=self.device, dtype=self.dtype))
        i_dev = i_sym - (1.0 / 3.0) * i_vol
        
        d_elastic = self.bulk * i_vol + 2.0 * self.shear * i_dev
        self.elastic_batch = d_elastic.unsqueeze(0).repeat(total_points, 1, 1)

    def update(self, micro_strain: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes elastic stress via direct tensor contraction (sigma = D : epsilon).

        Args:
            micro_strain (torch.Tensor): Micro-strain tensor of shape [total_points, tensor_components].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Stress tensor [total_points, tensor_components].
                - Elastic stiffness matrix [total_points, 6, 6].
        """
        stress = torch.bmm(self.elastic_batch, micro_strain.unsqueeze(2)).squeeze(2)
        return stress, self.elastic_batch

    def commit(self) -> None:
        """No internal state variables to commit in pure linear elasticity."""
        pass

    def get_history(self) -> torch.Tensor:
        """
        Returns a zero-valued scalar tensor since linear elasticity is history-independent.
        """
        return torch.zeros((self.total_points, 1), device=self.device, dtype=self.dtype)


class J2Material3DVectorized(Material):
    """
    Fully vectorized 3D J2 Plasticity model.
    Processes the entire batch of material points simultaneously using tensor operations.
    """
    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # Elastic properties
        self.young_modulus = 79.5e3
        self.poisson_ratio = 0.33

        # Lamé constants
        self.shear_modulus = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
        self.bulk_modulus = self.young_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))

        self.tolerance = 1e-8
        self.max_iterations = 100

    def configure(self, total_points: int, tensor_components: int = 6) -> None:
        """
        Initializes the state variables and auxiliary tensors for the vectorized batch.
        """
        self.total_points = total_points
        self.tensor_components = tensor_components

        # History variables: Plastic strain tensor and equivalent plastic strain scalar
        self.plastic_strain = torch.zeros((total_points, self.tensor_components), 
                                          device=self.device, dtype=torch.float64)
        self.eq_plastic_strain = torch.zeros(total_points, 
                                             device=self.device, dtype=torch.float64)
        
        self.new_plastic_strain = torch.zeros_like(self.plastic_strain)
        self.new_eq_plastic_strain = torch.zeros_like(self.eq_plastic_strain)

        # Voigt notation identity vector [1, 1, 1, 0, 0, 0]
        self.i_voigt = torch.tensor([1., 1., 1., 0., 0., 0.], 
                                    device=self.device, dtype=torch.float64)

        # Auxiliary projection matrices in Voigt notation
        self.i_vol = torch.outer(self.i_voigt, self.i_voigt)
        self.i_sym = torch.diag(torch.tensor([1., 1., 1., 0.5, 0.5, 0.5], 
                                             device=self.device, dtype=torch.float64))
        self.i_dev = self.i_sym - (1.0 / 3.0) * self.i_vol

        # 3D Elastic Stiffness Matrix
        self.d_elastic = self.bulk_modulus * self.i_vol + 2.0 * self.shear_modulus * self.i_dev

        # Expand the elastic matrix for all integration points [total_points, 6, 6]
        self.elastic_stiffness_batch = self.d_elastic.unsqueeze(0).repeat(total_points, 1, 1)

    def _yield_stress(self, eq_p_strain: torch.Tensor) -> torch.Tensor:
        """Hardening law (Yield Stress)."""
        return 444.1184 + 90.0740 * (1.0 - torch.exp(-eq_p_strain * 113.0268))

    def _hardening_modulus(self, eq_p_strain: torch.Tensor) -> torch.Tensor:
        """Derivative of the hardening law (H)."""
        return 90.0740 * 113.0268 * torch.exp(-eq_p_strain * 113.0268)

    def update(self, strain_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constitutive evaluation: Receives new strain, returns updated stress and tangent matrix.
        """
        # 1. Elastic Predictor (Trial Step)
        elastic_strain_tr = strain_new - self.plastic_strain
        stress_tr = torch.bmm(self.elastic_stiffness_batch, elastic_strain_tr.unsqueeze(2)).squeeze(2)

        # 2. Hydrostatic and Deviatoric Decomposition
        pressure_tr = (stress_tr[:, 0] + stress_tr[:, 1] + stress_tr[:, 2]) / 3.0
        s_tr = stress_tr - pressure_tr.unsqueeze(1) * self.i_voigt

        # 3. J2 Invariant and von Mises Equivalent Stress (q_tr)
        j2_tr = 0.5 * (s_tr[:, 0]**2 + s_tr[:, 1]**2 + s_tr[:, 2]**2 +
                       2.0 * (s_tr[:, 3]**2 + s_tr[:, 4]**2 + s_tr[:, 5]**2))
        q_tr = torch.sqrt(3.0 * j2_tr + 1e-10)

        yield_stress_old = self._yield_stress(self.eq_plastic_strain)

        # 4. Yield Criterion
        f_tr = q_tr - yield_stress_old
        plasticity_mask = f_tr > self.tolerance

        # Initialize with elastic assumption
        stress_updated = stress_tr.clone()
        tangent_matrix = self.elastic_stiffness_batch.clone()

        self.new_plastic_strain = self.plastic_strain.clone()
        self.new_eq_plastic_strain = self.eq_plastic_strain.clone()

        # 5. Return Mapping (Plastic Correction) via Boolean Masking
        if torch.any(plasticity_mask):
            q_tr_p = q_tr[plasticity_mask]
            eq_p_strain_old_p = self.eq_plastic_strain[plasticity_mask]

            # Root finding for plastic multiplier
            delta_gamma = self._find_root_3d(q_tr_p, eq_p_strain_old_p)

            # Plastic flow direction: n = (3/2) * s / q
            n_dir = 1.5 * s_tr[plasticity_mask] / q_tr_p.unsqueeze(1)

            # Update scalar history
            self.new_eq_plastic_strain[plasticity_mask] += delta_gamma

            # Update tensorial plastic strain (accounting for Voigt shear factor 2)
            d_eps_p = delta_gamma.unsqueeze(1) * n_dir
            d_eps_p[:, 3:] *= 2.0
            self.new_plastic_strain[plasticity_mask] += d_eps_p

            # Update Stress (Radial Return)
            stress_updated[plasticity_mask] -= 2.0 * self.shear_modulus * delta_gamma.unsqueeze(1) * n_dir

            # 6. Consistent Algorithmic Tangent Matrix
            h_modulus = self._hardening_modulus(eq_p_strain_old_p + delta_gamma)

            theta1 = 1.0 - (3.0 * self.shear_modulus * delta_gamma) / q_tr_p
            theta2 = ((3.0 * self.shear_modulus) / (3.0 * self.shear_modulus + h_modulus) 
                      - (3.0 * self.shear_modulus * delta_gamma) / q_tr_p)

            i_dev_batch = self.i_dev.unsqueeze(0).repeat(delta_gamma.size(0), 1, 1)
            i_vol_batch = self.i_vol.unsqueeze(0).repeat(delta_gamma.size(0), 1, 1)

            # n tensor n (outer product in Voigt)
            n_tensor_n = torch.bmm(n_dir.unsqueeze(2), n_dir.unsqueeze(1))

            c_algorithmic = (self.bulk_modulus * i_vol_batch +
                             2.0 * self.shear_modulus * theta1.unsqueeze(1).unsqueeze(2) * i_dev_batch -
                             (4.0 / 3.0) * self.shear_modulus * theta2.unsqueeze(1).unsqueeze(2) * n_tensor_n)

            tangent_matrix[plasticity_mask] = c_algorithmic

        return stress_updated, tangent_matrix

    def _find_root_3d(self, q_tr: torch.Tensor, eq_p_strain_old: torch.Tensor) -> torch.Tensor:
        """Scalar Newton-Raphson for the Return Mapping algorithm."""
        delta_gamma = torch.zeros_like(q_tr)

        for _ in range(self.max_iterations):
            f_val = q_tr - 3.0 * self.shear_modulus * delta_gamma - self._yield_stress(eq_p_strain_old + delta_gamma)

            converged = torch.abs(f_val) < self.tolerance
            if torch.all(converged):
                break

            h_prime = self._hardening_modulus(eq_p_strain_old + delta_gamma)
            f_der = -3.0 * self.shear_modulus - h_prime

            d_delta_gamma = f_val / f_der
            delta_gamma -= d_delta_gamma * ~converged

        return delta_gamma

    def commit(self) -> None:
        """Commits the converged internal history variables."""
        self.plastic_strain = self.new_plastic_strain.clone()
        self.eq_plastic_strain = self.new_eq_plastic_strain.clone()

    def get_history(self) -> torch.Tensor:
        """Returns the accumulated equivalent plastic strain."""
        return self.new_eq_plastic_strain


class J2Material3DLooped(J2Material3DVectorized):
    """
    Legacy 3D J2 Plasticity model.
    Processes material points one by one using a Python 'for' loop.
    Inherits initialization from the vectorized class to avoid code duplication.
    """
    def update(self, strain_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constitutive evaluation: Loops through each material point sequentially.
        """
        stress_updated = torch.zeros_like(strain_new)
        tangent_matrix = torch.zeros_like(self.elastic_stiffness_batch)
        
        self.new_plastic_strain = self.plastic_strain.clone()
        self.new_eq_plastic_strain = self.eq_plastic_strain.clone()

        # Sequential loop over every material point
        for i in range(self.total_points):
            eps_new_i = strain_new[i]
            
            # 1. Elastic Predictor
            eps_el_tr_i = eps_new_i - self.plastic_strain[i]
            sig_tr_i = torch.matmul(self.d_elastic, eps_el_tr_i)
            
            # 2. Hydrostatic and Deviatoric Decomposition
            p_tr_i = (sig_tr_i[0] + sig_tr_i[1] + sig_tr_i[2]) / 3.0
            s_tr_i = sig_tr_i - p_tr_i * self.i_voigt
            
            # 3. J2 and von Mises
            j2_tr_i = 0.5 * (s_tr_i[0]**2 + s_tr_i[1]**2 + s_tr_i[2]**2 +
                             2.0 * (s_tr_i[3]**2 + s_tr_i[4]**2 + s_tr_i[5]**2))
            q_tr_i = torch.sqrt(3.0 * j2_tr_i + 1e-10)
            
            yield_stress_old_i = self._yield_stress(self.eq_plastic_strain[i])
            f_tr_i = q_tr_i - yield_stress_old_i
            
            # 4. Return Mapping
            if f_tr_i > self.tolerance:
                delta_gamma_i = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                
                # Newton-Raphson scalar loop for a single point
                for _ in range(self.max_iterations):
                    f_val_i = q_tr_i - 3.0 * self.shear_modulus * delta_gamma_i - self._yield_stress(self.eq_plastic_strain[i] + delta_gamma_i)
                    if torch.abs(f_val_i) < self.tolerance:
                        break
                    h_prime_i = self._hardening_modulus(self.eq_plastic_strain[i] + delta_gamma_i)
                    f_der_i = -3.0 * self.shear_modulus - h_prime_i
                    delta_gamma_i -= f_val_i / f_der_i
                
                n_dir_i = 1.5 * s_tr_i / q_tr_i
                
                self.new_eq_plastic_strain[i] += delta_gamma_i
                
                d_eps_p_i = delta_gamma_i * n_dir_i
                d_eps_p_i[3:] *= 2.0
                self.new_plastic_strain[i] += d_eps_p_i
                
                stress_updated[i] = sig_tr_i - 2.0 * self.shear_modulus * delta_gamma_i * n_dir_i
                
                # Tangent Matrix
                h_modulus_i = self._hardening_modulus(self.eq_plastic_strain[i] + delta_gamma_i)
                theta1_i = 1.0 - (3.0 * self.shear_modulus * delta_gamma_i) / q_tr_i
                theta2_i = ((3.0 * self.shear_modulus) / (3.0 * self.shear_modulus + h_modulus_i) 
                            - (3.0 * self.shear_modulus * delta_gamma_i) / q_tr_i)
                
                n_tensor_n_i = torch.outer(n_dir_i, n_dir_i)
                
                tangent_matrix[i] = (self.bulk_modulus * self.i_vol +
                                     2.0 * self.shear_modulus * theta1_i * self.i_dev -
                                     (4.0 / 3.0) * self.shear_modulus * theta2_i * n_tensor_n_i)
            else:
                stress_updated[i] = sig_tr_i
                tangent_matrix[i] = self.d_elastic
                
        return stress_updated, tangent_matrix


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')

    num_points = 5000  # Large batch to see the performance difference
    
    print(f"--- Benchmarking J2Material3D for {num_points} integration points ---")
    
    # Generate random strain increments (some elastic, some heavily plastic)
    strain_test = torch.rand((num_points, 6), device=device, dtype=torch.float64) * 0.05
    strain_test[:, 3:] = 0.0  # Zero out initial shear for simplicity

    # 1. Vectorized Test
    mat_vec = J2Material3DVectorized(device)
    mat_vec.configure(total_points=num_points)
    
    start_time = time.perf_counter()
    stress_vec, tangent_vec = mat_vec.update(strain_test)
    end_time_vec = time.perf_counter() - start_time
    
    print(f"Vectorized Execution Time: {end_time_vec:.6f} seconds")

    # 2. Looped Test
    mat_loop = J2Material3DLooped(device)
    mat_loop.configure(total_points=num_points)
    
    start_time = time.perf_counter()
    stress_loop, tangent_loop = mat_loop.update(strain_test)
    end_time_loop = time.perf_counter() - start_time
    
    print(f"Looped Execution Time:     {end_time_loop:.6f} seconds")
    print(f"Performance Gain:          {end_time_loop / end_time_vec:.2f}x faster")
    
    # Verify mathematical equivalence
    stress_diff = torch.max(torch.abs(stress_vec - stress_loop))
    print(f"Maximum Stress Discrepancy between methods: {stress_diff:.2e}")