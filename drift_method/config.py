# -*- coding: utf-8 -*-
"""Configuration container for the preference-shift subsequence modules.

``DriftConfig`` groups every hyper-parameter consumed by the adaptive partition
module, the intra-subsequence encoder and the structure constraints. Field names
follow the notation used in the paper:

* ``K``          — maximum number of historical sub-sequences.
* ``kappa_u``    — per-user effective number of sub-sequences, predicted from the
                   shift statistics and bounded by ``[K_min, K_max]``.
* ``tau_a``      — temperature of the soft assignment ``a_{u,t,k}``.
* ``tau_m``      — temperature of the activity gate ``m_{u,k}``.
* ``theta_*``    — coefficients of the linear logit that produces ``kappa_u``.
* ``r``          — length of the recent window (``recent_window_r``), i.e. the
                   number of most recent items excluded from partitioning and
                   consumed directly by the encoder.

Only the fields actually used by the main experiment are kept, so every value here
has an effect on the reported results.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DriftConfig:
    # ---- Item representation ------------------------------------------------
    codebook: Tuple[int, int, int] = (256, 256, 256)  # per-level codebook sizes
    target_depth: int = 3                  # number of SID levels per item (M)
    pad_token_id: int = 0
    max_seq_len: int = 50                  # history length in items (L)
    recent_window_r: int = 5               # recent-window size r
    d_model: int = 96                      # hidden size, matches the backbone

    # ---- Adaptive soft partition -------------------------------------------
    K: int = 5                             # global maximum number of sub-sequences
    K_min: float = 1.0                     # lower bound of kappa_u
    K_max: float = 5.0                     # upper bound of kappa_u (<= K)
    tau_a: float = 0.5                     # soft-assignment temperature
    tau_m: float = 0.3                     # activity-gate temperature
    use_gumbel: bool = True                # Gumbel-Softmax perturbation on assignment
    gumbel_tau: float = 1.0                # Gumbel temperature
    coord_normalize: str = "kappa"          # cumulative-coordinate normalization:
                                           #   "raw" / "K" / "kappa"; "kappa" divides
                                           #   by the predicted kappa_u so that the
                                           #   partition granularity adapts per user.
    # Linear-logit coefficients that produce kappa_u from the shift statistics.
    theta0: float = 0.0                    # bias
    theta_d: float = 1.5                   # weight on the shift degree d_u
    theta_L: float = 0.5                   # weight on the normalized history length
    theta_b: float = 0.5                   # weight on the boundary-intensity statistic
    theta_v: float = 0.5                   # weight on the representation-variance statistic
    L_max_ref: int = 200                   # reference length used to normalize ell_u

    # ---- Intra-sub-sequence encoder ----------------------------------------
    intra_encoder: str = "linear_mha"      # "linear_mha" (default) / "mha" /
                                           # "sparse_mha" / "linear_fusion"
    intra_num_heads: int = 4               # attention heads (must divide d_model)
    intra_topk: int = 8                    # top-k for the "sparse_mha" variant

    # ---- Distribution / semantic statistics --------------------------------
    sid_layer1_size: int = 256             # size of the first-level SID vocabulary
    jsd_smooth_eps: float = 1e-3           # smoothing added to empirical distributions

    # ---- Timestamps ---------------------------------------------------------
    use_real_timestamp: bool = True        # use real interaction timestamps for the
                                           # temporal terms; when False (or when the
                                           # dataset has no timestamps) those terms
                                           # fall back to a constant with zero gradient.
    time_unit_scale: float = 86400.0       # seconds -> days, keeps log1p(dt) well scaled
    ts_min_dt_floor: float = 1.0 / 86400.0  # lower bound on dt (1 second), avoids log(0)

    # ---- Boundary-density statistics ---------------------------------------
    rho_min: float = 0.05                  # target boundary-density lower bound
    rho_max: float = 0.3                   # target boundary-density upper bound
    huber_delta: float = 0.1               # Huber delta for the density statistic

    # ---- Numerics / misc ----------------------------------------------------
    eps: float = 1e-8
    seed: int = 2025
    device: str = "cuda"

    def __post_init__(self):
        assert self.recent_window_r >= 1, "recent_window_r must be >= 1"
        assert self.recent_window_r < self.max_seq_len, \
            "recent_window_r must be < max_seq_len, otherwise no long-term history remains"
        assert self.K_max <= self.K, "K_max must be <= K (upper bound on sub-sequences)"


# Backward-compatible alias used by some module-level type hints.
DAPSConfig = DriftConfig
