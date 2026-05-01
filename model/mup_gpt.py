"""µP wiring. Builds a GPT with mup attention scaling and MuReadout, then applies set_base_shapes.

The proxy (base/delta) models must have the same n_layer as the actual model because
mup.set_base_shapes matches parameters by name. Shapes are always computed in memory
since they differ by depth and cannot be shared across model sizes in a single file.
"""
import copy
from pathlib import Path

from model.gpt import GPT, GPTConfig


def _make_proxy_cfg(template: GPTConfig, d_model: int) -> GPTConfig:
    cfg = copy.deepcopy(template)
    cfg.d_model = d_model
    cfg.d_ff = 4 * d_model
    # n_head must divide d_model. 4 divides both 128 and 256, and n_head does not
    # affect parameter shapes so any valid divisor works here.
    cfg.n_head = 4
    # n_layer is kept from template -- parameter names (blocks.0 .. blocks.n_layer-1)
    # must match exactly between proxy and actual model.
    cfg.attn_scale_mode = "mup"
    cfg.mup_readout = True
    cfg.tie_weights = False
    return cfg


def build_mup_model(cfg: GPTConfig, base_shapes_path: Path | None = None) -> GPT:
    from mup import set_base_shapes

    # base_shapes_path is accepted for call-site compatibility but ignored;
    # shapes are model-depth-specific and computed in memory.
    cfg = copy.deepcopy(cfg)
    cfg.attn_scale_mode = "mup"
    cfg.mup_readout = True
    cfg.tie_weights = False

    base_cfg = _make_proxy_cfg(cfg, d_model=128)
    delta_cfg = _make_proxy_cfg(cfg, d_model=256)

    base_model = GPT(base_cfg)
    delta_model = GPT(delta_cfg)

    actual = GPT(cfg)
    set_base_shapes(actual, base_model, delta=delta_model)

    del base_model, delta_model
    return actual
