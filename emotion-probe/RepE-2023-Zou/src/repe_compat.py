"""repe_compat — 让 RepE 在 CPU / MPS / 任意 device 上跑通的兼容层。

RepE 源码 repe/rep_readers.py 里 `project_onto_direction` / `recenter` 硬编码了
`.cuda()`（第 13/24/26/28 行，已逐字核对 /tmp/repe_ref），本地无 NVIDIA GPU 时
（macOS CPU/MPS）必崩。这里 monkeypatch 这两个函数为 device-agnostic 版本：
张量统一用 torch.as_tensor，mean 跟随输入 device，GPU 端同样安全。

另外 `apply_compat()` 还负责注册 RepE 的 rep-reading pipeline（幂等）。

用法：
    from repe_compat import apply_compat
    apply_compat()                       # 打补丁 + 注册 pipeline
    from transformers import pipeline
    pipe = pipeline("rep-reading", model=..., tokenizer=...)
"""
from __future__ import annotations

import numpy as np
import torch

_PATCHED = False
_REGISTERED = False


def _project_onto_direction(H, direction):
    """Project matrix H (n, d) onto direction vector (d,) — device-agnostic 版。

    原版强制 .cuda()，这里改为：把 direction 作为 device 锚点，H 跟随
    direction.device；都不是张量时落在 CPU。语义与原版完全一致：H·dir/‖dir‖。
    """
    if not isinstance(direction, torch.Tensor):
        direction = torch.as_tensor(direction, dtype=torch.float32)
    # device 锚点：direction 在哪，H 就转到哪
    target_device = direction.device
    if not isinstance(H, torch.Tensor):
        H = torch.as_tensor(H, dtype=torch.float32, device=target_device)
    else:
        H = H.to(target_device)
    if H.dtype != direction.dtype:
        H = H.to(direction.dtype)
    mag = torch.norm(direction)
    assert not torch.isinf(mag).any()
    projection = H.matmul(direction) / mag
    return projection


def _recenter(x, mean=None):
    """减去均值的 recenter — device-agnostic 版（原版强制 .cuda()）。"""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        x = x.float() if x.dtype == torch.bfloat16 else x
    if mean is None:
        mean = torch.mean(x, dim=0, keepdim=True)
    else:
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
        else:
            mean = mean.to(device=x.device, dtype=x.dtype)
    return x - mean


def apply_compat(register_pipeline: bool = True) -> None:
    """打 device-agnostic 补丁（幂等），可选注册 rep-reading pipeline。"""
    global _PATCHED, _REGISTERED

    if not _PATCHED:
        import repe.rep_readers as rr

        # 替换模块级函数 + 同模块内已绑定的引用（get_signs/transform 内直接调用名字，
        # 它们在调用时从模块全局查找，所以替换模块属性即可生效）。
        rr.project_onto_direction = _project_onto_direction
        rr.recenter = _recenter
        _PATCHED = True

    if register_pipeline and not _REGISTERED:
        from repe import repe_pipeline_registry

        repe_pipeline_registry()
        _REGISTERED = True


def patched_signatures() -> dict:
    """返回补丁状态，供 smoke/manifest 记录。"""
    return {
        "project_onto_direction": "device-agnostic (anchor=direction.device)",
        "recenter": "device-agnostic (follows x.device)",
        "patched": _PATCHED,
        "pipeline_registered": _REGISTERED,
    }
