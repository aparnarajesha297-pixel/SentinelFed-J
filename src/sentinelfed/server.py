import torch


def fedavg(updates: list[dict], weights: list[int]) -> dict:
    """Sample-weighted average of client state dicts (McMahan et al., 2017).

    Integer buffers (e.g. step counters) are not averaged; the first
    client's value is kept.
    """
    total = float(sum(weights))
    out = {}
    for key, ref in updates[0].items():
        if not torch.is_floating_point(ref):
            out[key] = ref.clone()
            continue
        out[key] = sum(u[key] * (w / total) for u, w in zip(updates, weights))
    return out
