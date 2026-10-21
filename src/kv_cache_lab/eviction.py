"""Keep sinks, recent tokens, and tokens with the highest observed attention scores."""


def select_retained(scores, budget: int, sink: int, recent: int):
    """Torch implementation used on GPU; indices stay chronological for RoPE and masks."""
    import torch

    count = scores.numel()
    if budget < 1 or sink < 0 or recent < 1 or sink + recent > budget:
        raise ValueError("Require budget >= sink + recent, sink >= 0, recent >= 1")
    if count <= budget:
        return torch.arange(count, device=scores.device)
    keep = torch.zeros(count, dtype=torch.bool, device=scores.device)
    keep[:sink] = True
    keep[-recent:] = True
    candidates = scores.clone()
    candidates[keep] = -torch.inf
    remaining = budget - sink - recent
    if remaining:
        # Stable sort gives deterministic tie handling.
        keep[torch.argsort(candidates, descending=True, stable=True)[:remaining]] = True
    return torch.where(keep)[0]
