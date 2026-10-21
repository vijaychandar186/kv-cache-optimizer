"""Single-request OLMo3 research cache with explicit absolute-position masks.

All full-attention layers share a retention map, selected by mean attention importance.
Sliding layers retain their contiguous recent window. Attention dequantizes one layer at
a time; this is a correctness-oriented prototype with O(T) append/dequantize/compaction.
"""

import torch
from transformers.cache_utils import Cache, DynamicLayer

from .eviction import select_retained


class ResearchLayer(DynamicLayer):
    def __init__(self, quantized: bool, backend: str):
        super().__init__()
        self.quantized = quantized
        self.backend = backend
        self.positions = None
        self.scores = None
        self.key_scale = self.value_scale = None
        self.seen = 0

    def _encode(self, x):
        if not self.quantized:
            return x.contiguous(), None
        if self.backend == "triton":
            from .kernels import quantize

            return quantize(x)
        x32 = x.float()
        scale = (x32.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        z = x32 / scale
        q = torch.where(z >= 0, (z + 0.5).floor(), (z - 0.5).ceil())
        return q.clamp(-127, 127).to(torch.int8), scale

    def _decode(self, x, scale):
        if not self.quantized:
            return x
        if self.backend == "triton":
            from .kernels import dequantize

            return dequantize(x, scale, self.dtype)
        return (x.float() * scale).to(self.dtype)

    def _gather(self, x, indices):
        if self.backend == "triton":
            from .kernels import compact

            return compact(x, indices.to(x.device))
        return x.index_select(2, indices.to(x.device)).contiguous()

    def update(self, key_states, value_states, cache_kwargs=None):
        positions = cache_kwargs["cache_position"].to(key_states.device)
        if not self.is_initialized:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.is_initialized = True
            keys, values = key_states, value_states
            self.positions = positions
            self.scores = torch.zeros(positions.numel(), device=self.device)
            self.keys, self.key_scale = self._encode(keys)
            self.values, self.value_scale = self._encode(values)
        else:
            keys = torch.cat((self._decode(self.keys, self.key_scale), key_states), dim=2)
            values = torch.cat((self._decode(self.values, self.value_scale), value_states), dim=2)
            # Quantize only new tokens: do not accumulate requantization error on old tokens.
            new_k, new_ks = self._encode(key_states)
            new_v, new_vs = self._encode(value_states)
            self.keys = torch.cat((self.keys, new_k), dim=2)
            self.values = torch.cat((self.values, new_v), dim=2)
            if self.quantized:
                self.key_scale = torch.cat((self.key_scale, new_ks), dim=2)
                self.value_scale = torch.cat((self.value_scale, new_vs), dim=2)
            self.positions = torch.cat((self.positions, positions))
            self.scores = torch.cat((self.scores, torch.zeros_like(positions, dtype=torch.float32)))
        self.seen += positions.numel()
        return keys, values

    def retain(self, indices):
        if indices.numel() == self.positions.numel():
            return
        self.keys = self._gather(self.keys, indices)
        self.values = self._gather(self.values, indices)
        if self.quantized:
            self.key_scale = self._gather(self.key_scale, indices)
            self.value_scale = self._gather(self.value_scale, indices)
        self.positions = self.positions[indices.to(self.device)]
        self.scores = self.scores[indices.to(self.device)]

    def get_seq_length(self):
        return self.seen

    def get_mask_sizes(self, cache_position):
        return (0 if self.positions is None else self.positions.numel()) + len(cache_position), 0

    def storage_bytes(self):
        tensors = (
            self.keys,
            self.values,
            self.key_scale,
            self.value_scale,
            self.positions,
            self.scores,
        )
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


class ResearchCache(Cache):
    def __init__(self, config, quantized=False, budget=None, sink=4, recent=128, backend="triton"):
        self.layer_types = config.layer_types
        self.window = config.sliding_window
        self.budget, self.sink, self.recent = budget, sink, recent
        self.device_for_policy = None
        super().__init__(
            layers=[ResearchLayer(quantized, backend) for _ in range(config.num_hidden_layers)]
        )

    def masks(self, positions, dtype, device):
        masks = {}
        for kind in set(self.layer_types):
            index = self.layer_types.index(kind)
            layer = self.layers[index]
            old = layer.positions
            keys = positions if old is None else torch.cat((old.to(device), positions))
            allowed = keys[None, :] <= positions[:, None]
            if kind == "sliding_attention":
                allowed &= keys[None, :] > positions[:, None] - self.window
            mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
            mask.masked_fill_(~allowed, torch.finfo(dtype).min)
            masks[kind] = mask[None, None, :, :]
        return masks

    def finish_step(self):
        full = [i for i, kind in enumerate(self.layer_types) if kind == "full_attention"]
        indices = None
        if full and self.budget:
            device = self.layers[full[0]].device
            scores = torch.stack([self.layers[i].scores.to(device) for i in full]).mean(0)
            indices = select_retained(scores, self.budget, self.sink, self.recent)
        for i, kind in enumerate(self.layer_types):
            layer = self.layers[i]
            if kind == "sliding_attention":
                # Keep W-1 old keys: the next query adds the Wth key.
                keep = torch.arange(
                    max(0, len(layer.positions) - self.window + 1),
                    len(layer.positions),
                    device=layer.device,
                )
                layer.retain(keep)
            elif indices is not None:
                layer.retain(indices)

    def storage_bytes(self):
        return sum(layer.storage_bytes() for layer in self.layers)


def research_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    groups = query.shape[1] // key.shape[1]
    # SDPA performs attention; compute only the final query's attention weights for scoring.
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
        enable_gqa=groups != 1,
    )
    expanded_key = key.repeat_interleave(groups, dim=1)
    scores = (query[:, :, -1:, :].float() @ expanded_key.float().transpose(-1, -2)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, -1:, :].float()
    importance = scores.softmax(-1).mean(dim=(0, 1, 2))
    layer = module._kv_lab_cache.layers[module.layer_idx]
    layer.scores = 0.9 * layer.scores + importance.to(layer.device)
    return output.transpose(1, 2).contiguous(), None
