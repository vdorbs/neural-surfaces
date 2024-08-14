from torch import arange, cat, cos, float64, ReLU, sin, Tensor
from torch.nn import Linear, Module, Sequential


class HarmonicEmbedding(Module):
    def __init__(self, num_freqs: int, base_freq: float, dtype=float64):
        Module.__init__(self)
        freqs = base_freq * (2 ** arange(num_freqs, dtype=dtype))
        self.register_buffer(freqs)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_dims = x.shape[:-1]
        reshaped_freqs = self.freqs.reshape(tuple(1 for _ in batch_dims) + (1, -1))
        args = (reshaped_freqs * x.unsqueeze(-1)).flatten(start_dim=-2)
        emb = cat([cos(args), sin(args)], dim=-1)
        return emb
    

class SphericalSurfaceModel(Module):
    def __init__(self, num_freqs: int, base_freq: float, hidden_dim: int, num_hidden_layers: int, dtype=float64):
        Module.__init__(self)

        emb_dim = 3 * 2 * num_freqs
        layers = [HarmonicEmbedding(num_freqs, base_freq, dtype=dtype), Linear(emb_dim, hidden_dim, dtype=dtype), ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [Linear(hidden_dim, hidden_dim, dtype=dtype), ReLU()]
        layers += [Linear(hidden_dim, 3, dtype=dtype)]
        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
    
    def normal(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    