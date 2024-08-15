from torch import arange, cat, cos, float64, sin, stack, Tensor, tensor
from torch.autograd import grad, set_grad_enabled
from torch.linalg import cross, norm
from torch.nn import Linear, Module, ReLU, Sequential


class HarmonicEmbedding(Module):
    """Computes cosines and sines of coordinates, each multiplied by an exponentially growing sequence of frequencies"""
    def __init__(self, num_freqs: int, base_freq: float, dtype=float64):
        """
        Args:
            num_freqs (int): number of frequencies per coordinate
            base_freq (float): multiplier for all frequencies
        """
        Module.__init__(self)
        freqs = base_freq * (2 ** arange(num_freqs, dtype=dtype))
        self.register_buffer('freqs', freqs)
        
    def forward(self, x: Tensor) -> Tensor:
        """Computes harmonic embedding

        Args:
            x (Tensor): batch_dims * d input

        Returns:
            batch_dims * (d * 2 * num_freqs) embedding
        """
        batch_dims = x.shape[:-1]
        reshaped_freqs = self.freqs.reshape(tuple(1 for _ in batch_dims) + (1, -1))
        args = (reshaped_freqs * x.unsqueeze(-1)).flatten(start_dim=-2)
        emb = cat([cos(args), sin(args)], dim=-1)
        return emb
    

class SphericalSurfaceModel(Module):
    """Neural map from the unit sphere S^2 to R^3 space"""
    def __init__(self, num_freqs: int, base_freq: float, hidden_dim: int, num_hidden_layers: int, dtype=float64):
        """
        Args:
            num_freqs (int): number of frequencies per coordinate in harmonic embedding
            base_freq (float): multiplier for all frequencies in harmonic embedding
            num_hidden_layers (int): number of hidden layers in MLP
        """
        Module.__init__(self)

        emb_dim = 3 * 2 * num_freqs
        layers = [HarmonicEmbedding(num_freqs, base_freq, dtype=dtype), Linear(emb_dim, hidden_dim, dtype=dtype), ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [Linear(hidden_dim, hidden_dim, dtype=dtype), ReLU()]
        layers += [Linear(hidden_dim, 3, dtype=dtype)]
        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Maps query points on the sphere to point on surface

        Args:
            x (Tensor): batch_dims * 3 list of points on unit sphere

        Returns:
            batch_dims * 3 list of points on surface
        """
        return self.layers(x)
    
    def normal(self, x: Tensor) -> Tensor:
        """Maps query points on the sphere to unit normal vectors to the surface

        Args:
            x (Tensor): batch_dims * 3 list of points on unit sphere

        Returns:
            batch_dims * 3 list of vectors normal to surface
        """
        grads = []
        with set_grad_enabled(True):
            x_with_grad = x.clone().requires_grad_(True)
            for i in range(3):
                grad_i, = grad((x_with_grad + self(x_with_grad))[..., i].sum(), x_with_grad, create_graph=True)
                grads.append(grad_i)

        jac = stack(grads, dim=-2)
        jac_crosses = cross(jac[..., tensor([1, 2, 0]), :], jac[..., tensor([2, 0, 1]), :], dim=-2).transpose(-2, -1)
        N = (jac_crosses * x.unsqueeze(-2)).sum(dim=-1)
        N = N / norm(N, dim=-1, keepdims=True)
        return N
    