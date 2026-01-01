from torch import nn, Tensor, tensor, pow, sqrt, tanh, pi


class GELU(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1 + tanh(sqrt(tensor(2.0 / pi)) * (x + 0.044715 * pow(x, 3))))
