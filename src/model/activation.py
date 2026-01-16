from torch import nn, Tensor, tensor, pow, sqrt, tanh, pi


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the GELU activation function as described in the paper "Gaussian Error Linear Units (GELUs)" by Hendrycks and Gimpel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying GELU.
        """
        return 0.5 * x * (1 + tanh(sqrt(tensor(2.0 / pi)) * (x + 0.044715 * pow(x, 3))))
