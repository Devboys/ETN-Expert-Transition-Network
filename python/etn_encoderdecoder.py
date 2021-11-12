import torch
from torch import nn

__all__ = ['Encoder', 'Decoder']

class Encoder(nn.Module):
    def __init__(self, dims: tuple):
        """
        Creates a simple 3-layer encoder network.

        :param dims: tuple of size 3. Defines the dimensions of the encoder in order -> (input, h0, output)
        """
        super().__init__()
        self.dims = dims

        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.LeakyReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.LeakyReLU(),
        )

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        """
        The forward operation of the encoder network.

        :param input_vec: The input vector
        :return: An encoded input vector.
        """
        return self.layers(input_vec)


class Decoder(nn.Module):
    def __init__(self, dims):
        """
        Creates a simple 4-layer decoder network.

        :param dims: tuple of size 3. Defines the dimensions of the encoder in order -> (input, h0, h1, output)
        """
        super().__init__()
        self.dims = dims

        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.LeakyReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.LeakyReLU(),
            nn.Linear(dims[2], dims[3])
        )

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        """
        The forward operation of the encoder network.

        :param input_vec: The input vector
        :return: An encoded input vector.
        """
        return self.layers(input_vec)
