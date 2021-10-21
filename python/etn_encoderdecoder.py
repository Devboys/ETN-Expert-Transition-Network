import torch
from torch import nn

__all__ = ['Encoder', 'Decoder']

class Encoder(nn.Module):
    def __init__(self, input_size, h0_size, output_size):
        """
        Creates a simple 3-layer encoder network.

        :param input_size: num nodes in input layer
        :param h0_size: num nodes in hidden layer
        :param output_size: num nodes in output layer
        """
        super().__init__()
        self.input_size = input_size
        self.h0_size = h0_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, h0_size),
            nn.LeakyReLU(),
            nn.Linear(h0_size, output_size),
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
    def __init__(self, input_size, h0_size, h1_size, output_size):
        """
        Creates a simple 4-layer decoder network.
        :param input_size: Num nodes in input layer
        :param h0_size: num nodes in first hidden layer
        :param h1_size: num nodes in second hidden layer
        :param output_size: num nodes in output layer
        """
        super().__init__()
        self.input_size = input_size
        self.h0_size = h0_size
        self.h1_size = h1_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, self.h0_size),
            nn.LeakyReLU(),
            nn.Linear(self.h0_size, self.h1_size),
            nn.LeakyReLU(),
            nn.Linear(self.h1_size, self.output_size)
        )

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        """
        The forward operation of the encoder network.

        :param input_vec: The input vector
        :return: An encoded input vector.
        """
        return self.layers(input_vec)
