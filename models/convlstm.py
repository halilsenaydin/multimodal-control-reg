import torch
import torch.nn as nn
from torch import Tensor


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell implementation."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        """Initialize ConvLSTM cell.

        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
            kernel_size: Kernel size for convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        # Convolutional layer for gates
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        # Initialize forget-gate bias to 1 for better gradient flow at step 0.
        # Gates are ordered [i, f, o, g]; forget gate occupies the second quarter.
        nn.init.constant_(self.conv.bias[hidden_channels : 2 * hidden_channels], 1.0)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of ConvLSTM cell.

        Args:
            x: Input tensor of shape [B, C, H, W]
            h: Hidden state tensor of shape [B, C, H, W]
            c: Cell state tensor of shape [B, C, H, W]

        Returns:
            Tuple of (next_hidden, next_cell) tensors
        """
        # Concatenate input and hidden state along channel dimension
        concat = torch.cat([x, h], dim=1)

        # Apply convolution
        gates = self.conv(concat)

        # Split into 4 gates
        i, f, o, g = gates.chunk(4, dim=1)

        # Apply activation functions
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Compute next cell state
        c_next = f * c + i * g

        # Compute next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """ConvLSTM network implementation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int = 1,
    ):
        """Initialize ConvLSTM.

        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
            kernel_size: Kernel size for convolution
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # Create LSTM layers
        self.layers = nn.ModuleList(
            [
                ConvLSTMCell(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of ConvLSTM.

        Args:
            x: Input tensor of shape [B, T, C, H, W]

        Returns:
            Last hidden state tensor of shape [B, hidden_channels, H, W]
        """
        batch_size, seq_len, channels, height, width = x.size()

        # Initialize hidden and cell states
        h = [
            torch.zeros(
                batch_size, self.hidden_channels, height, width, device=x.device
            )
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(
                batch_size, self.hidden_channels, height, width, device=x.device
            )
            for _ in range(self.num_layers)
        ]

        # Process each time step
        for t in range(seq_len):
            # Get input for this time step
            input_t = x[:, t]

            # Process through each layer
            for i, layer in enumerate(self.layers):
                # For the first layer, use input_t; for subsequent layers, use previous hidden state
                if i == 0:
                    h[i], c[i] = layer(input_t, h[i], c[i])
                else:
                    h[i], c[i] = layer(h[i - 1], h[i], c[i])

        # Return the last hidden state
        return h[-1]
