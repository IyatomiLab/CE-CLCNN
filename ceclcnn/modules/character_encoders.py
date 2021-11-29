import torch.nn as nn
from allennlp.common.from_params import FromParams
from overrides import overrides
from torchtyping import TensorType


class CharacterEncoder(nn.Module, FromParams):
    def __init__(
        self,
        out_channels: int = 32,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        encode_dim: int = 128,
    ) -> None:
        super().__init__()
        self._encode_dim = encode_dim

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=conv_kernel_size
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
        )
        self.fc1 = nn.Linear(800, encode_dim)
        self.fc2 = nn.Linear(encode_dim, encode_dim)
        self.max_pool_2d = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.relu = nn.ReLU()

    def encode(
        self,
        x: TensorType["batch_size * char_len", 1, "char_w", "char_h"],  # type: ignore # NOQA: F821
    ) -> TensorType["batch_size * char_len", 1, "encode_dim"]:  # type: ignore # NOQA: F821

        x = self.max_pool_2d(self.relu(self.conv1(x)))
        x = self.max_pool_2d(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 800)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return x

    @overrides
    def forward(
        self, char_imgs: TensorType["batch_size", "char_len", "char_w", "char_h"]  # type: ignore # NOQA: F821
    ) -> TensorType["batch_size", "char_len", "encode_dim"]:  # type: ignore # NOQA: F821

        batch_size, char_len, char_w, char_h = char_imgs.size()

        char_imgs = char_imgs.view(batch_size * char_len, 1, char_w, char_h)
        char_imgs = self.encode(char_imgs)
        char_imgs = char_imgs.view(batch_size, char_len, self._encode_dim)

        return char_imgs
