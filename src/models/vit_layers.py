import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=96,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
    ):

        super().__init__()

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        # Learnable positional embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_patches + 1,
                embed_dim,
            )
        )

    def forward(self, x):

        batch_size = x.size(0)

        x = self.projection(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [cls, x],
            dim=1,
        )

        x = x + self.position_embedding

        return x