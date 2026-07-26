import torch.nn as nn

from src.models.multihead_attention import MultiHeadSelfAttention


class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        mlp_ratio=4,
        dropout=0.1,
    ):

        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.net = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                embed_dim,
            ),

            nn.Dropout(dropout),
        )

    def forward(self, x):

        return self.net(x)

class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = FeedForward(
            embed_dim,
            mlp_ratio,
            dropout,
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x

    