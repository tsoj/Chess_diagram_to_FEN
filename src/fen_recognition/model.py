from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as models

from src import consts
from src.games import get_game

TILE_EMBED_DIM = 512
FULL_EMBED_DIM = 512


@dataclass
class AblationConfig:
    use_transformer: bool = True
    use_dense_instead: bool = False
    use_full_img: bool = True
    use_pos_embed: bool = True
    use_tiles: bool = True


def get_tile_model():
    result = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    result.classifier = nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=768, out_features=TILE_EMBED_DIM),
        nn.ReLU(),
    )
    return result


def get_full_img_model():
    result = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    result.classifier = nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=768, out_features=FULL_EMBED_DIM),
        nn.ReLU(),
    )
    return result


class BoardRec(nn.Module):
    def __init__(
        self,
        game: str,
        tile_size: int = consts.DEFAULT_TILE_SIZE,
        dropout: float = 0.1,
        ablation_config: AblationConfig | None = None,
    ):
        super().__init__()
        self.game = get_game(game)
        self.tile_size = tile_size
        self.board_h, self.board_w = consts.board_pixel_size(self.game, tile_size)
        self.num_squares = self.game.num_squares
        self.out_channels = len(self.game.piece_symbols) + 1
        self.ablation = ablation_config or AblationConfig()

        if self.ablation.use_tiles:
            self.tile = get_tile_model()

        if self.ablation.use_full_img:
            self.full = get_full_img_model()

        if self.ablation.use_pos_embed and self.ablation.use_tiles:
            self.tile_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_squares, TILE_EMBED_DIM)
            )
            nn.init.trunc_normal_(self.tile_pos_embed, std=0.02)

        # Projection from concatenated features to attention dimension
        self.embed_dim = 512
        pre_attn_in = (TILE_EMBED_DIM if self.ablation.use_tiles else 0) + (FULL_EMBED_DIM if self.ablation.use_full_img else 0)
        self.pre_attn_dense = nn.Sequential(
            nn.Linear(pre_attn_in, self.embed_dim),
            nn.ReLU(),
        )

        if self.ablation.use_transformer:
            # Transformer block: Attention → Add&Norm → FFN → Add&Norm
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(self.embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim * 4, self.embed_dim),
                nn.Dropout(dropout),
            )
            self.ffn_norm = nn.LayerNorm(self.embed_dim)
        elif self.ablation.use_dense_instead:
            # Per-square MLP with ~4M params to match transformer block
            self.dense_replacement = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim * 4, self.embed_dim),
                nn.Dropout(dropout),
            )

        # Classification head
        self.dense = nn.Sequential(
            nn.Linear(self.embed_dim, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.out_channels),
        )

    def forward(self, img):
        batch_size, ch, h, w = img.shape

        assert h == self.board_h
        assert w == self.board_w
        assert ch == 3

        parts = []

        if self.ablation.use_tiles:
            x = img
            x = x.unfold(2, self.tile_size, self.tile_size)
            x = x.unfold(3, self.tile_size, self.tile_size)
            x = x.permute(0, 2, 3, 1, 4, 5)

            x = x.reshape(
                batch_size * self.game.board_rows * self.game.board_cols,
                ch,
                self.tile_size,
                self.tile_size,
            )

            x = self.tile(x)
            x = x.reshape(batch_size, self.num_squares, -1)

            if self.ablation.use_pos_embed:
                x = x + self.tile_pos_embed

            parts.append(x)

        if self.ablation.use_full_img:
            z = self.full(img)
            z = z.reshape(batch_size, 1, -1)
            z = z.expand(-1, self.num_squares, -1)
            parts.append(z)

        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        # Project to attention dimension
        x = self.pre_attn_dense(x)

        if self.ablation.use_transformer:
            attn_out, _ = self.attention(x, x, x)
            x = self.attn_norm(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.ffn_norm(x + ffn_out)
        elif self.ablation.use_dense_instead:
            x = self.dense_replacement(x)

        # Per-square classification
        x = x.reshape(batch_size * self.num_squares, -1)
        x = self.dense(x)
        x = x.reshape(batch_size, self.num_squares, self.out_channels)

        return x

if __name__ == "__main__":
    model = BoardRec(game="xiangqi")
    print("CUDA Available:", torch.cuda.is_available())
