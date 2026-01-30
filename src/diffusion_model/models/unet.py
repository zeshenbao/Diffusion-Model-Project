"""U-Net architecture used for the diffusion model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DROPOUT = 0.1


def calc_pixel_size(
    size: int,
    kernel: int,
    padding: int,
    stride: int,
    *,
    num_convs: int = 2,
    down: bool = True,
    layer_num: int = 0,
) -> int:
    """Compute spatial size after a stack of convolutions and optional pooling."""
    for _ in range(num_convs):
        size = int((size - kernel + 2 * padding) / stride + 1)
    if down and layer_num > 0:
        return size // 2
    if layer_num > 0 and not down:
        return size * 2
    return size


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tstep: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
        half_dim = self.dim // 2
        exp = -math.log(max_period) / (half_dim - 1)
        embeddings = torch.exp(exp * torch.linspace(0, 1, half_dim, device=tstep.device))
        embeddings = tstep.float() * embeddings
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=1)
        return embeddings


class FastAttention(nn.Module):
    def __init__(self, channels: int, n_head: int) -> None:
        super().__init__()
        self.channels = channels
        self.n_heads = n_head

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        chunk_size = query.shape[-1] // self.n_heads
        query_chunks = torch.split(query, chunk_size, dim=-1)
        key_chunks = torch.split(key, chunk_size, dim=-1)
        value_chunks = torch.split(value, chunk_size, dim=-1)
        attn_val = None
        for chunk_idx in range(self.n_heads):
            temp = nn.functional.scaled_dot_product_attention(
                query_chunks[chunk_idx],
                key_chunks[chunk_idx],
                value=value_chunks[chunk_idx],
                dropout_p=DROPOUT,
            )
            attn_val = temp if attn_val is None else torch.cat([attn_val, temp], dim=-1)
        return attn_val


class AttentionBlock(nn.Module):
    """Self-attention block used within the U-Net."""

    def __init__(
        self,
        channels: int,
        size: int,
        n_head: int = 4,
        fast_attention: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.size = int(size)
        if fast_attention:
            self.fast_attn = True
            self.attn = FastAttention(channels=channels, n_head=n_head)
        else:
            self.fast_attn = False
            self.attn = nn.MultiheadAttention(channels, n_head, batch_first=True, dropout=DROPOUT)
        self.group_norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.group_norm(x)
        normed = normed.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)

        if self.fast_attn:
            attn_val = self.attn(normed, normed, normed)
        else:
            attn_val, _ = self.attn(normed, normed, normed)

        attn_val = attn_val.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        return attn_val + x


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        t_emdim: int = 32,
        class_emdim: int = 32,
        kernel: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.time_layer = nn.Linear(t_emdim, out_channels)
        self.class_layer = nn.Linear(class_emdim, out_channels)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.SiLU(),
        )
        self.gn_in = (
            nn.GroupNorm(8, in_channels)
            if in_channels % 8 == 0
            else nn.GroupNorm(1, in_channels)
        )
        self.gn_out = (
            nn.GroupNorm(8, out_channels)
            if out_channels % 8 == 0
            else nn.GroupNorm(1, out_channels)
        )
        self.conv_layer_final = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel, padding=padding),
            nn.SiLU(),
        )
        self.activation = nn.SiLU()
        self.residual_conv_layer = None
        if in_channels != out_channels:
            self.residual_conv_layer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.time_layer(t)[:, :, None, None]
        c_proj = self.class_layer(c)[:, :, None, None] if c is not None else None

        res = x
        x = self.activation(self.gn_in(x))
        x = self.conv_layer(x)

        final_input = x + t + c_proj if c_proj is not None else x + t
        out = self.gn_out(self.conv_layer_final(final_input))

        if self.residual_conv_layer is not None:
            res = self.residual_conv_layer(res)
        return out + res


class UpSampleCNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        t_emdim: int = 32,
        class_emdim: int = 32,
        n_head: int = 4,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.conv_layer = CNNBlock(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            t_emdim=t_emdim,
            class_emdim=class_emdim,
        )
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), 2)
        self.cross_attention = nn.MultiheadAttention(
            out_channels,
            n_head,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.cross_attn_gn = nn.GroupNorm(8, out_channels)
        self.use_cross_attn = use_cross_attn

    def forward(
        self,
        x: torch.Tensor,
        res: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        _, channels, width, height = x.shape
        crop_w = (res.shape[2] - width) // 2
        crop_h = (res.shape[3] - height) // 2

        if crop_w > 0 and crop_h > 0:
            res = res[:, :, crop_w:-crop_w, crop_h:-crop_h]

        attn_val = None
        if self.use_cross_attn:
            normed_x = self.cross_attn_gn(x)
            normed_x = normed_x.reshape(-1, channels, height * width).swapaxes(1, 2)
            normed_res = self.cross_attn_gn(res)
            normed_res = normed_res.reshape(-1, channels, height * width).swapaxes(1, 2)
            attn_val, _ = self.cross_attention(normed_x, normed_res, normed_res)
            attn_val = attn_val.swapaxes(2, 1).reshape(-1, channels, width, height)

        x = torch.cat([x, res], dim=1)
        out = self.conv_layer(x, t, c)
        if attn_val is not None:
            out = out + attn_val
        return out


class MidBlock(nn.Module):
    def __init__(
        self,
        channel: int,
        pixel_size: int,
        t_emdim: int = 32,
        fast_attn: bool = False,
    ) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(8, channel) if channel % 8 == 0 else nn.GroupNorm(1, channel)
        self.act = nn.SiLU()
        self.block1 = CNNBlock(channel, channel, t_emdim=t_emdim)
        self.attn = AttentionBlock(channels=channel, size=pixel_size, fast_attention=fast_attn)
        self.block2 = CNNBlock(channel, channel, t_emdim=t_emdim)
        self.dropout = nn.Dropout(DROPOUT)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_emdim),
            nn.Linear(t_emdim, t_emdim),
            nn.SiLU(),
        )
        self.class_mlp = nn.Embedding(10, t_emdim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.time_mlp(t)
        if c is not None:
            c = self.class_mlp(c)
            c = c.view(-1, c.shape[-1])
        x = self.act(self.gn(x))
        x = self.block1(x, t, c)
        x = self.attn(x)
        x = self.dropout(x)
        x = self.block2(x, t, c)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channel_list: list[int],
        output_size: int,
        t_emdim: int,
        *,
        class_emdim: int = 32,
        pixel_dims: int = 32,
        fast_attn: bool = False,
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList()
        self.attn_block = nn.ModuleList()
        self.num_blocks = len(in_channel_list)
        self.dropout = nn.Dropout(DROPOUT)
        pixel_sizes = [pixel_dims]
        self.gn = nn.GroupNorm(1, in_channel_list[0])
        self.input_layer = nn.Conv2d(
            in_channels=in_channel_list[0],
            out_channels=in_channel_list[1],
            kernel_size=3,
            padding=1,
        )
        for i in range(self.num_blocks - 1):
            self.block.append(
                CNNBlock(
                    in_channel_list[i],
                    out_channels=in_channel_list[i + 1],
                    t_emdim=t_emdim,
                    class_emdim=class_emdim,
                )
            )
            self.block.append(
                CNNBlock(
                    in_channel_list[i + 1],
                    out_channels=in_channel_list[i + 1],
                    t_emdim=t_emdim,
                    class_emdim=class_emdim,
                )
            )
            new_size = calc_pixel_size(
                pixel_sizes[-1],
                kernel=3,
                padding=1,
                stride=1,
                layer_num=i,
            )
            self.attn_block.append(
                AttentionBlock(
                    in_channel_list[i + 1],
                    new_size,
                    fast_attention=fast_attn,
                )
            )
            pixel_sizes.append(new_size)

        self.block.append(
            CNNBlock(
                in_channels=in_channel_list[-1],
                out_channels=output_size,
                t_emdim=t_emdim,
                class_emdim=class_emdim,
            )
        )
        self.block.append(
            CNNBlock(
                in_channels=output_size,
                out_channels=output_size,
                t_emdim=t_emdim,
                class_emdim=class_emdim,
            )
        )
        self.pool = nn.MaxPool2d(2)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_emdim),
            nn.Linear(t_emdim, t_emdim),
            nn.SiLU(),
        )
        self.class_mlp = nn.Embedding(10, class_emdim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        residuals: list[torch.Tensor] = []
        x = self.gn(x)
        t = self.time_mlp(t)
        if c is not None:
            c = self.class_mlp(c)
            c = c.view(-1, c.shape[-1])

        for i in range(self.num_blocks):
            x = self.block[2 * i](x, t, c)
            x = self.dropout(x)
            if i < self.num_blocks - 1:
                x = self.attn_block[i](x)
            x = self.block[2 * i + 1](x, t, c)
            if i != self.num_blocks - 1:
                residuals.append(x)
                x = self.pool(x)

        return x, residuals


class Decoder(nn.Module):
    def __init__(
        self,
        in_channel_list: list[int],
        output_size: int,
        t_emdim: int,
        *,
        class_emdim: int = 32,
        pixel_dims: int = 4,
        use_cross_attn: bool = False,
        fast_attn: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.block = nn.ModuleList()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_emdim),
            nn.Linear(t_emdim, t_emdim),
            nn.SiLU(),
        )
        self.class_mlp = nn.Embedding(10, class_emdim)

        pixel_sizes = [pixel_dims]
        self.num_blocks = len(in_channel_list)
        self.dropout = nn.Dropout(DROPOUT)
        self.attn_block = nn.ModuleList()
        for i in range(len(in_channel_list) - 1):
            self.block.append(
                UpSampleCNNBlock(
                    in_channel_list[i],
                    out_channels=in_channel_list[i + 1],
                    t_emdim=t_emdim,
                    class_emdim=class_emdim,
                    use_cross_attn=use_cross_attn,
                )
            )
            self.block.append(
                CNNBlock(
                    in_channel_list[i + 1],
                    out_channels=in_channel_list[i + 1],
                    t_emdim=t_emdim,
                    class_emdim=class_emdim,
                )
            )
            new_size = calc_pixel_size(
                pixel_sizes[-1],
                kernel=3,
                padding=1,
                stride=1,
                down=False,
                layer_num=i + 1,
            )
            self.attn_block.append(
                AttentionBlock(
                    in_channel_list[i + 1],
                    new_size,
                    fast_attention=fast_attn,
                )
            )
            pixel_sizes.append(new_size)

        self.block.append(
            CNNBlock(
                in_channel_list[i + 1],
                out_channels=in_channel_list[i + 1],
                t_emdim=t_emdim,
                class_emdim=class_emdim,
            )
        )
        self.block.append(
            CNNBlock(
                in_channel_list[i + 1],
                out_channels=output_size,
                t_emdim=t_emdim,
                class_emdim=class_emdim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        residuals: list[torch.Tensor],
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_residuals = len(residuals)
        t = self.time_mlp(t)
        if c is not None:
            c = self.class_mlp(c)
            c = c.view(-1, c.shape[-1])

        for i in range(self.num_blocks):
            if i < self.num_blocks - 1:
                x = self.block[2 * i](x, residuals[num_residuals - 1 - i], t, c)
                x = self.attn_block[i](x)
                x = self.dropout(x)
            else:
                x = self.block[2 * i](x, t, c)

            x = self.block[2 * i + 1](x, t, c)

        return x


class UNet(nn.Module):
    """U-Net architecture for diffusion models."""

    def __init__(
        self,
        enc_channel_list: list[int],
        dec_channel_list: list[int],
        t_emdim: int,
        class_emdim: int = 32,
        *,
        pixel_dims_enc: int = 128,
        pixel_dims_dec: int | None = None,
        use_cross_attn: bool = False,
        use_mid_blocks: bool = True,
        fast_attn: bool = False,
    ) -> None:
        super().__init__()
        if pixel_dims_dec is None:
            pixel_dims_dec = pixel_dims_enc // 2 ** (len(enc_channel_list) - 2)

        self.encoder = Encoder(
            enc_channel_list[:-1],
            enc_channel_list[-1],
            t_emdim,
            class_emdim=class_emdim,
            pixel_dims=pixel_dims_enc,
            fast_attn=fast_attn,
        )
        self.decoder = Decoder(
            dec_channel_list[:-1],
            dec_channel_list[-1],
            t_emdim,
            class_emdim=class_emdim,
            pixel_dims=pixel_dims_dec,
            use_cross_attn=use_cross_attn,
            fast_attn=fast_attn,
        )
        self.use_mid_blocks = use_mid_blocks
        if use_mid_blocks:
            final_pixel_size = calc_pixel_size(
                self.encoder.attn_block[-1].size,
                3,
                1,
                stride=1,
                num_convs=1,
                layer_num=len(enc_channel_list),
            )
            self.mid_blocks = MidBlock(
                enc_channel_list[-1],
                final_pixel_size,
                t_emdim,
                fast_attn=fast_attn,
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_out, residuals = self.encoder(x, t, c)
        if self.use_mid_blocks:
            enc_out = self.mid_blocks(enc_out, t, c)
        return self.decoder(enc_out, residuals, t, c)


def save_only_model(model: nn.Module, path: str) -> None:
    """Save only the model weights."""
    torch.save(model.state_dict(), path)


def load_model(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
):
    """Load model and optional optimizer/scheduler states."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    loss = checkpoint.get("loss")
    epoch = checkpoint.get("epoch", 0)
    save_interval = checkpoint.get("save_interval")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler, loss, epoch, save_interval
