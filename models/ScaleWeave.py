"""ScaleWeave: a patch-Transformer encoder coupled with a Scale-Coupled
Hypergraph (SCH) module for multivariate time series forecasting."""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from einops import rearrange

from utils.RevIN import RevIN
from utils.embed import DataEmbedding_wo_time


class SCHLayer(nn.Module):
    """One layer of hypergraph message passing over patch nodes (P), variable-
    scale nodes (V) and scale nodes (G), via hyperedges E1/E2/E3 (and E4 for
    cross-scale fusion). Each layer aggregates nodes into hyperedges, then
    updates nodes from the hyperedges they belong to."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        use_cross_scale_g: bool = True,
        use_g_gate: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_cross_scale_g = bool(use_cross_scale_g)
        self.use_g_gate = bool(use_g_gate)

        self.edge_e1 = nn.Linear(d_model, d_model)
        self.edge_e2 = nn.Linear(d_model, d_model)
        self.edge_e3 = nn.Linear(d_model, d_model)

        if self.use_cross_scale_g:
            self.edge_e4 = nn.Linear(d_model, d_model)
            self.update_g_cross = nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.use_g_gate:
                self.g_gate_proj = nn.Linear(2 * d_model, d_model)
                nn.init.zeros_(self.g_gate_proj.weight)
                nn.init.constant_(self.g_gate_proj.bias, -2.0)

        self.update_p = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update_v = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update_g = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.ln_p = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_g = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        P_list: List[torch.Tensor],
        V: torch.Tensor,
        G: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        B, M, S, _ = V.shape

        P_means = torch.stack([P.mean(dim=2) for P in P_list], dim=2)
        E1 = self.edge_e1(0.5 * P_means + 0.5 * V)
        E2 = self.edge_e2(0.5 * V.mean(dim=1) + 0.5 * G)
        E3 = self.edge_e3(V.mean(dim=2))

        new_P_list = []
        for s_idx, P in enumerate(P_list):
            E1_s = E1[:, :, s_idx:s_idx + 1, :].expand(-1, -1, P.size(2), -1)
            update = self.update_p(torch.cat([P, E1_s], dim=-1))
            new_P_list.append(self.ln_p(P + self.dropout(update)))

        E2_exp = E2.unsqueeze(1).expand(-1, M, -1, -1)
        E3_exp = E3.unsqueeze(2).expand(-1, -1, S, -1)
        v_update = self.update_v(torch.cat([V, E1, E2_exp, E3_exp], dim=-1))
        new_V = self.ln_v(V + self.dropout(v_update))

        if self.use_cross_scale_g:
            S_ = G.size(1)
            G_bar = (G.sum(dim=1, keepdim=True) - G) / max(S_ - 1, 1)
            E4 = self.edge_e4(G.mean(dim=1, keepdim=True)).expand(-1, S_, -1)

            g_self = self.update_g(torch.cat([G, E2], dim=-1))
            g_cross = self.update_g_cross(torch.cat([G, G_bar, E4], dim=-1))

            if self.use_g_gate:
                alpha = torch.sigmoid(
                    self.g_gate_proj(torch.cat([G, G_bar], dim=-1))
                )
                g_mix = alpha * g_cross + (1.0 - alpha) * g_self
            else:
                g_mix = 0.5 * (g_self + g_cross)
            new_G = self.ln_g(G + self.dropout(g_mix))
        else:
            g_update = self.update_g(torch.cat([G, E2], dim=-1))
            new_G = self.ln_g(G + self.dropout(g_update))

        return new_P_list, new_V, new_G


class ScaleCoupledHypergraph(nn.Module):
    """Multi-scale hypergraph module. Maps a flattened token tensor
    (B*M, N_total, d_model) to the same shape; the V and G nodes are built
    per forward from patch means plus learnable scale biases."""

    def __init__(
        self,
        d_model: int,
        num_scales: int,
        hsg_layers: int,
        gate_init: float = 0.5,
        dropout: float = 0.1,
        use_cross_scale_g: bool = True,
        use_g_gate: bool = True,
        use_learned_v_pool: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.hsg_layers = hsg_layers
        self.use_learned_v_pool = use_learned_v_pool

        self.layers = nn.ModuleList([
            SCHLayer(
                d_model,
                dropout=dropout,
                use_cross_scale_g=use_cross_scale_g,
                use_g_gate=use_g_gate,
            )
            for _ in range(hsg_layers)
        ])

        self.v_scale_bias = nn.Parameter(torch.zeros(num_scales, d_model))
        self.g_scale_bias = nn.Parameter(torch.zeros(num_scales, d_model))
        self.v_init = nn.Parameter(torch.zeros(d_model))
        self.g_init = nn.Parameter(torch.zeros(d_model))

        if self.use_learned_v_pool:
            self.q_V = nn.Parameter(torch.randn(num_scales, d_model) * 0.02)

        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        B: int,
        M: int,
        patch_nums: Sequence[int],
    ) -> torch.Tensor:
        N_total = tokens.size(1)
        x = tokens.view(B, M, N_total, self.d_model)
        P_list = list(torch.split(x, list(patch_nums), dim=2))

        if self.use_learned_v_pool:
            V_list = []
            for s_idx, P in enumerate(P_list):
                q = self.q_V[s_idx]
                scores = torch.einsum('bmnd,d->bmn', P, q) / (self.d_model ** 0.5)
                attn = torch.softmax(scores, dim=-1)
                V_list.append(torch.einsum('bmn,bmnd->bmd', attn, P))
            V = torch.stack(V_list, dim=2)
        else:
            V = torch.stack([P.mean(dim=2) for P in P_list], dim=2)
        V = V + self.v_scale_bias.view(1, 1, self.num_scales, self.d_model) + self.v_init

        G = V.mean(dim=1)
        G = G + self.g_scale_bias.view(1, self.num_scales, self.d_model) + self.g_init

        for layer in self.layers:
            P_list, V, G = layer(P_list, V, G)

        updated = torch.cat(P_list, dim=2).view(B * M, N_total, self.d_model)
        gate = torch.sigmoid(self.gate)
        return self.ln(gate * updated + (1.0 - gate) * tokens)


class _EncoderOutput:
    """Container exposing ``.last_hidden_state``."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class ScaleWeaveEncoder(nn.Module):
    """Randomly-initialised Transformer encoder (PatchTST-size) that invokes
    the SCH module at the layer indices in ``args.sch_inject_at``."""

    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_layer = args.e_layers
        self.sch_inject_at = args.sch_inject_at

        ff_inner = getattr(args, 'transformer_ff_inner', None) or (args.d_model * 4)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.n_heads,
                dim_feedforward=ff_inner,
                dropout=args.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.n_layer)
        ])
        self.pos_embed = nn.Embedding(1024, args.d_model)
        self.ln_f = nn.LayerNorm(args.d_model)
        self.drop = nn.Dropout(args.dropout)

        self.sch = ScaleCoupledHypergraph(
            d_model=args.d_model,
            num_scales=args.num_scales,
            hsg_layers=args.hsg_layers,
            gate_init=args.gate_init_prg,
            dropout=args.dropout,
            use_cross_scale_g=bool(getattr(args, 'cross_scale_g', 1)),
            use_g_gate=bool(getattr(args, 'g_gate', 1)),
            use_learned_v_pool=bool(getattr(args, 'learned_hyperedge_weights', 0)),
        )

        self.batch_size: int = 0
        self.num_vars: int = 0
        self.patch_nums: Sequence[int] = ()

    def set_batch_info(
        self,
        batch_size: int,
        num_vars: int,
        patch_nums: Sequence[int],
    ) -> None:
        self.batch_size = batch_size
        self.num_vars = num_vars
        self.patch_nums = patch_nums

    def _apply_sch(self, x: torch.Tensor, full_batch: int) -> torch.Tensor:
        N_total = sum(self.patch_nums)
        reshaped = x.view(self.batch_size * self.num_vars, N_total, self.d_model)
        sch_out = self.sch(reshaped, self.batch_size, self.num_vars, self.patch_nums)
        return sch_out.view(full_batch, -1, self.d_model)

    def forward(self, inputs_embeds: torch.Tensor, **_kwargs) -> _EncoderOutput:
        full_batch, seq_len, _ = inputs_embeds.shape
        pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        x = self.drop(inputs_embeds + self.pos_embed(pos))

        for i, blk in enumerate(self.blocks):
            if i in self.sch_inject_at:
                x = self._apply_sch(x, full_batch)
            x = blk(x)

        if self.n_layer in self.sch_inject_at:
            x = self._apply_sch(x, full_batch)

        return _EncoderOutput(self.ln_f(x))


class FlattenHead(nn.Module):
    """Flatten + linear head. Input (B, M, d, N), output (B, M, H)."""

    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(self.flatten(x)))


class ScaleWeave(nn.Module):
    """ScaleWeave forecaster: PatchTST-size Transformer encoder plus the
    Scale-Coupled Hypergraph (SCH) module. Scale granularity is set by ``--scale_patch_sizes`` and
    ``--scale_strides``."""

    def __init__(self, configs, device: str = "cuda:0"):
        super().__init__()
        self.device = device

        self.scale_patch_sizes = [int(x) for x in configs.scale_patch_sizes.split()]
        self.scale_strides = [int(x) for x in configs.scale_strides.split()]
        assert len(self.scale_patch_sizes) == len(self.scale_strides), \
            "scale_patch_sizes and scale_strides must have the same length"
        self.num_scales = len(self.scale_patch_sizes)
        configs.num_scales = self.num_scales
        configs.hsg_layers = getattr(configs, 'hsg_layers', 2)

        self.patch_nums = [
            (configs.seq_len - p) // s + 2
            for p, s in zip(self.scale_patch_sizes, self.scale_strides)
        ]
        self.total_patches = sum(self.patch_nums)

        self.padding_layers = nn.ModuleList([
            nn.ReplicationPad1d((0, s)) for s in self.scale_strides
        ])

        self.encoder = ScaleWeaveEncoder(configs)

        self.scale_embeddings = nn.ModuleList([
            DataEmbedding_wo_time(
                p, configs.d_model, configs.embed, configs.freq, configs.in_dropout,
            )
            for p in self.scale_patch_sizes
        ])

        self.use_time_embed = bool(int(getattr(configs, 'use_time_embed', 0)))
        if self.use_time_embed:
            freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
            d_time = freq_map.get(str(configs.freq), 4)
            self.time_projs = nn.ModuleList([
                nn.Linear(d_time, configs.d_model, bias=False)
                for _ in self.scale_patch_sizes
            ])

        d_ff_cfg = getattr(configs, 'd_ff', configs.d_model)
        self.d_ff = min(d_ff_cfg, configs.d_model)
        self.head_nf = self.d_ff * self.total_patches

        self.output_projection = FlattenHead(
            configs.enc_in,
            self.head_nf,
            configs.pred_len,
            head_dropout=configs.out_dropout,
        )

        for layer in (self.encoder, *self.scale_embeddings, self.output_projection):
            layer.to(device=device)
            layer.train()

        self.revin_flag = int(configs.revin_flag)
        if self.revin_flag == 1:
            self.revin_layer = RevIN(1).to(device)

        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.c_out = getattr(configs, 'c_out', configs.enc_in)

    def _per_scale_time_embeddings(
        self,
        x_mark: torch.Tensor,
        B: int,
        M: int,
    ) -> List[torch.Tensor]:
        """Project per-patch time marks to ``d_model`` and broadcast to (B*M, N_s, d)."""
        xm = x_mark.permute(0, 2, 1).contiguous()
        outs: List[torch.Tensor] = []
        for s_idx in range(self.num_scales):
            xm_s = self.padding_layers[s_idx](xm)
            xm_s = xm_s.unfold(
                dimension=-1,
                size=self.scale_patch_sizes[s_idx],
                step=self.scale_strides[s_idx],
            ).mean(dim=-1)
            xm_s = xm_s.permute(0, 2, 1)
            t = self.time_projs[s_idx](xm_s)
            t = t.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, t.size(1), -1)
            outs.append(t)
        return outs

    def forward(self, x: torch.Tensor, ii=None, x_mark: torch.Tensor = None) -> torch.Tensor:
        B, _, M = x.shape

        if self.revin_flag == 1:
            x = self.revin_layer(x, 'norm')
            means = stdev = None
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdev

        x = x.permute(0, 2, 1).contiguous()

        per_scale_time_emb = (
            self._per_scale_time_embeddings(x_mark, B, M)
            if (self.use_time_embed and x_mark is not None)
            else None
        )

        all_tokens = []
        for s_idx in range(self.num_scales):
            x_s = self.padding_layers[s_idx](x)
            x_s = x_s.unfold(
                dimension=-1,
                size=self.scale_patch_sizes[s_idx],
                step=self.scale_strides[s_idx],
            )
            x_s = rearrange(x_s, 'b m n p -> (b m) n p')
            x_s = self.scale_embeddings[s_idx](x_s)
            if per_scale_time_emb is not None:
                x_s = x_s + per_scale_time_emb[s_idx]
            all_tokens.append(x_s)
        enc_in = torch.cat(all_tokens, dim=1)

        self.encoder.set_batch_info(B, M, self.patch_nums)
        enc_out = self.encoder(enc_in).last_hidden_state

        enc_out = enc_out[:, :, :self.d_ff]
        enc_out = enc_out.view(-1, M, enc_out.size(-2), enc_out.size(-1))
        enc_out = enc_out.permute(0, 1, 3, 2).contiguous()
        outputs = self.output_projection(enc_out).permute(0, 2, 1).contiguous()

        if self.revin_flag == 1:
            outputs = self.revin_layer(outputs, 'denorm')
        else:
            outputs = outputs * stdev + means
        return outputs
