import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """简单的 concat 融合 + MLP。"""

    def __init__(self, dim1=512, dim2=512, mmhid=256, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim1 + dim2, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, 1),
        )

    def forward(self, ct_fea, pa_fea):
        return self.net(torch.cat([ct_fea, pa_fea], dim=1)).squeeze(-1)


class BilinearFusion(nn.Module):
    """Full PA 512D x CT 512D outer product with signed sqrt and L2 norm."""

    def __init__(self, dim1=512, dim2=512, in_dim=512, dropout_rate=0.0):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, dim1) if dim1 != in_dim else nn.Identity()
        self.proj2 = nn.Linear(in_dim, dim2) if dim2 != in_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(dim1 * dim2, 1)

    def forward(self, ct_fea, pa_fea):
        ct_fea = self.proj1(ct_fea)
        pa_fea = self.proj2(pa_fea)
        outer = torch.bmm(pa_fea.unsqueeze(2), ct_fea.unsqueeze(1))
        fused = outer.flatten(start_dim=1)
        fused = fused.sign() * torch.sqrt(fused.abs() + 1e-8)
        fused = F.normalize(fused, p=2, dim=1, eps=1e-8)
        return self.head(self.dropout(fused)).squeeze(-1)


class GatedFusion(nn.Module):
    """Learnable gate between CT and pathology features."""

    def __init__(self, dim1=512, dim2=512, mmhid=256, dropout_rate=0.3):
        super().__init__()
        self.pa_projection = (
            nn.Identity()
            if dim1 == dim2
            else nn.Sequential(
                nn.Linear(dim2, dim1),
                nn.LayerNorm(dim1),
                nn.ReLU(inplace=True),
            )
        )
        self.gate = nn.Sequential(
            nn.Linear(dim1 + dim2, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, dim1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(dim1, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, 1),
        )

    def forward(self, ct_fea, pa_fea):
        gate = self.gate(torch.cat([ct_fea, pa_fea], dim=1))
        pa_fea = self.pa_projection(pa_fea)
        fused = gate * ct_fea + (1 - gate) * pa_fea
        return self.head(fused).squeeze(-1)


class CrossAttnFusion(nn.Module):
    """Attention over two modality tokens followed by a survival head.

    A single query token attending to a single key/value token has a fixed
    attention weight of one and is therefore not meaningful cross-attention.
    Stacking both modalities as tokens lets the attention layer learn their
    self- and cross-modal interactions.
    """

    def __init__(self, dim=512, num_heads=4, mmhid=256, dropout_rate=0.2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.cross_ct = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, 1),
        )

    def forward(self, ct_fea, pa_fea):
        # [B, D] → two modality tokens [B, 2, D].
        tokens = torch.stack([ct_fea, pa_fea], dim=1)
        attended, _ = self.cross_ct(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        return self.head(tokens.flatten(start_dim=1)).squeeze(-1)


class FeatureSumFusion(nn.Module):
    is_new_fusion = True

    def __init__(self, ct_dim=512, pa_dim=512, dim=128, dropout_rate=0.0):
        super().__init__()
        if min(ct_dim, pa_dim, dim) <= 0 or not 0 <= dropout_rate < 1:
            raise ValueError("dimensions must be positive and dropout_rate must be in [0, 1)")
        self.ct_proj = nn.Sequential(nn.Linear(ct_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.pa_proj = nn.Sequential(nn.Linear(pa_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                  nn.Dropout(dropout_rate), nn.Linear(dim, 1))

    def forward(self, ct, pa):
        return self.head((self.ct_proj(ct)+self.pa_proj(pa))/2).squeeze(-1), {}


class FeatureProductFusion(nn.Module):
    is_new_fusion = True

    def __init__(self, ct_dim=512, pa_dim=512, dim=128, dropout_rate=0.0):
        super().__init__()
        if min(ct_dim, pa_dim, dim) <= 0 or not 0 <= dropout_rate < 1:
            raise ValueError("dimensions must be positive and dropout_rate must be in [0, 1)")
        self.ct_proj = nn.Sequential(nn.Linear(ct_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.pa_proj = nn.Sequential(nn.Linear(pa_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                  nn.Dropout(dropout_rate), nn.Linear(dim, 1))

    def forward(self, ct, pa):
        return self.head(self.ct_proj(ct)*self.pa_proj(pa)).squeeze(-1), {}


class LowRankFusion(nn.Module):
    is_new_fusion = True

    def __init__(self, ct_dim=512, pa_dim=512, dim=128, rank=4, dropout_rate=0.0):
        super().__init__()
        if (min(ct_dim, pa_dim, dim, rank) <= 0 or
                not 0 <= dropout_rate < 1):
            raise ValueError("dimensions/rank must be positive and dropout_rate must be in [0, 1)")
        self.ct_proj = nn.Sequential(nn.Linear(ct_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.pa_proj = nn.Sequential(nn.Linear(pa_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.ct_factor = nn.Parameter(torch.empty(rank, dim+1, dim))
        self.pa_factor = nn.Parameter(torch.empty(rank, dim+1, dim))
        nn.init.xavier_normal_(self.ct_factor)
        nn.init.xavier_normal_(self.pa_factor)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                  nn.Dropout(dropout_rate), nn.Linear(dim, 1))

    def forward(self, ct, pa):
        x, y = self.ct_proj(ct), self.pa_proj(pa)
        x = torch.cat([x, torch.ones_like(x[:, :1])], -1)
        y = torch.cat([y, torch.ones_like(y[:, :1])], -1)
        z = (torch.einsum('bi,rio->bro', x, self.ct_factor)*
             torch.einsum('bi,rio->bro', y, self.pa_factor)).sum(1)
        return self.head(z).squeeze(-1), {}


class CompactBilinearFusion(nn.Module):
    is_new_fusion = True

    def __init__(self, ct_dim=512, pa_dim=512, dim=128, sketch_dim=512,
                 dropout_rate=0.0):
        super().__init__()
        if min(ct_dim, pa_dim, dim, sketch_dim) <= 0 or not 0 <= dropout_rate < 1:
            raise ValueError("invalid feature/sketch dimensions or dropout_rate")
        self.sketch_dim = sketch_dim
        self.ct_proj = nn.Sequential(nn.Linear(ct_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.pa_proj = nn.Sequential(nn.Linear(pa_dim, dim), nn.LayerNorm(dim), nn.GELU())
        for modality in ('ct', 'pa'):
            self.register_buffer(f'hash_{modality}', torch.randint(
                sketch_dim, (dim+1,)))
            self.register_buffer(f'sign_{modality}', torch.randint(
                2, (dim+1,)).float()*2-1)
        self.head = nn.Sequential(nn.Linear(sketch_dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                  nn.Dropout(dropout_rate), nn.Linear(dim, 1))

    def tensor_sketch(self, x, y):
        def sketch(values, indices, signs):
            return values.new_zeros(values.size(0), self.sketch_dim).scatter_add(
                1, indices.expand(values.size(0), -1), values*signs)
        a, b = sketch(x, self.hash_ct, self.sign_ct), sketch(y, self.hash_pa, self.sign_pa)
        return torch.fft.irfft(torch.fft.rfft(a)*torch.fft.rfft(b), n=self.sketch_dim)

    def forward(self, ct, pa):
        x, y = self.ct_proj(ct), self.pa_proj(pa)
        x = torch.cat([x, torch.ones_like(x[:, :1])], -1)
        y = torch.cat([y, torch.ones_like(y[:, :1])], -1)
        z = self.tensor_sketch(x, y)
        z = F.normalize(z/(z.abs()+1e-6).sqrt(), dim=-1)
        return self.head(z).squeeze(-1), {}


class FiLMPatchFusion(nn.Module):
    """CT-conditioned patch channels, nonlinear processing, then MIL pooling."""
    needs_patches = True

    def __init__(self, ct_dim=512, patch_dim=1024, dim=128, dropout_rate=0.0):
        super().__init__()
        if min(ct_dim, patch_dim, dim) <= 0 or not 0 <= dropout_rate < 1:
            raise ValueError("dimensions must be positive and dropout_rate must be in [0, 1)")
        d = dim
        self.ct_proj = nn.Sequential(nn.Linear(ct_dim, d), nn.LayerNorm(d), nn.GELU())
        self.path_proj = nn.Sequential(nn.Linear(patch_dim, d), nn.LayerNorm(d), nn.GELU())
        self.modulation = nn.Linear(d, 2*d)
        # Start close to identity without blocking gradient into the conditioner.
        nn.init.normal_(self.modulation.weight, std=1e-3)
        nn.init.zeros_(self.modulation.bias)
        self.attention = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.head = nn.Sequential(nn.Linear(2*d, d), nn.LayerNorm(d), nn.GELU(),
                                  nn.Dropout(dropout_rate), nn.Linear(d, 1))
        self.is_new_fusion = True

    def forward(self, ct, pa, *, patches):
        if patches.ndim != 3 or patches.size(0) != ct.size(0):
            raise ValueError("patches must have shape [B,N,D] with the same batch as CT")
        x = self.ct_proj(ct)
        gamma, beta = self.modulation(x).chunk(2, -1)
        h = F.gelu((1+gamma.unsqueeze(1))*self.path_proj(patches)+beta.unsqueeze(1))
        weights = self.attention(h).softmax(dim=1)
        pooled = (weights*h).sum(1)
        return self.head(torch.cat([x, pooled], -1)).squeeze(-1), {}


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2*dim, dim))

    def forward(self, query, context):
        query = self.norm1(query+self.attention(query, context, context,
                                               need_weights=False)[0])
        return self.norm2(query+self.ffn(query))


def initialize_token_fusion(module, ct_dim, patch_dim, dim, num_path_tokens, ct_grid,
                            output_width, dropout_rate):
    if (min(ct_dim, patch_dim, dim, num_path_tokens, ct_grid) <= 0 or
            not 0 <= dropout_rate < 1 or int(ct_grid) != ct_grid or
            int(num_path_tokens) != num_path_tokens):
        raise ValueError("token fusion dimensions and token counts must be positive")
    module.ct_grid = ct_grid
    module.ct_proj = nn.Sequential(nn.Linear(ct_dim, dim), nn.LayerNorm(dim), nn.GELU())
    module.path_proj = nn.Sequential(nn.Linear(patch_dim, dim), nn.LayerNorm(dim), nn.GELU())
    module.path_queries = nn.Parameter(torch.randn(num_path_tokens, dim)/math.sqrt(dim))
    module.modality = nn.Parameter(torch.randn(2, dim)*.02)
    axis = torch.linspace(-1, 1, ct_grid)
    coordinates = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    module.register_buffer('ct_coordinates', coordinates)
    module.position = nn.Linear(3, dim)
    module.head = nn.Sequential(nn.Linear(output_width, dim), nn.LayerNorm(dim), nn.GELU(),
                                nn.Dropout(dropout_rate), nn.Linear(dim, 1))
    module.is_new_fusion = True
    module.needs_patches = True
    module.needs_spatial = True


def tokenize_fusion(module, spatial, patches):
    if spatial.ndim != 5 or patches.ndim != 3 or spatial.size(0) != patches.size(0):
        raise ValueError("expected CT [B,C,D,H,W] and patches [B,N,D]")
    if patches.size(1) == 0:
        raise ValueError("at least one pathology patch is required")
    if min(spatial.shape[-3:]) < module.ct_grid:
        raise ValueError('CT spatial map is smaller than the requested token grid')
    ct = F.adaptive_avg_pool3d(spatial, module.ct_grid).flatten(2).transpose(1, 2)
    ct = module.ct_proj(ct)+module.position(module.ct_coordinates)+module.modality[0]
    patches = module.path_proj(patches)
    scores = torch.einsum('kd,bnd->bkn', module.path_queries, patches)/math.sqrt(patches.size(-1))
    pa = torch.bmm(scores.softmax(-1), patches)+module.modality[1]
    return ct, pa


class TokenCoAttentionFusion(nn.Module):
    def __init__(self, ct_dim=512, patch_dim=1024, dim=128, num_heads=4,
                 num_path_tokens=16, ct_grid=2, num_layers=2, dropout_rate=0.0):
        super().__init__()
        if num_heads <= 0 or dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError('num_layers must be positive')
        initialize_token_fusion(self, ct_dim, patch_dim, dim, num_path_tokens,
                                ct_grid, 2*dim, dropout_rate)
        self.ct_blocks = nn.ModuleList([CrossAttentionBlock(dim, num_heads, dropout_rate)
                                        for _ in range(num_layers)])
        self.pa_blocks = nn.ModuleList([CrossAttentionBlock(dim, num_heads, dropout_rate)
                                        for _ in range(num_layers)])

    def forward(self, ct, pa, *, patches, spatial):
        x, y = tokenize_fusion(self, spatial, patches)
        for ct_block, pa_block in zip(self.ct_blocks, self.pa_blocks):
            old_x, old_y = x, y
            x, y = ct_block(old_x, old_y), pa_block(old_y, old_x)
        return self.head(torch.cat([x.mean(1), y.mean(1)], -1)).squeeze(-1), {}


class BottleneckFusion(nn.Module):
    def __init__(self, ct_dim=512, patch_dim=1024, dim=128, num_heads=4,
                 num_path_tokens=16, ct_grid=2, num_latents=4, num_layers=2,
                 dropout_rate=0.0):
        super().__init__()
        if num_heads <= 0 or dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        if (not isinstance(num_latents, int) or not isinstance(num_layers, int) or
                num_latents <= 0 or num_layers <= 0):
            raise ValueError('num_latents and num_layers must be positive')
        initialize_token_fusion(self, ct_dim, patch_dim, dim, num_path_tokens,
                                ct_grid, 3*dim, dropout_rate)
        self.latents = nn.Parameter(torch.randn(num_latents, dim)*.02)
        self.ct_blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            dim, num_heads, dim_feedforward=2*dim, dropout=dropout_rate, batch_first=True)
            for _ in range(num_layers)])
        self.pa_blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            dim, num_heads, dim_feedforward=2*dim, dropout=dropout_rate, batch_first=True)
            for _ in range(num_layers)])

    def forward(self, ct, pa, *, patches, spatial):
        x, y = tokenize_fusion(self, spatial, patches)
        shared = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        for ct_block, pa_block in zip(self.ct_blocks, self.pa_blocks):
            ct_count, pa_count = x.size(1), y.size(1)
            ct_output = ct_block(torch.cat([x, shared], 1))
            pa_output = pa_block(torch.cat([y, shared], 1))
            x, y = ct_output[:, :ct_count], pa_output[:, :pa_count]
            ct_shared = ct_output[:, ct_count:]
            pa_shared = pa_output[:, pa_count:]
            shared = (ct_shared + pa_shared) / 2
        z = torch.cat([x.mean(1), y.mean(1), shared.mean(1)], -1)
        return self.head(z).squeeze(-1), {}


class QueryFusion(nn.Module):
    def __init__(self, ct_dim=512, patch_dim=1024, dim=128, num_heads=4,
                 num_path_tokens=16, ct_grid=2, num_latents=4, num_layers=2,
                 dropout_rate=0.0):
        super().__init__()
        if num_heads <= 0 or dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        if (not isinstance(num_latents, int) or not isinstance(num_layers, int) or
                num_latents <= 0 or num_layers <= 0):
            raise ValueError('num_latents and num_layers must be positive')
        initialize_token_fusion(self, ct_dim, patch_dim, dim, num_path_tokens,
                                ct_grid, 3*dim, dropout_rate)
        self.latents = nn.Parameter(torch.randn(num_latents, dim)*.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(dim, num_heads, dropout_rate)
                                     for _ in range(num_layers)])

    def forward(self, ct, pa, *, patches, spatial):
        x, y = tokenize_fusion(self, spatial, patches)
        query = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        context = torch.cat([x, y], 1)
        for block in self.blocks:
            query = block(query, context)
        z = torch.cat([x.mean(1), y.mean(1), query.mean(1)], -1)
        return self.head(z).squeeze(-1), {}


class GraphFusion(nn.Module):
    def __init__(self, ct_dim=512, patch_dim=1024, dim=128,
                 num_path_tokens=16, ct_grid=2, num_layers=2,
                 graph_neighbors=3, dropout_rate=0.0):
        super().__init__()
        if (not isinstance(num_layers, int) or not isinstance(graph_neighbors, int) or
                num_layers <= 0 or graph_neighbors <= 0):
            raise ValueError('num_layers and graph_neighbors must be positive')
        self.graph_neighbors = graph_neighbors
        initialize_token_fusion(self, ct_dim, patch_dim, dim, num_path_tokens,
                                ct_grid, 2*dim, dropout_rate)
        self.message = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

    def graph(self, ct, pa):
        tokens = torch.cat([ct, pa], 1)
        n, m = ct.size(1), pa.size(1)
        for message, norm in zip(self.message, self.norms):
            similarity = F.normalize(tokens, dim=-1) @ F.normalize(tokens, dim=-1).transpose(1, 2)
            mask = torch.zeros_like(similarity, dtype=torch.bool)
            for start, size in ((0, n), (n, m)):
                k = min(self.graph_neighbors, size)
                index = similarity[:, :, start:start+size].topk(k, dim=-1).indices+start
                mask.scatter_(2, index, True)
            adjacency = similarity.masked_fill(~mask, float('-inf')).softmax(-1)
            tokens = norm(tokens+F.gelu(message(adjacency@tokens)))
        return tokens[:, :n], tokens[:, n:]

    def forward(self, ct, pa, *, patches, spatial):
        x, y = self.graph(*tokenize_fusion(self, spatial, patches))
        return self.head(torch.cat([x.mean(1), y.mean(1)], -1)).squeeze(-1), {}


class OptimalTransportFusion(nn.Module):
    def __init__(self, ct_dim=512, patch_dim=1024, dim=128,
                 num_path_tokens=16, ct_grid=2, ot_epsilon=0.1,
                 ot_iterations=100, dropout_rate=0.0):
        super().__init__()
        if (not math.isfinite(ot_epsilon) or ot_epsilon <= 0 or
                not isinstance(ot_iterations, int) or ot_iterations <= 0):
            raise ValueError('ot_epsilon and ot_iterations must be positive')
        self.ot_epsilon, self.ot_iterations = ot_epsilon, ot_iterations
        initialize_token_fusion(self, ct_dim, patch_dim, dim, num_path_tokens,
                                ct_grid, 2*dim, dropout_rate)
        self.ct_norm, self.pa_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def transport(self, ct, pa):
        cost = torch.cdist(F.normalize(ct, dim=-1), F.normalize(pa, dim=-1)).square()
        kernel = -cost/self.ot_epsilon
        u, v = torch.zeros_like(cost[:, :, 0]), torch.zeros_like(cost[:, 0, :])
        for _ in range(self.ot_iterations):
            u = -math.log(ct.size(1))-torch.logsumexp(kernel+v.unsqueeze(1), dim=2)
            v = -math.log(pa.size(1))-torch.logsumexp(kernel+u.unsqueeze(2), dim=1)
        return (kernel+u.unsqueeze(2)+v.unsqueeze(1)).exp()

    def forward(self, ct, pa, *, patches, spatial):
        x, y = tokenize_fusion(self, spatial, patches)
        plan = self.transport(x, y)
        old_x, old_y = x, y
        x = self.ct_norm(old_x+torch.bmm(plan, old_y)*old_x.size(1))
        y = self.pa_norm(old_y+torch.bmm(plan.transpose(1, 2), old_x)*old_y.size(1))
        return self.head(torch.cat([x.mean(1), y.mean(1)], -1)).squeeze(-1), {}


def combine_payloads(payloads):
    return {key: torch.cat([item[key] for item in payloads], 0)
            for key in payloads[0]}


class SharedPrivateFusion(nn.Module):
    is_new_fusion = True
    uses_auxiliary = True

    def __init__(self, ct_dim=512, pa_dim=512, dim=128, dropout_rate=0.0):
        super().__init__()
        if min(ct_dim, pa_dim, dim) <= 0 or not 0 <= dropout_rate < 1:
            raise ValueError("dimensions must be positive and dropout_rate must be in [0, 1)")
        self.ct_proj = nn.Sequential(nn.Linear(ct_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.pa_proj = nn.Sequential(nn.Linear(pa_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.shared = nn.Linear(dim, dim)
        self.private_ct, self.private_pa = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.reconstruct_ct = nn.Linear(2*dim, dim)
        self.reconstruct_pa = nn.Linear(2*dim, dim)
        self.head = nn.Sequential(nn.Linear(4*dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                  nn.Dropout(dropout_rate), nn.Linear(dim, 1))

    def forward(self, ct, pa):
        x, y = self.ct_proj(ct), self.pa_proj(pa)
        sx, sy = self.shared(x), self.shared(y)
        px, py = self.private_ct(x), self.private_pa(y)
        rx = self.reconstruct_ct(torch.cat([sx, px], -1))
        ry = self.reconstruct_pa(torch.cat([sy, py], -1))
        risk = self.head(torch.cat([sx, sy, px, py], -1)).squeeze(-1)
        return risk, dict(ct=x, pa=y, sx=sx, sy=sy, px=px, py=py, rx=rx, ry=ry)

    def auxiliary_loss(self, payloads):
        data = combine_payloads(payloads)
        sx, sy, px, py = (data[key] for key in ('sx', 'sy', 'px', 'py'))
        reconstruction = (F.mse_loss(data['rx'], data['ct'].detach())+
                          F.mse_loss(data['ry'], data['pa'].detach()))
        alignment = F.mse_loss(sx.mean(0), sy.mean(0))
        separation = sum((F.normalize(a, dim=-1)*F.normalize(b, dim=-1))
                         .sum(-1).square().mean()
                         for a, b in ((sx, px), (sy, py), (px, py)))
        return (reconstruction+alignment+separation)/3


class ContrastiveFusion(nn.Module):
    is_new_fusion = True
    uses_auxiliary = True

    def __init__(self, ct_dim=512, pa_dim=512, dim=128, temperature=0.1,
                 dropout_rate=0.0):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if min(ct_dim, pa_dim, dim) <= 0 or not 0 <= dropout_rate < 1:
            raise ValueError("dimensions must be positive and dropout_rate must be in [0, 1)")
        self.temperature = temperature
        self.ct_proj = nn.Sequential(nn.Linear(ct_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.pa_proj = nn.Sequential(nn.Linear(pa_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2*dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                  nn.Dropout(dropout_rate), nn.Linear(dim, 1))

    def forward(self, ct, pa):
        x, y = self.ct_proj(ct), self.pa_proj(pa)
        return self.head(torch.cat([x, y], -1)).squeeze(-1), dict(ct=x, pa=y)

    def auxiliary_loss(self, payloads):
        data = combine_payloads(payloads)
        x, y = data['ct'], data['pa']
        if x.size(0) < 2:
            return (x.sum()+y.sum())*0.
        logits = F.normalize(x, dim=-1)@F.normalize(y, dim=-1).T/self.temperature
        target = torch.arange(x.size(0), device=x.device)
        return (F.cross_entropy(logits, target)+F.cross_entropy(logits.T, target))/2


class DCCAFusion(nn.Module):
    is_new_fusion = True
    uses_auxiliary = True

    def __init__(self, ct_dim=512, pa_dim=512, dim=128,
                 covariance_ridge=1e-3, dropout_rate=0.0):
        super().__init__()
        if not math.isfinite(covariance_ridge) or covariance_ridge <= 0:
            raise ValueError('covariance_ridge must be finite and positive')
        if min(ct_dim, pa_dim, dim) <= 0 or not 0 <= dropout_rate < 1:
            raise ValueError("dimensions must be positive and dropout_rate must be in [0, 1)")
        self.covariance_ridge = covariance_ridge
        self.ct_proj = nn.Sequential(nn.Linear(ct_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.pa_proj = nn.Sequential(nn.Linear(pa_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2*dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                  nn.Dropout(dropout_rate), nn.Linear(dim, 1))

    def forward(self, ct, pa):
        x, y = self.ct_proj(ct), self.pa_proj(pa)
        return self.head(torch.cat([x, y], -1)).squeeze(-1), dict(ct=x, pa=y)

    def auxiliary_loss(self, payloads):
        data = combine_payloads(payloads)
        x, y = data['ct'], data['pa']
        if x.size(0) < 2:
            return (x.sum()+y.sum())*0.
        a, b = x.double()-x.double().mean(0), y.double()-y.double().mean(0)
        divisor = x.size(0)-1
        eye = torch.eye(a.size(1), device=x.device, dtype=torch.float64)
        left = torch.linalg.cholesky(a.T@a/divisor+self.covariance_ridge*eye)
        right = torch.linalg.cholesky(b.T@b/divisor+self.covariance_ridge*eye)
        cross = torch.linalg.solve_triangular(left, a.T@b/divisor, upper=False)
        cross = torch.linalg.solve_triangular(right, cross.T, upper=False).T
        count = min(x.size(0)-1, a.size(1))
        return -torch.linalg.svdvals(cross)[:count].mean().to(x.dtype)
