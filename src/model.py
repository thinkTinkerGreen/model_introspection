import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, C=192, nh=6, T=128, attn_impl="manual"):
        super().__init__()
        assert C % nh == 0
        self.C = C
        self.nh = nh
        self.hs = C // nh
        self.T = T
        self.attn_impl = attn_impl

        # Projections
        self.c_attn = nn.Linear(C, 3 * C, bias=False)
        self.c_proj = nn.Linear(C, C, bias=False)

        # Causal mask for manual attention
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        )
        self.captured_intermediates = {}

    def forward(self, x, capture_intermediates=False):
        B, T, C = x.size()

        # qkv projection: (B, T, 3*C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.C, dim=2)

        # Reshape to (B, nh, T, hs)
        k = k.view(B, T, self.nh, self.hs).transpose(1, 2)
        q = q.view(B, T, self.nh, self.hs).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hs).transpose(1, 2)

        if capture_intermediates:
            self.captured_intermediates["qkv"] = qkv
            self.captured_intermediates["q"] = q
            self.captured_intermediates["k"] = k
            self.captured_intermediates["v"] = v

        if self.attn_impl == "manual":
            # att_scores: (B, nh, T, T)
            att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hs))
            att_scores = att_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att_probs = F.softmax(att_scores, dim=-1)
            # att_out_per_head: (B, nh, T, hs)
            y = att_probs @ v
            if capture_intermediates:
                self.captured_intermediates["att_scores"] = att_scores
                self.captured_intermediates["att_probs"] = att_probs
                self.captured_intermediates["att_out_per_head"] = y
        elif self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            if capture_intermediates:
                self.captured_intermediates["att_out_per_head"] = y
        else:
            raise ValueError(f"Unknown attn_impl: {self.attn_impl}")

        # att_out_merged: (B, T, C)
        att_out_merged = y.transpose(1, 2).contiguous().view(B, T, C)
        if capture_intermediates:
            self.captured_intermediates["att_out_merged"] = att_out_merged

        out = self.c_proj(att_out_merged)
        return out


class MLP(nn.Module):
    def __init__(self, C=192):
        super().__init__()
        self.c_fc = nn.Linear(C, 4 * C, bias=False)
        self.c_proj = nn.Linear(4 * C, C, bias=False)
        self.captured_intermediates = {}

    def forward(self, x, capture_intermediates=False):
        hidden = self.c_fc(x)
        act = F.gelu(hidden, approximate="none")
        if capture_intermediates:
            self.captured_intermediates["mlp_hidden"] = act
        out = self.c_proj(act)
        return out


class Block(nn.Module):
    def __init__(self, C=192, nh=6, T=128, attn_impl="manual"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(C, bias=False)
        self.attn = CausalSelfAttention(C, nh, T, attn_impl=attn_impl)
        self.ln_2 = nn.LayerNorm(C, bias=False)
        self.mlp = MLP(C)

    def forward(self, x, capture_intermediates=False):
        x = x + self.attn(self.ln_1(x), capture_intermediates=capture_intermediates)
        x = x + self.mlp(self.ln_2(x), capture_intermediates=capture_intermediates)
        return x


class GPT(nn.Module):
    def __init__(self, V=65, C=192, L=6, nh=6, T=128, attn_impl="manual"):
        super().__init__()
        self.V = V
        self.C = C
        self.L = L
        self.nh = nh
        self.hs = C // nh
        self.T = T
        self.attn_impl = attn_impl

        self.wte = nn.Embedding(V, C)
        self.wpe = nn.Embedding(T, C)
        self.blocks = nn.ModuleList([
            Block(C=C, nh=nh, T=T, attn_impl=attn_impl) for _ in range(L)
        ])
        self.ln_f = nn.LayerNorm(C, bias=False)
        self.lm_head = nn.Linear(C, V, bias=False)

        # Weight tying: lm_head.weight is wte.weight
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * L))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def set_attn_impl(self, attn_impl):
        self.attn_impl = attn_impl
        for b in self.blocks:
            b.attn.attn_impl = attn_impl

    def get_parameter_counts(self):
        """
        N_total: all unique parameters
        N_wpe: position embedding parameters (T * C)
        N: N_total - N_wpe
        """
        # Unique parameters accounting for weight tying
        unique_params = set()
        n_total = 0
        for p in self.parameters():
            if id(p) not in unique_params:
                unique_params.add(id(p))
                n_total += p.numel()
        n_wpe = self.wpe.weight.numel()
        n = n_total - n_wpe
        return n_total, n_wpe, n

    def forward(self, idx, targets=None, capture_intermediates=False):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx)       # (B, T, C)
        pos_emb = self.wpe(pos)       # (T, C)
        x = tok_emb + pos_emb         # (B, T, C)

        for b in self.blocks:
            x = b(x, capture_intermediates=capture_intermediates)

        x = self.ln_f(x)              # (B, T, C)
        logits = self.lm_head(x)      # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.V), targets.view(-1), ignore_index=-100)

        return logits, loss
