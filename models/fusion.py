import torch
import torch.nn as nn

class MoleculeFusion_v3(nn.Module):
    def __init__(self, d_model, mode="self_attn"):
        super().__init__()
        self.mode = mode
        self.sm_proj = nn.Linear(d_model, d_model // 2)
        self.gnn_proj = nn.Linear(d_model, d_model // 2)

        if mode == "self_attn":
            self.attn = nn.MultiheadAttention(
                d_model, num_heads=2, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU()
            )

    def forward(self, sm, gnn):
        z = torch.cat([
            self.sm_proj(sm),
            self.gnn_proj(gnn)
        ], dim=-1)
        if self.mode == "self_attn":
            h = z.unsqueeze(1)
            out,_ = self.attn(h,h,h)
            return self.norm(out.squeeze(1) + z)
        return self.mlp(z)

class AttnFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, mol, prot, inter):
        x = torch.stack([mol, prot, inter], dim=1)
        out,_ = self.attn(x,x,x)
        return self.fc(out.mean(1)).squeeze(-1)
