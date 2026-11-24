import torch
from torch import nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(96),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Linear(96, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.adaptive_avg_pool3d(self.net(x), 1).flatten(1))

class AlzheimerUnifiedNet(nn.Module):
    def __init__(self, task: str, route: str, num_classes: int = 3, fusion: str = "gated"):
        super().__init__()
        self.task = task
        self.route = route
        self.fusion = fusion
        self.use_mri = route in ("FULL", "MRI_PET", "MRI_TABLE", "MRI_ONLY")
        self.use_pet = route in ("FULL", "MRI_PET")
        self.use_table = route in ("FULL", "MRI_TABLE", "TABLE_ONLY")
        self.mri_encoder = ConvEncoder() if self.use_mri else None
        self.pet_encoder = ConvEncoder() if self.use_pet else None
        self.table_encoder = (
            nn.Sequential(nn.Linear(12, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU())
            if self.use_table
            else None
        )
        self.modality = nn.Parameter(torch.randn(3, 128) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))
        self.cross_query = nn.Parameter(torch.randn(1, 1, 128) * 0.02)
        self.cross_attention = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
        self.qc_projection = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 128))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.2), nn.Linear(64, 3 if task == "classification" else 1))

    def forward(self, mri: torch.Tensor, pet: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        tokens = []
        present = []
        if self.use_mri:
            tokens.append(self.mri_encoder(mri) + self.modality[0])
            present.append(1.0)
        else:
            tokens.append(torch.zeros((mri.shape[0], 128), device=mri.device))
            present.append(0.0)
        if self.use_pet:
            tokens.append(self.pet_encoder(pet) + self.modality[1])
            present.append(1.0)
        else:
            tokens.append(torch.zeros((mri.shape[0], 128), device=mri.device))
            present.append(0.0)
        if self.use_table:
            tokens.append(self.table_encoder(table) + self.modality[2])
            present.append(1.0)
        else:
            tokens.append(torch.zeros((mri.shape[0], 128), device=mri.device))
            present.append(0.0)
        stack = torch.stack(tokens, dim=1)
        encoded = self.transformer(stack)
        if self.fusion == "concat":
            fused = encoded.mean(dim=1)
        elif self.fusion == "cross_attention":
            query = self.cross_query.expand(encoded.shape[0], -1, -1)
            fused = self.cross_attention(query, encoded, encoded, need_weights=False)[0].squeeze(1)
        else:
            gate_input = encoded.mean(dim=1)
            if self.fusion == "gated_qc":
                qc = torch.tensor(present, device=mri.device, dtype=gate_input.dtype).view(1, 3)
                gate_input = gate_input + self.qc_projection(qc)
            weights = torch.softmax(self.gate(gate_input), dim=1)
            present_tensor = torch.tensor(present, device=mri.device).view(1, 3)
            weights = weights * present_tensor
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            fused = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(fused)
