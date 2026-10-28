"""Embedding and shell-aggregation modules for the HDFF model."""
import json
import math
from pathlib import Path
import torch
from torch import nn
from hdff_model_portable.training_workspace.models.hdff_element_study_v1 import ElementStudy, EmptyEmbedding
from hdff_model_portable.training_workspace.models.hdff_scalable import LocalHexEncoder, ResidualProjection, masked_mean


class CompactRole(nn.Module):
    """Expand compact role embeddings while preserving their scale."""
    def __init__(self, dimension, output_dim):
        super().__init__()
        if output_dim % dimension:
            raise ValueError("Role dimension must divide edge dimension")
        self.embedding = nn.Embedding(4, dimension)
        self.repeats = output_dim // dimension

    def forward(self, ids):
        return self.embedding(ids).repeat_interleave(self.repeats, dim=-1) / math.sqrt(self.repeats)


class ZeroRole(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, ids):
        return torch.zeros((*ids.shape, self.dimension), device=ids.device)


class InverseShellEncoder(LocalHexEncoder):
    def forward(self, batch):
        windows = self.hex_attention(self.encode_edges(batch), batch["window_mask"])
        shells, present = [], []
        for k in range(self.n_shells):
            mask = batch["window_mask"] & (batch["window_shell"] == k)
            shells.append(masked_mean(windows, mask, dim=2))
            present.append(mask.any(dim=2))
        valid = torch.stack(present, dim=-1)
        radii = batch["shell_radii"][:, None, :]
        raw = valid.to(windows.dtype) / radii.clamp_min(1e-8)
        weights = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = (torch.stack(shells, dim=2) * weights.unsqueeze(-1)).sum(dim=2)
        local = self.shell_fusion(pooled) * batch["special_mask"].unsqueeze(-1)
        return local, masked_mean(local, batch["special_mask"], dim=1)


class EmbeddingAblation(ElementStudy):
    def __init__(self, config):
        config = dict(config)
        options = config["study"]["embedding_ablation"]
        self.inverse_shell = bool(options.get("inverse_shell", False))
        if self.inverse_shell:
            config["shell_emb_dim"] = 0
        if options.get("no_layer_embedding", False):
            config["relation_emb_dim"] = 0
        super().__init__(config)
        if self.mode != "properties" or self.architecture != "interaction":
            raise ValueError("This study freezes the properties-only interaction backbone")
        local = self.local_encoder
        if self.inverse_shell:
            local.__class__ = InverseShellEncoder
            local.shell_embedding = EmptyEmbedding()
            local.shell_fusion = ResidualProjection(local.edge_dim, 2*local.local_dim,
                                                   local.local_dim, config["dropout"])
            data = json.loads(Path(config["study"]["shell_radii_file"]).read_text())
            ids = sorted(data["radii_by_structure"])
            self.radius_index = {sid:i for i,sid in enumerate(ids)}
            self.register_buffer("radius_table", torch.tensor([data["radii_by_structure"][sid] for sid in ids]))
        role_dim = int(options.get("role_dim", 128))
        if role_dim == 16:
            local.role_embedding = CompactRole(16, local.edge_dim)
        elif role_dim == 0:
            local.role_embedding = ZeroRole(local.edge_dim)
            del local.role_bias
            local.register_buffer("role_bias", torch.zeros(local.num_heads, 4))
        elif role_dim != local.edge_dim:
            raise ValueError("Unsupported role dimension")
        if options.get("no_layer_embedding", False):
            self.global_encoder.layer_relation_embedding = EmptyEmbedding()

    def forward(self, batch):
        if self.inverse_shell:
            batch = dict(batch)
            ids = [str(m.get("structure_id", m.get("sample_id"))) for m in batch["metadata"]]
            index = torch.tensor([self.radius_index[sid] for sid in ids], device=self.radius_table.device)
            batch["shell_radii"] = self.radius_table.index_select(0, index)
        return super().forward(batch)


def build_model(model_config):
    return EmbeddingAblation(model_config)
