"""HDFF graph encoders and regression modules."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activate_last: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        dims = [input_dim] + [hidden_dim] * max(num_layers - 1, 0) + [output_dim]
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            is_last = index == len(dims) - 2
            if not is_last or activate_last:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualProjection(nn.Module):
    """Residual projection with output normalization."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim)
        self.main = MLP(input_dim, hidden_dim, output_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.skip(x) + self.main(x))


class PreNormResidualBlock(nn.Module):
    """Pre-normalized residual MLP."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.main = MLP(dim, hidden_dim, dim, num_layers=2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.main(self.norm(x))


class RBFExpansion(nn.Module):
    def __init__(self, count: int, max_distance: float, gamma: float | None = None):
        super().__init__()
        if count < 1:
            raise ValueError("RBF count must be >= 1.")
        centers = torch.linspace(0.0, float(max_distance), count)
        self.register_buffer("centers", centers)
        if gamma is None:
            step = float(max_distance) / max(count - 1, 1)
            gamma = 1.0 / max(step * step, 1.0e-6)
        self.gamma_r = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        gamma = F.softplus(self.gamma_r) + 1.0e-8
        return torch.exp(-gamma * (distances.unsqueeze(-1) - self.centers) ** 2)


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / count


class LocalHexEncoder(nn.Module):
    """Encode local six-edge neighborhoods."""

    def __init__(
        self,
        atom_prop_dim: int,
        num_elements: int,
        n_shells: int,
        element_emb_dim: int,
        shell_emb_dim: int,
        edge_dim: int,
        local_dim: int,
        num_heads: int,
        dropout: float,
        refinement_layers: int,
    ):
        super().__init__()
        if edge_dim % num_heads != 0:
            raise ValueError("edge_dim must be divisible by local attention heads.")
        self.edge_dim = edge_dim
        self.local_dim = local_dim
        self.n_shells = n_shells
        self.num_heads = num_heads
        self.head_dim = edge_dim // num_heads

        self.element_embedding = nn.Embedding(num_elements, element_emb_dim)
        self.shell_embedding = nn.Embedding(n_shells, shell_emb_dim)
        node_dim = atom_prop_dim + element_emb_dim + 2
        raw_edge_dim = atom_prop_dim + shell_emb_dim + 1
        edge_input_dim = 2 * node_dim + raw_edge_dim
        self.edge_encoder = ResidualProjection(edge_input_dim, 2 * edge_dim, edge_dim, dropout)

        self.register_buffer("role_ids", torch.tensor([0, 1, 1, 2, 3, 3], dtype=torch.long))
        self.role_embedding = nn.Embedding(4, edge_dim)
        self.role_bias = nn.Parameter(torch.zeros(num_heads, 4))
        self.query = nn.Linear(edge_dim, edge_dim)
        self.key = nn.Linear(edge_dim, edge_dim)
        self.value = nn.Linear(edge_dim, edge_dim)
        self.gate = nn.Linear(2 * edge_dim, edge_dim)
        self.update = MLP(2 * edge_dim, 2 * edge_dim, edge_dim, num_layers=2, dropout=dropout)
        self.update_norm = nn.LayerNorm(edge_dim)
        self.refinement = nn.ModuleList(
            [PreNormResidualBlock(edge_dim, 2 * edge_dim, dropout) for _ in range(refinement_layers)]
        )
        self.shell_fusion = ResidualProjection(n_shells * edge_dim, 2 * local_dim, local_dim, dropout)

    def node_features(
        self,
        props: torch.Tensor,
        element_ids: torch.Tensor,
        center: torch.Tensor,
        dopant: torch.Tensor,
    ) -> torch.Tensor:
        element = self.element_embedding(element_ids)
        return torch.cat([props, element, center.unsqueeze(-1), dopant.unsqueeze(-1)], dim=-1)

    def encode_edges(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        props = batch["local_edge_atom_props"]
        element_ids = batch["local_edge_element_ids"]
        center_flags = batch["local_edge_center_flags"]
        dopant_flags = batch["local_edge_dopant_flags"]
        prop_diff = batch["local_edge_prop_diff"]
        center_edge = batch["local_edge_center_mask"].unsqueeze(-1)
        shell = batch["window_shell"].clamp_min(0)
        start_node = self.node_features(
            props[..., 0, :], element_ids[..., 0], center_flags[..., 0], dopant_flags[..., 0]
        )
        end_node = self.node_features(
            props[..., 1, :], element_ids[..., 1], center_flags[..., 1], dopant_flags[..., 1]
        )
        shell_emb = self.shell_embedding(shell).unsqueeze(3).expand(*prop_diff.shape[:-1], -1)
        raw_edge = torch.cat([prop_diff, shell_emb, center_edge], dim=-1)
        return self.edge_encoder(torch.cat([start_node, end_node, raw_edge], dim=-1))

    def hex_attention(self, edge_repr: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_special, max_windows, role_count, _ = edge_repr.shape
        roles = self.role_ids.to(edge_repr.device)
        edge_repr = edge_repr + self.role_embedding(roles).view(1, 1, 1, role_count, self.edge_dim)
        q = self.query(edge_repr[..., 0, :]).view(
            batch_size, max_special, max_windows, self.num_heads, self.head_dim
        )
        k = self.key(edge_repr).view(
            batch_size, max_special, max_windows, role_count, self.num_heads, self.head_dim
        )
        v = self.value(edge_repr).view(
            batch_size, max_special, max_windows, role_count, self.num_heads, self.head_dim
        )
        scores = (q.unsqueeze(3) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + self.role_bias[:, roles].transpose(0, 1).view(
            1, 1, 1, role_count, self.num_heads
        )
        weights = torch.softmax(scores, dim=3)
        context = (weights.unsqueeze(-1) * v).sum(dim=3).reshape(
            batch_size, max_special, max_windows, self.edge_dim
        )
        center_edge = edge_repr[..., 0, :]
        zeta = torch.sigmoid(self.gate(torch.cat([center_edge, context], dim=-1)))
        updated = self.update_norm(
            center_edge + zeta * self.update(torch.cat([center_edge, context], dim=-1))
        )
        for block in self.refinement:
            updated = block(updated)
        return updated * window_mask.unsqueeze(-1).to(updated.dtype)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        window_mask = batch["window_mask"]
        special_mask = batch["special_mask"]
        center_edge = self.hex_attention(self.encode_edges(batch), window_mask)
        shell_reprs = []
        for shell_index in range(self.n_shells):
            shell_mask = window_mask & (batch["window_shell"] == shell_index)
            shell_count = shell_mask.sum(dim=2, keepdim=True).clamp_min(1)
            pooled = (center_edge * shell_mask.unsqueeze(-1).to(center_edge.dtype)).sum(dim=2)
            pooled = pooled / shell_count.to(center_edge.dtype)
            pooled = pooled * shell_mask.any(dim=2, keepdim=True).to(center_edge.dtype)
            shell_reprs.append(pooled)
        site_local = self.shell_fusion(torch.cat(shell_reprs, dim=-1))
        site_local = site_local * special_mask.unsqueeze(-1).to(site_local.dtype)
        return site_local, masked_mean(site_local, special_mask, dim=1)


class GlobalRelationLayer(nn.Module):
    """Relation-aware attention with configurable normalization."""

    def __init__(
        self,
        hidden_dim: int,
        relation_dim: int,
        num_heads: int,
        dropout: float,
        pre_norm: bool,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by global attention heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.pre_norm = pre_norm
        self.attention_norm = nn.LayerNorm(hidden_dim) if pre_norm else nn.Identity()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relation_bias = MLP(relation_dim, max(64, hidden_dim // 2), num_heads, 2, dropout)
        self.relation_message = MLP(relation_dim, hidden_dim, hidden_dim, 2, dropout)
        self.update_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.update = MLP(2 * hidden_dim, 2 * hidden_dim, hidden_dim, 2, dropout)
        self.update_norm = nn.Identity() if pre_norm else nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, 2, dropout)

    def forward(
        self,
        node: torch.Tensor,
        relation: torch.Tensor,
        distances: torch.Tensor,
        layer_relations: torch.Tensor,
        special_mask: torch.Tensor,
        lambda_same: torch.Tensor,
        lambda_cross: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_special, _ = node.shape
        normed = self.attention_norm(node)
        q = self.query(normed).view(batch_size, max_special, self.num_heads, self.head_dim)
        k = self.key(normed).view(batch_size, max_special, self.num_heads, self.head_dim)
        v = self.value(normed).view(batch_size, max_special, self.num_heads, self.head_dim)
        scores = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(self.head_dim)
        scores = scores + self.relation_bias(relation)
        lambdas = torch.where(layer_relations == 0, lambda_same, lambda_cross)
        scores = scores - lambdas.unsqueeze(-1) * distances.unsqueeze(-1)

        eye = torch.eye(max_special, dtype=torch.bool, device=node.device).unsqueeze(0)
        pair_mask = special_mask.unsqueeze(1) & special_mask.unsqueeze(2) & ~eye
        has_neighbors = pair_mask.any(dim=2, keepdim=True).unsqueeze(-1)
        scores = scores.masked_fill(~pair_mask.unsqueeze(-1), -1.0e9)
        weights = torch.softmax(scores, dim=2)
        weights = torch.where(has_neighbors, weights, torch.zeros_like(weights))
        relation_msg = self.relation_message(relation).view(
            batch_size, max_special, max_special, self.num_heads, self.head_dim
        )
        values = v.unsqueeze(1) + relation_msg
        message = (weights.unsqueeze(-1) * values).sum(dim=2).reshape(
            batch_size, max_special, self.hidden_dim
        )
        eta = torch.sigmoid(self.update_gate(torch.cat([node, message], dim=-1)))
        mid = self.update_norm(node + eta * self.update(torch.cat([node, message], dim=-1)))
        if self.pre_norm:
            out = mid + self.ffn(self.ffn_norm(mid))
        else:
            out = self.ffn_norm(mid + self.ffn(mid))
        return out * special_mask.unsqueeze(-1).to(out.dtype)


class GlobalRelationEncoder(nn.Module):
    def __init__(
        self,
        atom_prop_dim: int,
        num_elements: int,
        local_dim: int,
        hidden_dim: int,
        element_emb_dim: int,
        relation_emb_dim: int,
        global_rbf_dim: int,
        global_rbf_max_distance: float,
        num_heads: int,
        num_layers: int,
        initial_decay: float,
        dropout: float,
        initial_residual_alpha: float,
        jumping_knowledge: bool,
        pre_norm: bool,
    ):
        super().__init__()
        if not 0.0 <= initial_residual_alpha < 1.0:
            raise ValueError("initial_residual_alpha must be in [0, 1).")
        self.initial_residual_alpha = initial_residual_alpha
        self.jumping_knowledge = jumping_knowledge
        self.element_embedding = nn.Embedding(num_elements, element_emb_dim)
        self.layer_relation_embedding = nn.Embedding(2, relation_emb_dim)
        self.global_rbf = RBFExpansion(global_rbf_dim, global_rbf_max_distance)
        self.site_init = nn.Sequential(
            MLP(local_dim + atom_prop_dim + element_emb_dim, 2 * hidden_dim, hidden_dim, 2, dropout),
            nn.LayerNorm(hidden_dim),
        )
        relation_dim = global_rbf_dim + relation_emb_dim
        self.layers = nn.ModuleList(
            [
                GlobalRelationLayer(hidden_dim, relation_dim, num_heads, dropout, pre_norm)
                for _ in range(num_layers)
            ]
        )
        self.jk_logits = nn.Parameter(torch.zeros(num_layers + 1)) if jumping_knowledge else None
        self.output_norm = nn.LayerNorm(hidden_dim) if pre_norm else nn.Identity()
        init_theta = inverse_softplus(initial_decay)
        self.theta_same = nn.Parameter(torch.tensor(init_theta))
        self.theta_cross = nn.Parameter(torch.tensor(init_theta))

    def relation_features(self, distances: torch.Tensor, layer_relations: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.global_rbf(distances), self.layer_relation_embedding(layer_relations)], dim=-1
        )

    def forward(self, site_local: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        special_mask = batch["special_mask"]
        node_input = torch.cat(
            [
                site_local,
                batch["special_atom_props"],
                self.element_embedding(batch["special_element_ids"]),
            ],
            dim=-1,
        )
        initial = self.site_init(node_input) * special_mask.unsqueeze(-1).to(site_local.dtype)
        node = initial
        states = [initial]
        relation = self.relation_features(
            batch["special_distances"], batch["special_layer_relations"]
        )
        lambda_same = F.softplus(self.theta_same) + 1.0e-8
        lambda_cross = F.softplus(self.theta_cross) + 1.0e-8
        for layer in self.layers:
            node = layer(
                node,
                relation,
                batch["special_distances"],
                batch["special_layer_relations"],
                special_mask,
                lambda_same,
                lambda_cross,
            )
            if self.initial_residual_alpha > 0:
                node = (
                    (1.0 - self.initial_residual_alpha) * node
                    + self.initial_residual_alpha * initial
                )
                node = node * special_mask.unsqueeze(-1).to(node.dtype)
            states.append(node)
        if self.jumping_knowledge:
            weights = torch.softmax(self.jk_logits, dim=0)
            node = sum(weight * state for weight, state in zip(weights, states))
        node = self.output_norm(node) * special_mask.unsqueeze(-1).to(node.dtype)
        return node, masked_mean(node, special_mask, dim=1)


class StructureEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float, refinement_layers: int):
        super().__init__()
        self.projection = ResidualProjection(input_dim, 2 * output_dim, output_dim, dropout)
        self.refinement = nn.ModuleList(
            [PreNormResidualBlock(output_dim, 2 * output_dim, dropout) for _ in range(refinement_layers)]
        )
        self.output_norm = nn.LayerNorm(output_dim) if refinement_layers > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        for block in self.refinement:
            x = block(x)
        return self.output_norm(x)


class BranchFusion(nn.Module):
    def __init__(
        self,
        local_dim: int,
        special_dim: int,
        structure_dim: int,
        fusion_dim: int,
        dropout: float,
        target_dim: int,
        refinement_layers: int,
    ):
        super().__init__()
        self.align_local = nn.Sequential(nn.Linear(local_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.align_global = nn.Sequential(nn.Linear(special_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.align_struct = nn.Sequential(nn.Linear(structure_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.gate = MLP(3 * fusion_dim, fusion_dim, 3, 2, dropout)
        self.fusion_projection = ResidualProjection(3 * fusion_dim, 2 * fusion_dim, fusion_dim, dropout)
        self.refinement = nn.ModuleList(
            [PreNormResidualBlock(fusion_dim, 2 * fusion_dim, dropout) for _ in range(refinement_layers)]
        )
        self.output_norm = nn.LayerNorm(fusion_dim) if refinement_layers > 0 else nn.Identity()
        self.output = MLP(fusion_dim, max(64, fusion_dim // 2), target_dim, 2, dropout)

    def forward(self, z_local: torch.Tensor, z_global: torch.Tensor, z_str: torch.Tensor) -> torch.Tensor:
        z_local = self.align_local(z_local)
        z_global = self.align_global(z_global)
        z_str = self.align_struct(z_str)
        gate_input = torch.cat([z_local, z_global, z_str], dim=-1)
        weights = torch.softmax(self.gate(gate_input), dim=-1)
        fused = self.fusion_projection(
            torch.cat(
                [
                    weights[:, 0:1] * z_local,
                    weights[:, 1:2] * z_global,
                    weights[:, 2:3] * z_str,
                ],
                dim=-1,
            )
        )
        for block in self.refinement:
            fused = block(fused)
        return self.output(self.output_norm(fused))


class ScalableHDFFRegressor(nn.Module):
    def __init__(
        self,
        atom_prop_dim: int,
        global_feature_dim: int,
        target_dim: int,
        n_shells: int,
        num_elements: int = 3,
        element_emb_dim: int = 16,
        shell_emb_dim: int = 8,
        relation_emb_dim: int = 8,
        global_rbf_dim: int = 16,
        global_rbf_max_distance: float = 20.0,
        edge_dim: int = 128,
        local_dim: int = 128,
        special_dim: int = 128,
        global_dim: int = 128,
        fusion_dim: int = 128,
        local_heads: int = 4,
        global_heads: int = 4,
        global_layers: int = 2,
        distance_decay_init: float = 0.1,
        dropout: float = 0.05,
        special_pooling: str = "mean",
        local_refinement_layers: int = 0,
        structure_refinement_layers: int = 0,
        fusion_refinement_layers: int = 0,
        initial_residual_alpha: float = 0.0,
        jumping_knowledge: bool = False,
        deep_stabilization: bool = False,
        variant_name: str = "scalable",
    ):
        super().__init__()
        if special_pooling != "mean":
            raise ValueError("HDFF scalable variants use masked mean pooling.")
        self.local_encoder = LocalHexEncoder(
            atom_prop_dim,
            num_elements,
            n_shells,
            element_emb_dim,
            shell_emb_dim,
            edge_dim,
            local_dim,
            local_heads,
            dropout,
            local_refinement_layers,
        )
        self.global_encoder = GlobalRelationEncoder(
            atom_prop_dim,
            num_elements,
            local_dim,
            special_dim,
            element_emb_dim,
            relation_emb_dim,
            global_rbf_dim,
            global_rbf_max_distance,
            global_heads,
            global_layers,
            distance_decay_init,
            dropout,
            initial_residual_alpha,
            jumping_knowledge,
            deep_stabilization,
        )
        self.structure_encoder = StructureEncoder(
            global_feature_dim, global_dim, dropout, structure_refinement_layers
        )
        self.fusion = BranchFusion(
            local_dim,
            special_dim,
            global_dim,
            fusion_dim,
            dropout,
            target_dim,
            fusion_refinement_layers,
        )
        self.architecture_spec: dict[str, Any] = {
            "variant": variant_name,
            "edge_dim": edge_dim,
            "local_dim": local_dim,
            "special_dim": special_dim,
            "global_dim": global_dim,
            "fusion_dim": fusion_dim,
            "local_heads": local_heads,
            "global_heads": global_heads,
            "global_layers": global_layers,
            "local_refinement_layers": local_refinement_layers,
            "structure_refinement_layers": structure_refinement_layers,
            "fusion_refinement_layers": fusion_refinement_layers,
            "initial_residual_alpha": initial_residual_alpha,
            "jumping_knowledge": jumping_knowledge,
            "normalization": (
                "pre-LN in deep residual/relation blocks"
                if deep_stabilization
                else "HDFF-v1 post-LN pattern"
            ),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        site_local, z_local = self.local_encoder(batch)
        _, z_global = self.global_encoder(site_local, batch)
        z_str = self.structure_encoder(batch["global_features"])
        return self.fusion(z_local, z_global, z_str)


def validate_expected(model_config: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    config = dict(model_config)
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: configured={actual!r}, required={required!r}"
            for key, (actual, required) in mismatches.items()
        )
        raise ValueError(f"Architecture/config mismatch: {details}")
    return config
