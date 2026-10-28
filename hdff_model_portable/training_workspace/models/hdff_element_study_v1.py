"""Element features, site interactions, and regression readouts."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from hdff_model_portable.training_workspace.models.hdff_scalable import (
    ScalableHDFFRegressor, MLP, masked_mean,
)

MODEL_ID = "hdff_element_study_v1"


class EmptyEmbedding(nn.Module):
    def forward(self, ids):
        return torch.empty((*ids.shape, 0), device=ids.device, dtype=torch.float32)


class ElementStudy(ScalableHDFFRegressor):
    def __init__(self, config):
        config = dict(config)
        study = config.pop("study")
        self.mode = study["representation"]
        self.architecture = study.get("architecture", "baseline")
        if self.mode not in ("embedding", "properties", "both"):
            raise ValueError("Invalid representation")
        if self.architecture not in ("baseline", "additive", "interaction"):
            raise ValueError("Invalid architecture")
        props = study["property_indices"] if self.mode != "embedding" else []
        globals_ = study["global_indices"]
        config["atom_prop_dim"] = len(props)
        config["global_feature_dim"] = len(globals_)
        if self.mode == "properties":
            config["element_emb_dim"] = 0
        super().__init__(**config, local_refinement_layers=2,
                         structure_refinement_layers=2, fusion_refinement_layers=2,
                         initial_residual_alpha=0.1, jumping_knowledge=True,
                         deep_stabilization=True, variant_name=MODEL_ID)
        self.register_buffer("property_indices", torch.tensor(props, dtype=torch.long))
        self.register_buffer("global_indices", torch.tensor(globals_, dtype=torch.long))
        self.register_buffer("energy_mean", torch.tensor(float(study["target_mean"])))
        self.register_buffer("energy_std", torch.tensor(float(study["target_std"])))
        if self.mode == "properties":
            self.local_encoder.element_embedding = EmptyEmbedding()
            self.global_encoder.element_embedding = EmptyEmbedding()
        if self.architecture == "interaction":
            d = config["local_dim"]
            if d != config["special_dim"]:
                raise ValueError("Interaction requires aligned local/global dimensions")
            self.exchanges = nn.ModuleList([
                nn.ModuleDict({"norm": nn.LayerNorm(2*d),
                               "to_local": MLP(2*d, 2*d, d, 2, config["dropout"]),
                               "to_global": MLP(2*d, 2*d, d, 2, config["dropout"])})
                for _ in self.global_encoder.layers
            ])
            self.exchange_gates = nn.Parameter(torch.full((len(self.exchanges), 2), -2.0))
        self.architecture_spec.update(study)
        self.architecture_spec["energy_readout"] = (
            "sum of dopant-site scalar eV contributions, then target normalization"
            if self.architecture == "additive" else "original normalized scalar head")

    def select_inputs(self, batch):
        batch = dict(batch)
        for key in ("local_edge_atom_props", "local_edge_prop_diff", "special_atom_props"):
            batch[key] = batch[key].index_select(-1, self.property_indices)
        batch["global_features"] = batch["global_features"].index_select(-1, self.global_indices)
        return batch

    def interacting_nodes(self, local, batch):
        enc = self.global_encoder
        mask = batch["special_mask"].unsqueeze(-1).to(local.dtype)
        initial = enc.site_init(torch.cat([
            local, batch["special_atom_props"],
            enc.element_embedding(batch["special_element_ids"])], dim=-1)) * mask
        node, states = initial, [initial]
        relation = enc.relation_features(batch["special_distances"], batch["special_layer_relations"])
        for index, layer in enumerate(enc.layers):
            node = layer(node, relation, batch["special_distances"],
                         batch["special_layer_relations"], batch["special_mask"],
                         F.softplus(enc.theta_same)+1e-8, F.softplus(enc.theta_cross)+1e-8)
            node = ((1-enc.initial_residual_alpha)*node + enc.initial_residual_alpha*initial)*mask
            exchange = self.exchanges[index]
            joint = exchange["norm"](torch.cat([local, node], dim=-1))
            gates = torch.sigmoid(self.exchange_gates[index])
            local = (local + gates[0]*exchange["to_local"](joint))*mask
            node = (node + gates[1]*exchange["to_global"](joint))*mask
            states.append(node)
        node = sum(w*s for w, s in zip(torch.softmax(enc.jk_logits, dim=0), states))
        return local, enc.output_norm(node)*mask

    def forward(self, batch):
        batch = self.select_inputs(batch)
        local, zlocal = self.local_encoder(batch)
        if self.architecture == "interaction":
            local, node = self.interacting_nodes(local, batch)
            zlocal = masked_mean(local, batch["special_mask"], dim=1)
            zglobal = masked_mean(node, batch["special_mask"], dim=1)
        else:
            node, zglobal = self.global_encoder(local, batch)
        structure = self.structure_encoder(batch["global_features"])
        if self.architecture == "additive":
            b, s, _ = local.shape
            energy = self.fusion(local.reshape(b*s, -1), node.reshape(b*s, -1),
                                 structure[:, None, :].expand(-1, s, -1).reshape(b*s, -1))
            raw = (energy.reshape(b, s, -1)*batch["special_mask"].unsqueeze(-1)).sum(dim=1)
            return (raw-self.energy_mean)/self.energy_std
        return self.fusion(zlocal, zglobal, structure)


def build_model(model_config):
    return ElementStudy(model_config)
