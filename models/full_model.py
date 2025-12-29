import torch.nn as nn
from models.chemberta import ChemBERTaRegressor
from models.gnn import GNNModel
from models.protein import ProteinTransformerFlex
from models.fusion import MoleculeFusion_v3, AttnFusion

class FullKinaseInhibitorNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["fusion_dim"]

        self.smiles_encoder = ChemBERTaRegressor(d)
        self.gnn_encoder = GNNModel(hidden_dim=d//2)
        self.mol_fusion = MoleculeFusion_v3(d, config["mol_fusion_mode"])

        self.prot_encoder = ProteinTransformerFlex(
            config["protein_model_type"],
            config["protein_local_paths"],
            config["freeze_protein"],
            config["freeze_protein_ratio"]
        )
        self.prot_proj = nn.Linear(
            self.prot_encoder.hidden_size, d
        )
        self.fusion = AttnFusion(d)

    def forward(self, sm_ids, sm_mask, gnn, pr_ids, pr_mask):
        sm = self.smiles_encoder(sm_ids, sm_mask)
        gnn = self.gnn_encoder.get_embedding(gnn)
        mol = self.mol_fusion(sm, gnn)
        prot = self.prot_proj(
            self.prot_encoder.get_embedding(pr_ids, pr_mask)
        )
        return self.fusion(mol, prot, mol * prot)
