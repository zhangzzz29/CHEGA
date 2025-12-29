import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from data.loaders_pdb import load_pdbbind_data
from data.dataset import MultiModalKinaseDataset, multimodal_collate
from models.full_model import FullKinaseInhibitorNet
from train.trainer import train_fullmodel
from train.evaluator import evaluate_regression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "fusion_dim": 256,
    "mol_fusion_mode": "self_attn",
    "protein_model_type": "esm2",
    "protein_local_paths": {
        "esm2": "esm2_localpath"
    },
    "freeze_protein": False,
    "freeze_protein_ratio": 0.3
}

df_train = load_pdbbind_data("pdb_train.xlsx")
df_test = load_pdbbind_data("pdb_test.xlsx")

ctok = AutoTokenizer.from_pretrained("bert_localpath", local_files_only=True)
ptok = AutoTokenizer.from_pretrained("esm2_localpath", local_files_only=True)

train_ds = MultiModalKinaseDataset(
    df_train.Ligand_SMILES.tolist(),
    df_train.Protein.tolist(),
    df_train.binding_label.values,
    ctok, ptok,
    augment=True
)
test_ds = MultiModalKinaseDataset(
    df_test.Ligand_SMILES.tolist(),
    df_test.Protein.tolist(),
    df_test.binding_label.values,
    ctok, ptok
)

train_dl = DataLoader(train_ds, 32, shuffle=True, collate_fn=multimodal_collate)
test_dl = DataLoader(test_ds, 32, shuffle=False, collate_fn=multimodal_collate)

model = FullKinaseInhibitorNet(config)

model = train_fullmodel(
    model,
    train_dl,
    test_dl,
    epochs=70,
    lr=2e-4,
    save_path="best_pdb.pt",
    device=device
)
