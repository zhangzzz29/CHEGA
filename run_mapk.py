import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from data.loaders_mapk import load_mapk_data
from data.dataset import MultiModalKinaseDataset, multimodal_collate
from models.full_model import FullKinaseInhibitorNet
from train.trainer import train_fullmodel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "fusion_dim": 256,
    "mol_fusion_mode": "self_attn",
    "protein_model_type": "esm2",
    "protein_local_paths": {
        "esm2": "esm2_localpath"
    },
    "freeze_protein": False,
    "freeze_protein_ratio": 0.5
}

df_train = load_mapk_data("Original-training-data.xlsx")
df_val = load_mapk_data("Original-validation-data.xlsx")

ctok = AutoTokenizer.from_pretrained("bert_localpath", local_files_only=True)
ptok = AutoTokenizer.from_pretrained("esm2_localpath", local_files_only=True)

train_ds = MultiModalKinaseDataset(
    df_train.Ligand_SMILES.tolist(),
    df_train.Protein.tolist(),
    df_train.log_Kd.values,
    ctok, ptok,
    augment = True  # 开启数据增强
)
val_ds = MultiModalKinaseDataset(
    df_val.Ligand_SMILES.tolist(),
    df_val.Protein.tolist(),
    df_val.log_Kd.values,
    ctok, ptok
)

train_dl = DataLoader(train_ds, 16, shuffle=True, collate_fn=multimodal_collate)
val_dl = DataLoader(val_ds, 16, shuffle=False, collate_fn=multimodal_collate)

model = FullKinaseInhibitorNet(config)

train_fullmodel(
    model,
    train_dl,
    val_dl,
    epochs=70,
    lr=2e-4,
    save_path="best_mapk.pt",
    device=device
)
