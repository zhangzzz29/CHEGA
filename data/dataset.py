import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from models.gnn import smiles_to_graph
from utils.augment import MolAugment, ProteinAugment

class MultiModalKinaseDataset(Dataset):
    def __init__(
        self,
        smiles,
        proteins,
        labels,
        chemberta_tokenizer,
        protein_tokenizer,
        max_smiles_len=128,
        max_protein_len=2048,
        use_gnn=True,
        use_chemberta=True,
        augment = False
    ):
        self.smiles = smiles
        self.proteins = proteins
        self.labels = labels
        self.ctok = chemberta_tokenizer
        self.ptok = protein_tokenizer
        self.use_gnn = use_gnn
        self.use_chemberta = use_chemberta
        self.max_smiles_len = max_smiles_len
        self.max_protein_len = max_protein_len
        self.augment = augment

        if augment:
            self.mol_aug = MolAugment()
            self.prot_aug = ProteinAugment()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {}

        smi = self.smiles[idx]
        prot = self.proteins[idx]

        if self.augment:
            smi = self.mol_aug.randomize_smiles(smi)
            prot = self.prot_aug.random_substitution(prot)

        if self.use_chemberta:
            enc = self.ctok(
                self.smiles[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_smiles_len,
                return_tensors="pt"
            )
            item["smiles_input"] = enc["input_ids"].squeeze(0)
            item["smiles_mask"] = enc["attention_mask"].squeeze(0)

        if self.use_gnn:
            item["gnn"] = smiles_to_graph(self.smiles[idx])

        p = self.ptok(
            self.proteins[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_protein_len,
            return_tensors="pt"
        )
        item["prot_ids"] = p["input_ids"].squeeze(0)
        item["prot_mask"] = p["attention_mask"].squeeze(0)

        item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def multimodal_collate(batch):
    out = {}
    for k in batch[0]:
        if k == "gnn":
            out[k] = Batch.from_data_list([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
