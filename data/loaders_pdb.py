import pandas as pd

def load_pdbbind_data(path):
    df = pd.read_excel(path)
    df = df.rename(columns={
        "compound_iso_smiles": "Ligand_SMILES",
        "target_sequence": "Protein",
        "label": "binding_label"
    })
    df = df.dropna(subset=["Ligand_SMILES", "Protein", "binding_label"])
    return df
