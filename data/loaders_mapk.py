import pandas as pd
import numpy as np

def load_mapk_data(path):
    xls = pd.ExcelFile(path)
    dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None, skiprows=1)
        df = df.iloc[:, :3]
        df.columns = ["Ligand_SMILES", "Kd_nM", "Protein"]
        df["Kinase"] = sheet
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["Kd_nM"] = pd.to_numeric(df["Kd_nM"], errors="coerce")
    df = df[(df["Kd_nM"] > 1e-3) & (df["Kd_nM"] < 1e8)]
    df["log_Kd"] = np.log10(df["Kd_nM"] + 1e-9)
    df = df.dropna()
    return df
