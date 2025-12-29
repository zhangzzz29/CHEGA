# CHEGA

A framework CHEGA for predicting kinase inhibitor affinity by combining self attention mechanism and pre trained model

## Tasks
- Kinase inhibitor binding affinity prediction (MAPK / multi-kinase)
- Drug–Target Affinity prediction (PDBBind 2016)

## Model
- ChemBERTa (SMILES encoder)
- GCN (molecular graph encoder)
- ESM2 (protein encoder)
- Three-way dynamic self-attention fusion

## Reproducibility
- All pretrained models are loaded locally

## Run
```bash
python run_mapk.py
python run_pdbbind.py

