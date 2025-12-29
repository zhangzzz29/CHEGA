import torch.nn as nn
from transformers import AutoModel

class ChemBERTaRegressor(nn.Module):
    def __init__(self, out_dim=256, model_path="bert_localpath"):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(
            model_path, local_files_only=True
        )
        hidden = self.chemberta.config.hidden_size
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def get_embedding(self, input_ids, attention_mask=None):
        out = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask=None):
        emb = self.get_embedding(input_ids, attention_mask)
        return self.regressor(emb)
