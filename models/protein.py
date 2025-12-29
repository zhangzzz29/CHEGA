import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ProteinTransformerFlex(nn.Module):
    def __init__(self, model_type, local_paths,
                 freeze_layers=False, freeze_ratio=1.0):
        super().__init__()
        path = local_paths[model_type]
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True
        )
        self.hidden_size = self.model.config.hidden_size

        if freeze_layers:
            layers = list(self.model.encoder.layer)
            k = int(len(layers) * freeze_ratio)
            for layer in layers[:k]:
                for p in layer.parameters():
                    p.requires_grad = False

    def get_embedding(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return out.last_hidden_state[:, 0, :]
