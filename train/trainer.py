import torch
import numpy as np
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

def train_fullmodel(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    save_path,
    device,
    patience=15,
    freeze_backbone=False
):
    model.to(device)

    if freeze_backbone:
        for n,p in model.named_parameters():
            if "chemberta" in n or "protein" in n:
                p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-3
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(0.1 * epochs * len(train_loader)),
        epochs * len(train_loader)
    )

    huber = torch.nn.HuberLoss()
    mse = torch.nn.MSELoss()

    best = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        losses = []

        for b in train_loader:
            for k in b:
                b[k] = b[k].to(device)

            pred = model(
                b["smiles_input"],
                b["smiles_mask"],
                b["gnn"],
                b["prot_ids"],
                b["prot_mask"]
            )

            loss = 0.7 * huber(pred, b["label"]) + 0.3 * mse(pred, b["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        model.eval()
        vloss = []
        with torch.no_grad():
            for b in val_loader:
                for k in b:
                    b[k] = b[k].to(device)
                pred = model(
                    b["smiles_input"],
                    b["smiles_mask"],
                    b["gnn"],
                    b["prot_ids"],
                    b["prot_mask"]
                )
                vloss.append(huber(pred, b["label"]).item())

        v = np.mean(vloss)
        print(f"Epoch {epoch+1} | ValLoss {v:.4f}")

        if v < best:
            best = v
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(save_path))
    return model
