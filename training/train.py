from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from utils import get_training_args, get_dino, extract_dino_features, spei_r2_scores, get_collate_fn
from model_base import DINO_DeepRegressor


# ---- training loop ----

def _train(model, loader, val_loader, lr, epochs, save_dir):
    opt = optim.Adam(model.regressor.parameters(), lr)
    loss_fn = nn.MSELoss()
    best_r2, best_ep = -1.0, 0
    ckpt = Path(save_dir, "model.pth")

    bar = tqdm(range(epochs), position=0, leave=True)
    for ep in bar:

        # ---- train ----
        model.train()
        ep_loss = 0
        preds, gts = [], []
        for feats, y in tqdm(loader, "train", position=1, leave=False):
            y = y.cuda()
            opt.zero_grad()
            out = model.regressor(model.tokens_to_linear(feats.cuda()).squeeze())
            loss = loss_fn(y, out)
            loss.backward(); opt.step()
            ep_loss += loss
            preds.extend(out.detach().cpu().numpy().tolist())
            gts.extend(y.detach().cpu().numpy().tolist())

        gts, preds = np.array(gts), np.array(preds)
        r30, r1y, r2y = spei_r2_scores(gts, preds)
        log = {"loss": ep_loss.item() / len(loader), "ep": ep,
               "t_30d": r30, "t_1y": r1y, "t_2y": r2y}

        # ---- validate ----
        model.eval()
        ep_loss = 0
        preds, gts = [], []
        with torch.no_grad():
            for feats, y in tqdm(val_loader, "val", position=1, leave=False):
                y = y.cuda()
                out = model.regressor(model.tokens_to_linear(feats.cuda()).squeeze())
                loss = loss_fn(y, out)
                ep_loss += loss
                preds.extend(out.detach().cpu().numpy().tolist())
                gts.extend(y.detach().cpu().numpy().tolist())

        gts, preds = np.array(gts), np.array(preds)
        r30, r1y, r2y = spei_r2_scores(gts, preds)
        log |= {"v_loss": ep_loss.item() / len(loader),
                "v_30d": r30, "v_1y": r1y, "v_2y": r2y}

        avg_r2 = (r30 + r1y + r2y) / 3.0
        if avg_r2 >= best_r2:
            best_r2, best_ep = avg_r2, ep
            torch.save(model.regressor.state_dict(), ckpt)
        log |= {"best_ep": best_ep, "best_r2": best_r2}
        bar.set_postfix(log)

    model.regressor.load_state_dict(torch.load(ckpt))


# ---- main ----

def main():
    args = get_training_args()
    ds = load_dataset("imageomics/sentinel-beetles", token=args.hf_token)

    dino, proc = get_dino()
    model = DINO_DeepRegressor(dino).cuda()

    def _transforms(examples):
        examples["pixel_values"] = [
            proc(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            for img in examples["file_path"]
        ]
        return examples

    train_ds = ds["train"].with_transform(_transforms)
    val_ds = ds["validation"].with_transform(_transforms)

    # ---- extract frozen features, build tensor dataloaders ----
    loaders = []
    for i, dset in enumerate([train_ds, val_ds]):
        dl = DataLoader(dset, batch_size=args.batch_size,
                        shuffle=(i == 0), num_workers=args.num_workers,
                        collate_fn=get_collate_fn())
        X, Y = extract_dino_features(dl, dino)
        dl = DataLoader(torch.utils.data.TensorDataset(X, Y),
                        batch_size=args.batch_size, shuffle=(i == 0),
                        num_workers=args.num_workers)
        loaders.append(dl)

    _train(model, loaders[0], loaders[1],
           lr=args.lr, epochs=args.epochs,
           save_dir=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
