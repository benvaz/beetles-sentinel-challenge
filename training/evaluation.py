from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from utils import get_training_args, get_dino, spei_r2_scores, save_results, compile_event_predictions, get_collate_fn
from model_base import DINO_DeepRegressor


# ---- evaluation ----

def _evaluate(model, loader):
    with torch.inference_mode():
        abs_err = 0
        preds, gts, events = [], [], []
        for imgs, targets, eids in tqdm(loader, desc="evaluating"):
            imgs, targets = imgs.cuda(), targets.cuda()
            out = model(imgs)
            abs_err += torch.mean(torch.abs(out - targets), dim=0)
            preds.extend(out.detach().cpu().numpy())
            gts.extend(targets.detach().cpu().numpy())
            events.extend(np.array(eids))

        g_ev, p_ev = compile_event_predictions(gts, preds, events)
        r30, r1y, r2y = spei_r2_scores(g_ev, p_ev)

        mae = abs_err / len(loader)
        for i, lbl in enumerate(["30d", "1y", "2y"]):
            print(f"  MAE SPEI_{lbl}: {mae[i].item():.4f}  r2: {[r30, r1y, r2y][i]:.4f}")

    return [x.item() for x in mae], [r30, r1y, r2y]


# ---- main ----

def main():
    args = get_training_args()
    here = Path(__file__).resolve().parent

    dino, proc = get_dino()
    model = DINO_DeepRegressor(dino).cuda()
    model.regressor.load_state_dict(torch.load(here / "model.pth"))

    ds = load_dataset("imageomics/sentinel-beetles", token=args.hf_token, split="validation")

    def _transforms(examples):
        examples["pixel_values"] = [
            proc(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            for img in examples["file_path"]
        ]
        return examples

    test_ds = ds.with_transform(_transforms)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=get_collate_fn(["eventID"]))

    mae, r2 = _evaluate(model, loader)
    save_results(here / "results.json", mae, r2)


if __name__ == "__main__":
    main()
