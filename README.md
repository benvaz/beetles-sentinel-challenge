# Beetles as Sentinel Taxa — SPEI Drought Prediction

Ensemble of three DINOv2-based MIL models predicting drought conditions (SPEI) from carabid beetle specimen images for the [2025 HDR SMood Challenge](https://www.codabench.org/competitions/9854/).

## Repository Structure
```
submission/
  model.py                        # inference (CodaBench submission)
  lookup.json                     # species prior lookup table
  bioclip_svr_predictions.json    # BioCLIP SVR predictions
  requirements.txt
training/
  train.py                        # training loop
  evaluation.py                   # local validation
  utils.py                        # shared helpers
  model_base.py                   # DINO_DeepRegressor architecture
```

Model weights (v2_model.pth, v5_model.pth, v6_model.pth) are available in [Releases](../../releases).

## Approach

Three frozen DINOv2 (facebook/dinov2-base) models combined via ensemble averaging with species-level lookup blending:

- **V2** — Gated attention MIL with mu/logvar heads
- **V5** — Species + domain embedding fusion MIL
- **V6** — Species prior + residual MIL

Final prediction: `mu = alpha * ensemble_mu + (1 - alpha) * species_lookup_mu`

## Training
```bash
python training/train.py --hf_token <TOKEN>
```

## Requirements

torch, transformers, numpy, tqdm, scikit-learn, datasets

## References

- Oquab et al. "DINOv2: Learning robust visual features without supervision." arXiv:2304.07193 (2023).
- NEON Carabid Beetle Data Product DP1.10022.001
- [Challenge sample repo](https://github.com/Imageomics/HDR-SMood-Challenge-sample)

## License

MIT
