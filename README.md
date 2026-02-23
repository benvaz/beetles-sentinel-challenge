# Beetles as Sentinel Taxa — SPEI Drought Prediction

Ensemble of three DINOv2-based MIL models predicting drought conditions (SPEI) from carabid beetle specimen images for the [2025 HDR SMood Challenge](https://www.codabench.org/competitions/9854/).

## Structure
```
submission/
  model.py                        # inference (CodaBench submission)
  lookup.json                     # species prior lookup table
  bioclip_svr_predictions.json
  requirements.txt
training/
  train.py
  evaluation.py
  utils.py
  model_base.py
```

Model weights are in [Releases](../../releases).

## Approach

Three frozen DINOv2 (facebook/dinov2-base) MIL models ensembled with species-level lookup blending:

- **V2** — Gated attention MIL
- **V5** — Species + domain embedding fusion MIL
- **V6** — Species prior + residual MIL

## References

- Oquab et al. "DINOv2: Learning robust visual features without supervision." arXiv:2304.07193 (2023).
- NEON Carabid Beetle Data Product DP1.10022.001
- [Challenge sample repo](https://github.com/Imageomics/HDR-SMood-Challenge-sample)

## License

MIT
