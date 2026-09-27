# RakshaNet AI/ML Upgrade Notes

## What changed

- The existing UI remains the default interface. The upgrade changes data handling and intelligence behavior rather than replacing the visual design.
- Live landslide features now use the same 24-hour temperature and wind semantics as the training dataset. This removes a train/inference feature-definition mismatch.
- ONNX prediction parsing now prefers the exported `label` and `probabilities` outputs and only uses shape-based fallbacks when necessary.
- A historical NER landslide inventory context layer was added. It reports nearby historical-event density and nearest-event distance without pretending that those counts are model probabilities.
- Terrain context was added through an optional OpenTopoData ASTER 30 m elevation query. A sampled elevation gradient is reported as a terrain slope proxy; it is contextual only and is not silently injected into the deployed v1 model.
- Model signal explanations use a profile generated from the deployed model and training dataset. These are range/importance signals, not causal explanations.
- A data-confidence score was added to indicate input completeness/source availability. It is not statistical model confidence.
- The GIS risk layer can now request a batch ML risk grid from `/api/intelligence/grid`, with a rule-based fallback when the backend is unavailable.
- Route results can be checked by the ML route-risk service at `/api/routes/risk`.
- `/api/alerts/evaluate` creates localized advisory text in English, Hindi, Assamese and Bengali. `/api/alerts/dispatch` supports an optional server-side webhook delivery adapter.
- Emergency reports now use IndexedDB for true offline browser queuing and automatically retry when connectivity returns. The service worker caches core local pages/assets for offline navigation.
- The model training script uses spatially grouped validation so paired event/control rows at the same coordinates are not split across train/test folds.
- `ml/model_profile.json` and `ml/model_validation_report.json` document the current deployed model context.

## Important limitation

The shipped ONNX model is still the existing 16-feature `rakshanet-landslide-v1` artifact. The upgrade intentionally does not replace it with an unverified larger-feature model because the build environment did not have the ONNX export dependency required to safely regenerate the artifact.

The current deployed model therefore does **not** directly consume terrain, satellite-derived numeric indices, ground-movement telemetry, or soil-moisture measurements as model features. Those sources are surfaced as context and readiness signals so future retraining can add them without creating a misleading claim today.

## Next retraining target

Use the combined GSI + NASA inventory for susceptibility/context and a weather-enriched, date-valid sample for supervised labels. Add terrain, soil moisture and satellite-derived numerical features only after a documented feature schema, leakage-resistant validation, calibration, and a new ONNX artifact are produced.
