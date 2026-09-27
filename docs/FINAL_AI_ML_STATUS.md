# RakshaNet — Final AI/ML Implementation Status

## Active ML artifact
The shipped inference artifact remains `ml/landslide_model.onnx` (`rakshanet-landslide-v1`) with 16 weather/location features. The runtime feature semantics are aligned to the training pipeline for the temperature and wind windows.

## Integrated intelligence layers
- Live coordinate weather -> active ML features
- Historical NER/GSI/NASA-derived inventory -> contextual susceptibility
- Terrain elevation/slope -> contextual signal when available
- Coordinate soil moisture -> displayed contextual signal when available
- Assam government telemetry -> freshness-checked source layer
- NASA GIBS -> satellite observation layer
- Composite operational risk index -> combines the active ML estimate with contextual signals; it is **not** a calibrated probability
- ML risk grid -> map-ready endpoint
- Route risk sampling -> evacuation-route risk assessment
- Alert evaluation -> localized warning/advisory generation
- Offline emergency reports -> IndexedDB queue + service-worker caching for app shell

## Honest model limitation
The current ONNX model does not yet consume terrain, satellite-derived numerical features, direct ground soil sensors, river level, or ground-movement telemetry as trained input columns. The code exposes these sources as context/readiness information rather than silently pretending that they are model features.

## Training upgrade
`ml/train_landslide_model.py` now uses grouped spatial validation to reduce location leakage and writes to `landslide_model_candidate.pkl` so a training run cannot silently break the active model. `ml/export_landslide_onnx.py` exports a candidate ONNX artifact without replacing the active `landslide_model.onnx`. Activation should happen only after independent validation.

## Production prerequisites
1. Rebuild the training table from the larger historical inventory with matched weather windows.
2. Add terrain/satellite/soil/sensor columns and retrain a new artifact.
3. Validate on a truly independent time/region hold-out and calibrate probabilities.
4. Configure authenticated SMS/push delivery and production database/storage policies.
