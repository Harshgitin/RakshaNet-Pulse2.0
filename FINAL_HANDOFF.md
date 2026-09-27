# RakshaNet Extreme Final Handoff

This build keeps the existing RakshaNet visual language while making the data and alert flows more explicit and location-driven.

## Final behavior
- Registration stores permanent location and selected hazard preferences.
- Existing users are kept logged in in the same browser and land on Profile instead of being asked to register again.
- Intelligence Center supports Monitoring Focus, Current Location, Search Location and exact-coordinate analysis.
- Home Region Analysis calls the live multi-hazard endpoint rather than a static score.
- Official NDMA/SACHET alerts are matched by location text and, when the CAP data provides them, by CAP polygon/circle geometry.
- Flood guidance combines GloFAS river-discharge information with live/forecast rainfall; it is explicitly an index, not a fabricated probability.
- NER overview refreshes live weather/hydrology, official alerts and nearby USGS earthquake activity for state reference points.
- User alert preferences drive the automated scanner hazard-by-hazard: heavy rain, flood, landslide, thunderstorm, extreme heat and cyclone.
- App/browser Web Push and SMS are separate delivery channels. Real SMS still requires an approved provider account/configuration.
- Offline mode caches core pages and queues emergency reports locally in IndexedDB until network connectivity returns. Live external data is intentionally unavailable while offline.

## Important truthfulness rule
RakshaNet never presents a derived hazard index as an official probability. Official NDMA/IMD/district warnings are shown separately and take precedence. The deployed landslide ONNX percentage remains the model output of the currently trained artifact.

## Run
```bash
npm install
npm run check
npm start
```

Open `http://localhost:5000`.
