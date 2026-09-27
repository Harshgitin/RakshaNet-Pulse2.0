# RakshaNet 🛡️

RakshaNet is an AI-assisted disaster response prototype focused on early warning, landslide risk assessment, live GIS data, emergency reporting, SOS support, volunteer coordination, and an admin dashboard.

## Tech stack
- Frontend: HTML, CSS, JavaScript
- Backend: Node.js + Express
- Database: Supabase PostgreSQL
- Evidence storage: Supabase Storage
- GIS: OpenStreetMap / Overpass / OSRM
- Weather & warnings: IMD and Open-Meteo integrations
- ML: ONNX Runtime with a Random Forest landslide model

## Run locally
1. Install Node.js LTS.
2. Open this folder in VS Code.
3. Create a `.env` file from `.env.example`.
4. Add your own Supabase database URL, Supabase server-side secret key, and JWT secret. **Never commit `.env`.**
5. Install dependencies:
   ```bash
   npm install
   ```
6. Start the server:
   ```bash
   npm start
   ```
7. Open:
   http://localhost:5000

## Main API endpoints
- `GET /api/health` — backend health check
- `GET /api/db-test` — database connectivity test
- `POST /api/predict/landslide` — landslide ML inference
- `POST /api/auth/register` — register a user
- `POST /api/auth/login` — login
- `PUT /api/auth/profile/:id` — update profile
- `POST /api/admin/login` — administrator login
- `GET /api/dashboard` — admin dashboard data (JWT protected)
- `GET /api/reports` / `POST /api/reports` — emergency reports
- `GET /api/sos` / `POST /api/sos` — SOS alerts
- `GET /api/volunteers` / `POST /api/volunteers` — volunteers
- `GET /api/imd/district-warnings` — IMD district warning proxy
- `POST /api/gis/nearby` — Overpass GIS proxy

## Data & secrets
Supabase PostgreSQL is the primary persistent database for users, emergency reports, SOS alerts and volunteers. Evidence files are stored in the `rakshanet-evidence` Supabase Storage bucket.

The JSON files under `data/` are only safe local placeholders/legacy artifacts. `data/users.json` is intentionally ignored by Git because it can contain account information and password hashes.

Runtime uploads and generated weather cache files are also ignored by Git.

## Deployment
The same Node.js server can run on Render or another Node hosting platform. Configure the environment variables from `.env.example` in the hosting dashboard; do not upload secrets to GitHub.

## Current project scope
Real provider-based SMS alerts are **not enabled unless `SMS_WEBHOOK_URL` is configured**. The alert engine can generate localized in-app/browser advisories and can dispatch to a configured webhook.

## AI/ML upgrade
The current build keeps the existing 16-feature `rakshanet-landslide-v1` ONNX model and strengthens the surrounding intelligence pipeline. Live feature semantics are aligned between training and inference, historical NER inventory context is available, optional terrain context is queried through OpenTopoData, GIS can request ML risk grids, route risk can be assessed, localized early-warning text is available, and emergency reports use IndexedDB offline sync.

The deployed model is still preliminary and is not an official warning system. A composite operational risk index is also available, but it is not a calibrated probability. Terrain, satellite-derived numerical indices, direct soil sensors, and ground-movement telemetry are exposed as context/readiness inputs but are not silently claimed as model features until a new validated model is trained and exported.

Training safety:
- `ml/train_landslide_model.py` writes `ml/landslide_model_candidate.pkl` and `ml/model_candidate_validation_report.json` only. It never replaces the active ONNX/PKL pair automatically.
- `ml/export_landslide_onnx.py` exports `ml/landslide_model_candidate.onnx` only. Activate a new artifact manually after independent validation.

New intelligence endpoints:
- `POST /api/intelligence/grid` — ML risk estimates for up to 25 map points
- `POST /api/routes/risk` — ML risk assessment for sampled route coordinates
- `POST /api/alerts/evaluate` — localized alert/advisory generation
- `POST /api/alerts/dispatch` — optional webhook delivery adapter

Run `npm run check` after installation to validate the project structure and syntax.


## Automated early-warning delivery

RakshaNet includes a server-side automated early-warning engine. It can periodically evaluate monitored users' landslide intelligence, deduplicate repeated alerts, and deliver consented warnings through Web Push (app/browser) and SMS.

### App/browser alerts

Run `npm install`, then `npm run generate:vapid`, copy the generated `VAPID_*` values into `.env`, and use **My Profile → Enable App Alerts**.

### Automated SMS + push

Configure an approved SMS provider and set `ALERT_AUTOMATION_ENABLED=true`, plus the interval/threshold settings. Twilio REST uses `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and either `TWILIO_FROM_NUMBER` or `TWILIO_MESSAGING_SERVICE_SID`. A generic `SMS_WEBHOOK_URL` is also supported. `ALERT_TEST_PHONE` is the controlled SMS test destination.

For Indian SMS delivery, use a sender/template configuration that meets the current provider and telecom requirements.
