# RakshaNet Pulse 2.0 — SIH26069

Pulse 2.0 is an isolated National Weather Big Data Analytics module. Existing RakshaNet pages remain unchanged except for one new Home button that opens `pages/pulse.html`.

## Live connectors
- NDMA SACHET official CAP/RSS alerts
- IMD district warning API (uses `IMD_API_KEY` when required by the live service)
- Bluesky public search for `#IMD` and related weather hashtags
- GDELT public web/news index
- NWDP/NWIC telemetry already used by RakshaNet
- RakshaNet citizen emergency reports stored in the existing database

## Integrity logic
- Source IDs prevent exact duplicates.
- Text similarity + time/location checks screen near-duplicates.
- Official records are marked as official; social/web/citizen records remain untrusted until corroborated or manually verified.
- Missing location/media/GPS fields are not invented.

## Admin verification
`PATCH /api/pulse/events/:id/verification` is admin-protected and accepts `VERIFIED_OFFICIAL`, `CORROBORATED`, `NEEDS_REVIEW`, `UNVERIFIED_CITIZEN`, or `LOW_CONFIDENCE`.

## Local setup
```powershell
npm install
npm start
```
Then open `http://localhost:5000/` and click **Advance Weather Pulse**.
