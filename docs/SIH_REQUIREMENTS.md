# RakshaNet vs SIH / MDoNER Problem Statement

This build maps the supplied problem statement to actual application capabilities.

| Requirement | Implementation status |
|---|---|
| Rainfall patterns | Live weather + Assam telemetry where available |
| Soil moisture sensors | Coordinate soil-moisture feed shown; direct ground sensor adapter remains optional |
| Satellite imagery | NASA GIBS imagery and layer controls |
| Terrain/slope | OpenTopoData/terrain context and slope proxy |
| Historical landslide records | RakshaNet NER/GSI/NASA inventory context |
| AI/ML high-risk zones | Deployed ONNX landslide model + live hazard grid |
| Possible landslide events | Location-based preliminary ML signal |
| Real-time alerts | Hazard-specific alert engine; official SACHET/IMD alerts are separated from derived signals |
| GIS | Live map, selected-location map, NER hazard overview, historical-event markers |
| Geo-tagged photo/video reports | Emergency reporting with location and evidence |
| Risk severity dashboard | Low / Watch / High / Extreme operational bands |
| Road connectivity | Routing and route-risk assessment |
| Weather-linked forecasts | Seven-day weather-linked hazard indices |
| Emergency response prioritisation | Existing emergency priority/admin flow |
| Multilingual notifications | English, Hindi, Assamese and Bengali alert templates |
| Low-network/offline | Service-worker cache + IndexedDB emergency queue + resync |
| IMD weather/warnings | Direct IMD integration when credentials/access are available; official SACHET fallback shown when IMD is unavailable |
| Satellite feeds | NASA GIBS |
| Sensor data | Government telemetry + coordinate weather |
| Automated SMS/app early warning | Web Push + SMS adapter + automated preference-driven scanner |
| Cloud architecture | Supabase/PostgreSQL |

## Important implementation distinction

RakshaNet intentionally does not label every contextual data source as a trained ML feature. The currently deployed landslide ONNX artifact has a fixed 16-feature weather/location input schema. Terrain, historical context, hydrology, satellite imagery, official warnings and telemetry are integrated into the surrounding intelligence and operational layers unless a retrained multimodal model is explicitly produced and validated.
