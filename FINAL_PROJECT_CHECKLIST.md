# RakshaNet Final Project Checklist

## User/account
- [x] Registration with permanent location
- [x] Six hazard preferences stored and editable in Profile
- [x] Persistent same-browser login experience
- [x] Delivery status displayed in Profile

## Intelligence
- [x] Current Location
- [x] Search Location
- [x] Monitoring Focus presets preserved
- [x] Exact coordinate analysis
- [x] Live weather/hydrology/terrain/history sources labeled
- [x] Official NDMA/SACHET alerts surfaced separately
- [x] NER live hazard overview
- [x] USGS earthquake activity overlay
- [x] Seven-day hazard forecast

## Early warning
- [x] Hazard-specific automation for heavy rain/flood/landslide/thunderstorm/extreme heat/cyclone
- [x] Web Push infrastructure
- [x] SMS provider adapter
- [x] Deduplication and delivery audit
- [x] Multilingual alert text foundation

## Resilience
- [x] Service worker core cache
- [x] Offline emergency report queue
- [x] Sync when connectivity returns

## Honest limitations
- [ ] Real SMS requires provider credentials and applicable Indian sender/template registration
- [ ] IMD direct API access may require provider/IP whitelisting
- [ ] Live satellite imagery is available; numerical satellite ML features require a configured processing feed
- [ ] Ground-movement sensor integration requires an actual geotechnical data source
- [ ] Landslide model percentage is preliminary and must not be presented as a guaranteed probability
