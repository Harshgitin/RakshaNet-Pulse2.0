# RakshaNet Early Warning Setup

RakshaNet's registration stores a user's permanent location and selected hazard preferences. The automated scanner evaluates each selected hazard independently at that stored location.

Supported alert preferences:
- Heavy Rainfall (`heavy_rain`)
- Flood (`flood`)
- Landslide (`landslide`)
- Thunderstorm (`thunderstorm`)
- Extreme Heat (`extreme_heat`)
- Cyclone (`cyclone`)

## 1. Browser/app push

Run:

```bash
npm install
npm run generate:vapid
```

Copy the generated values to `.env`:

```env
VAPID_SUBJECT=mailto:alerts@example.com
VAPID_PUBLIC_KEY=...
VAPID_PRIVATE_KEY=...
```

Start RakshaNet and sign in. In **My Profile**, select **Enable App Alerts** and allow browser notifications.

The service worker receives Web Push notifications in the background. Browsers still require user permission and an active service worker; localhost is treated as a trustworthy origin by modern browsers, while deployed push should use HTTPS.

## 2. SMS

Configure one approved SMS provider. The built-in adapter supports Twilio or a generic SMS webhook.

Twilio example:

```env
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_MESSAGING_SERVICE_SID=...
```

or:

```env
TWILIO_FROM_NUMBER=...
```

Then set a safe test number:

```env
ALERT_TEST_PHONE=+91XXXXXXXXXX
```

Do not expose provider credentials in browser code.

## 3. Automated monitoring

After provider configuration and controlled testing, enable:

```env
ALERT_AUTOMATION_ENABLED=true
ALERT_AUTOMATION_INTERVAL_MS=300000
ALERT_MIN_SCORE=60
ALERT_DEDUP_MINUTES=60
```

The scanner then:

```text
registered user
  -> permanent location
  -> selected hazard preferences
  -> live weather/hydrology/official-alert intelligence
  -> hazard-specific threshold
  -> duplicate check
  -> Web Push and/or SMS
  -> delivery audit
```

Official NDMA/SACHET alerts can bypass the derived-score threshold so a matched official warning is not suppressed just because a derived index is lower.

## 4. Testing without broadcasting

Keep automation disabled while testing:

```env
ALERT_AUTOMATION_ENABLED=false
```

Use the Profile **Test App Alert** button for Web Push after subscription. For SMS, configure `ALERT_TEST_PHONE` and call the controlled `/api/alerts/test` endpoint.

## 5. Truthfulness rules

- A RakshaNet-derived hazard index is not an official probability.
- A matched NDMA/SACHET/IMD warning is displayed separately and takes precedence.
- GloFAS guidance uses the largest river in approximately a 5 km area at the requested coordinate and can miss small/local waterways.
- Live satellite imagery is available, but the current deployed landslide model does not claim to consume raw satellite pixels.
- Real SMS delivery requires an approved provider account and the sender/template/telecom requirements applicable to the deployment country/operator.
