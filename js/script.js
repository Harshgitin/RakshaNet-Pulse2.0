/* =========================================================
   RAKSHANET - MAIN CLIENT ENGINE
   Live weather + risk + GIS + emergency + volunteers
   ========================================================= */


/* =========================================================
   BACKEND
========================================================= */

const RN_BACKEND = {

    base:
        window.location.protocol.startsWith("http")
            ? ""
            : "http://localhost:5000",

    async request(path, options = {}) {

        const url =
            `${RN_BACKEND.base}${path}`;

        const response =
            await fetch(url, options);

        if (!response.ok) {

            let message =
                `Backend HTTP ${response.status}`;

            try {

                const data =
                    await response.json();

                if (data?.error) {
                    message +=
                        `: ${data.error}`;
                }

                if (data?.details) {
                    message +=
                        ` (${data.details})`;
                }

            } catch (_) {
                // Non-JSON response
            }

            throw new Error(message);
        }

        return response.json();
    },

    available: false
};


async function detectBackend() {

    try {

        await RN_BACKEND.request(
            "/api/health"
        );

        RN_BACKEND.available = true;

        localStorage.setItem(
            "rakshanet_backend",
            "online"
        );

    } catch {

        RN_BACKEND.available = false;

        localStorage.setItem(
            "rakshanet_backend",
            "offline"
        );
    }
}


registerRakshaNetServiceWorker();
window.addEventListener('online', async () => {
    await detectBackend();
    await flushQueuedEmergencyReports();
});
window.addEventListener('load', async () => {
    await detectBackend();
    await flushQueuedEmergencyReports();
});
setInterval(() => {
    if (navigator.onLine) flushQueuedEmergencyReports();
}, 60 * 1000);


/* =========================================================
   CONFIG
========================================================= */

const RN_CONFIG = {

    imd: {

        districtWarnings:
            "https://mausam.imd.gov.in/api/warnings_district_api.php",

        districtRainfall:
            "https://mausam.imd.gov.in/api/districtwise_rainfall_api.php",

        rss:
            "https://mausam.imd.gov.in/imd_latest/contents/dist_nowcast_rss.php"

    },

    weather:
        "https://api.open-meteo.com/v1/forecast",

    geocode:
        "https://nominatim.openstreetmap.org/search",

    overpass:
        "https://overpass-api.de/api/interpreter",

    route:
        "https://router.project-osrm.org/route/v1"
};


/* =========================================================
   HELPERS
========================================================= */

const $ =
    id =>
        document.getElementById(id);


const clamp =
    (n, min, max) =>
        Math.max(
            min,
            Math.min(max, n)
        );


const sleep =
    ms =>
        new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    ms
                )
        );


function esc(value) {

    return String(
        value ?? ""
    ).replace(
        /[&<>'"]/g,
        c =>
            ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                "'": "&#39;",
                "\"": "&quot;"
            }[c])
    );
}


/* =========================================================
   GENERIC FETCH WITH TIMEOUT
========================================================= */

async function fetchJSON(
    url,
    options = {},
    timeout = 12000
) {

    const controller =
        new AbortController();

    const timer =
        setTimeout(
            () =>
                controller.abort(),
            timeout
        );

    try {

        const response =
            await fetch(
                url,
                {
                    ...options,

                    signal:
                        controller.signal,

                    cache:
                        "no-store"
                }
            );

        if (!response.ok) {

            throw new Error(
                `HTTP ${response.status}`
            );
        }

        return await response.json();

    } finally {

        clearTimeout(timer);
    }
}


/* =========================================================
   OFFLINE-FIRST SYNC
   Uses IndexedDB so emergency reports can survive reloads
   and synchronize when connectivity returns.
========================================================= */

const RN_OFFLINE_DB = 'rakshanet-offline-v2';
const RN_OFFLINE_STORE = 'reports';

function rnOfflineDb() {
    return new Promise((resolve, reject) => {
        if (!('indexedDB' in window)) {
            reject(new Error('IndexedDB is unavailable in this browser.'));
            return;
        }
        const request = indexedDB.open(RN_OFFLINE_DB, 1);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(RN_OFFLINE_STORE)) {
                db.createObjectStore(RN_OFFLINE_STORE, {
                    keyPath: 'id'
                });
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error('IndexedDB open failed.'));
    });
}

async function queueEmergencyReportOffline(report, file = null) {
    const db = await rnOfflineDb();
    const fileData = file ? await fileToDataURL(file) : null;
    const item = {
        id: report.id,
        queuedAt: new Date().toISOString(),
        report: {
            ...report,
            synced: false,
            backendId: null
        },
        evidence: fileData
    };

    await new Promise((resolve, reject) => {
        const tx = db.transaction(RN_OFFLINE_STORE, 'readwrite');
        tx.objectStore(RN_OFFLINE_STORE).put(item);
        tx.oncomplete = resolve;
        tx.onerror = () => reject(tx.error || new Error('Could not queue report.'));
    });
    db.close();

    return item;
}

async function readQueuedEmergencyReports() {
    const db = await rnOfflineDb();
    const items = await new Promise((resolve, reject) => {
        const tx = db.transaction(RN_OFFLINE_STORE, 'readonly');
        const request = tx.objectStore(RN_OFFLINE_STORE).getAll();
        request.onsuccess = () => resolve(request.result || []);
        request.onerror = () => reject(request.error || new Error('Could not read offline queue.'));
    });
    db.close();
    return items;
}

async function removeQueuedEmergencyReport(id) {
    const db = await rnOfflineDb();
    await new Promise((resolve, reject) => {
        const tx = db.transaction(RN_OFFLINE_STORE, 'readwrite');
        tx.objectStore(RN_OFFLINE_STORE).delete(id);
        tx.oncomplete = resolve;
        tx.onerror = () => reject(tx.error || new Error('Could not remove queued report.'));
    });
    db.close();
}

function dataUrlToBlob(dataUrl) {
    if (!dataUrl || typeof dataUrl !== 'string') return null;
    const [meta, encoded] = dataUrl.split(',');
    if (!encoded) return null;
    const mime = /data:([^;]+);/i.exec(meta)?.[1] || 'application/octet-stream';
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Blob([bytes], { type: mime });
}

async function flushQueuedEmergencyReports() {
    if (!navigator.onLine) return { synced: 0, remaining: null };
    let queued = [];
    try {
        queued = await readQueuedEmergencyReports();
    } catch (error) {
        console.warn('Offline queue unavailable:', error);
        return { synced: 0, remaining: null };
    }

    let synced = 0;
    for (const item of queued) {
        try {
            const fd = new FormData();
            const report = item.report || {};
            Object.entries(report).forEach(([key, value]) => {
                if (key === 'evidence') return;
                if (typeof value === 'object' && value !== null) {
                    fd.append(key, JSON.stringify(value));
                } else {
                    fd.append(key, String(value ?? ''));
                }
            });
            fd.set('synced', 'true');

            const evidenceBlob = dataUrlToBlob(item.evidence?.dataUrl);
            if (evidenceBlob) {
                fd.append(
                    'evidence',
                    evidenceBlob,
                    item.evidence?.name || 'offline-evidence'
                );
            }

            const saved = await RN_BACKEND.request('/api/reports', {
                method: 'POST',
                body: fd
            });
            await removeQueuedEmergencyReport(item.id);
            synced++;

            try {
                const localReports = JSON.parse(
                    localStorage.getItem('rakshanet_reports') || '[]'
                );
                localReports.unshift({
                    ...report,
                    synced: true,
                    backendId: saved.id,
                    evidence: saved.evidence || null
                });
                localStorage.setItem(
                    'rakshanet_reports',
                    JSON.stringify(localReports.slice(0, 100))
                );
            } catch (_) {}
        } catch (error) {
            console.warn('Queued emergency report remains pending:', error.message);
            break;
        }
    }

    return { synced, remaining: Math.max(0, queued.length - synced) };
}

function registerRakshaNetServiceWorker() {
    if (!('serviceWorker' in navigator)) return;
    if (!/^https?:$/.test(window.location.protocol)) return;

    navigator.serviceWorker.register(
        `${window.location.origin}/sw.js`,
        { scope: '/' }
    ).catch(error => {
        console.warn('RakshaNet service worker registration failed:', error.message);
    });
}


/* =========================================================
   GEOCODING
========================================================= */

async function geocodePlace(query) {

    const url =
        new URL(
            RN_CONFIG.geocode
        );

    url.search =
        new URLSearchParams({

            q:
                `${query}, India`,

            format:
                "jsonv2",

            limit:
                "1",

            addressdetails:
                "1"

        });


    const data =
        await fetchJSON(
            url.toString()
        );


    if (!data.length) {

        throw new Error(
            "Place not found. Try a city, district or state in India."
        );
    }


    return {

        lat:
            Number(
                data[0].lat
            ),

        lon:
            Number(
                data[0].lon
            ),

        name:
            data[0].display_name
    };
}


/* =========================================================
   WEATHER
========================================================= */

async function getWeather(
    lat,
    lon
) {

    const url =
        new URL(
            RN_CONFIG.weather
        );

    url.search =
        new URLSearchParams({

            latitude:
                String(lat),

            longitude:
                String(lon),

            timezone:
                "auto",

            current:
                "temperature_2m,relative_humidity_2m," +
                "apparent_temperature,precipitation,rain," +
                "cloud_cover,visibility,is_day," +
                "weather_code,wind_speed_10m,wind_gusts_10m",

            hourly:
                "precipitation,rain,showers,weather_code," +
                "wind_speed_10m,wind_gusts_10m,cloud_cover,temperature_2m",

            daily:
                "precipitation_sum,rain_sum,precipitation_hours," +
                "precipitation_probability_max,weather_code,temperature_2m_max," +
                "temperature_2m_min,wind_speed_10m_max",

            past_days:
                "7",

            forecast_days:
                "7"

        });

    return await fetchJSON(
        url.toString()
    );
}


function summarizeWeather(data) {

    const current = data.current || {};
    const hourly = data.hourly || {};
    const daily = data.daily || {};

    const hourlyTimes = Array.isArray(hourly.time) ? hourly.time : [];
    const nowReference = current.time ? Date.parse(current.time) : Date.now();

    let currentIndex = 0;

    if (hourlyTimes.length) {
        let bestDiff = Infinity;
        hourlyTimes.forEach((value, index) => {
            const parsed = Date.parse(value);
            if (!Number.isFinite(parsed)) return;
            const diff = Math.abs(parsed - nowReference);
            if (diff < bestDiff) {
                bestDiff = diff;
                currentIndex = index;
            }
        });
    }

    const start24 = Math.max(0, currentIndex - 23);
    const end24 = Math.min(hourlyTimes.length, currentIndex + 1);
    const futureEnd24 = Math.min(hourlyTimes.length, currentIndex + 25);

    const sumWindow = (values, start, end) =>
        (Array.isArray(values) ? values : [])
            .slice(start, end)
            .reduce((sum, value) => sum + Number(value || 0), 0);

    const maxWindow = (values, start, end) => {
        const clean = (Array.isArray(values) ? values : [])
            .slice(start, end)
            .map(Number)
            .filter(Number.isFinite);
        return clean.length ? Math.max(...clean) : 0;
    };

    const precip24 = sumWindow(hourly.precipitation, start24, end24);
    const rain24 = sumWindow(hourly.rain, start24, end24);
    const next24Precip = sumWindow(hourly.precipitation, currentIndex, futureEnd24);
    const maxWind = maxWindow(hourly.wind_speed_10m, start24, end24);
    const maxWindNext24 = maxWindow(hourly.wind_speed_10m, currentIndex, futureEnd24);

    const maxTemp = maxWindow(daily.temperature_2m_max, 0, 3) || Number(current.temperature_2m || 0);
    const maxProb = Math.max(
        ...((daily.precipitation_probability_max || [])
            .slice(0, 3)
            .map(Number)
            .filter(Number.isFinite)),
        0
    );
    const rain3d = (daily.precipitation_sum || [])
        .slice(0, 3)
        .reduce((a, b) => a + Number(b || 0), 0);

    return {
        temp: Number(current.temperature_2m || 0),
        apparentTemp: Number(current.apparent_temperature ?? current.temperature_2m ?? 0),
        humidity: Number(current.relative_humidity_2m || 0),
        cloudCover: Number(current.cloud_cover ?? 0),
        visibility: Number(current.visibility ?? 0),
        rainNow: Number(current.rain || 0),
        precipitationNow: Number(current.precipitation || 0),
        precip24,
        rain24,
        next24Precip,
        currentWind: Number(current.wind_speed_10m || 0),
        currentWindGust: Number(current.wind_gusts_10m || 0),
        maxWind,
        maxWindNext24,
        maxTemp,
        maxProb,
        rain3d,
        code: Number(current.weather_code || 0),
        isDay: Number(current.is_day ?? 1),
        updatedAt: current.time || new Date().toISOString()
    };
}



/* =========================================================
   LANDSLIDE FEATURES
========================================================= */

function buildLandslideFeatures(
    data,
    lat,
    lon
) {
    const current = data.current || {};
    const hourly = data.hourly || {};
    const daily = data.daily || {};
    const times = Array.isArray(hourly.time) ? hourly.time : [];
    const nowReference = current.time ? Date.parse(current.time) : Date.now();

    const series = {
        rain: Array.isArray(hourly.rain) ? hourly.rain : [],
        precipitation: Array.isArray(hourly.precipitation) ? hourly.precipitation : [],
        temperature: Array.isArray(hourly.temperature_2m) ? hourly.temperature_2m : [],
        wind: Array.isArray(hourly.wind_speed_10m) ? hourly.wind_speed_10m : []
    };

    function valuesWithin(hours, source) {
        const cutoff = nowReference - hours * 60 * 60 * 1000;
        const values = [];
        for (let i = 0; i < times.length; i++) {
            const parsed = Date.parse(times[i]);
            if (!Number.isFinite(parsed) || parsed > nowReference || parsed < cutoff) continue;
            const value = Number(source[i]);
            if (Number.isFinite(value)) values.push(value);
        }
        return values;
    }

    const sum = values => values.reduce((total, value) => total + value, 0);
    const mean = values => values.length ? sum(values) / values.length : null;
    const max = values => values.length ? Math.max(...values) : null;
    const min = values => values.length ? Math.min(...values) : null;
    const rain1 = valuesWithin(24, series.rain);
    const rain3 = valuesWithin(72, series.rain);
    const rain7 = valuesWithin(168, series.rain);
    const precip1 = valuesWithin(24, series.precipitation);
    const precip3 = valuesWithin(72, series.precipitation);
    const precip7 = valuesWithin(168, series.precipitation);
    const temp1 = valuesWithin(24, series.temperature);
    const wind1 = valuesWithin(24, series.wind);

    const fallbackDailyRain = (daily.rain_sum || []).map(Number).filter(Number.isFinite);
    const fallbackDailyPrecip = (daily.precipitation_sum || []).map(Number).filter(Number.isFinite);
    const fallbackDailyHours = (daily.precipitation_hours || []).map(Number).filter(Number.isFinite);

    return {
        rain_1d: Number(sum(rain1.length ? rain1 : fallbackDailyRain.slice(-1)).toFixed(3)),
        rain_3d: Number(sum(rain3.length ? rain3 : fallbackDailyRain.slice(-3)).toFixed(3)),
        rain_7d: Number(sum(rain7.length ? rain7 : fallbackDailyRain.slice(-7)).toFixed(3)),
        precip_1d: Number(sum(precip1.length ? precip1 : fallbackDailyPrecip.slice(-1)).toFixed(3)),
        precip_3d: Number(sum(precip3.length ? precip3 : fallbackDailyPrecip.slice(-3)).toFixed(3)),
        precip_7d: Number(sum(precip7.length ? precip7 : fallbackDailyPrecip.slice(-7)).toFixed(3)),
        precip_hours_1d: precip1.length ? precip1.filter(value => value > 0).length : Number(fallbackDailyHours.slice(-1)[0] || 0),
        precip_hours_3d: precip3.length ? precip3.filter(value => value > 0).length : Number(sum(fallbackDailyHours.slice(-3))),
        precip_hours_7d: precip7.length ? precip7.filter(value => value > 0).length : Number(sum(fallbackDailyHours.slice(-7))),
        temp_mean: mean(temp1) ?? Number(current.temperature_2m || 0),
        temp_max: max(temp1) ?? Number(current.temperature_2m || 0),
        temp_min: min(temp1) ?? Number(current.temperature_2m || 0),
        wind_max: max(wind1) ?? Number(current.wind_speed_10m || 0),
        latitude: Number(lat),
        longitude: Number(lon),
        month: new Date(nowReference).getMonth() + 1
    };
}



/* =========================================================
   ML
========================================================= */

async function predictLandslideML(
    weatherData,
    lat,
    lon
) {

    const features =
        buildLandslideFeatures(
            weatherData,
            lat,
            lon
        );


    const response =
        await RN_BACKEND.request(
            "/api/predict/landslide",
            {

                method:
                    "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body:
                    JSON.stringify(
                        features
                    )
            }
        );


    return {

        ...response,

        features
    };
}



/* =========================================================
   ML GRID RISK
========================================================= */

async function rnFetchMLRiskGrid(points) {
    if (!Array.isArray(points) || !points.length) return null;
    try {
        return await RN_BACKEND.request('/api/intelligence/grid', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points })
        });
    } catch (error) {
        console.warn('ML risk grid unavailable; using local risk engine:', error.message);
        return null;
    }
}

/* =========================================================
   RISK
========================================================= */

function weatherCodeText(code) {

    if (
        [95, 96, 99]
            .includes(code)
    ) {
        return "Thunderstorm";
    }

    if (code >= 80)
        return "Rain showers";

    if (code >= 61)
        return "Rain";

    if (code >= 51)
        return "Drizzle";

    if (code >= 45)
        return "Fog";

    if (code >= 1)
        return "Cloudy";

    return "Clear";
}


function scoreRisk(w, hazard = "auto") {
    const flood = clamp(
        (Number(w.floodIndex ?? w.precip24 ?? 0) / 100) * 100,
        0, 100
    );
    const landslide = clamp(
        (Number(w.landslideIndex ?? w.rain3d ?? 0) / 220) * 50 +
        (Number(w.precip24 ?? 0) / 120) * 35 +
        (Number(w.maxWind ?? 0) / 80) * 15,
        0, 100
    );
    const heat = clamp(((Number(w.maxTemp ?? 0) - 32) / 10) * 100, 0, 100);
    const severe = clamp(
        (Number(w.maxWind ?? 0) / 90) * 55 +
        (Number(w.maxProb ?? 0) / 100) * 20 +
        ([95, 96, 99].includes(Number(w.code)) ? 35 : 0),
        0, 100
    );
    let score = Math.max(flood, landslide, heat, severe);
    if (hazard === 'flood') score = flood;
    if (hazard === 'landslide') score = landslide;
    if (hazard === 'heatwave' || hazard === 'extreme_heat') score = heat;
    if (hazard === 'severe' || hazard === 'thunderstorm') score = severe;
    return {
        score: Math.round(score),
        flood: Math.round(flood),
        landslide: Math.round(landslide),
        heat: Math.round(heat),
        severe: Math.round(severe)
    };
}


function riskBand(score) {

    if (score >= 80) {

        return {

            label:
                "EXTREME",

            cls:
                "rn-risk-extreme",

            priority:
                "P1 Critical"
        };
    }


    if (score >= 60) {

        return {

            label:
                "HIGH",

            cls:
                "rn-risk-high",

            priority:
                "P2 High"
        };
    }


    if (score >= 35) {

        return {

            label:
                "WATCH",

            cls:
                "rn-risk-medium",

            priority:
                "P3 Watch"
        };
    }


    return {

        label:
            "LOW",

        cls:
            "rn-risk-low",

        priority:
            "P4 Normal"
    };
}


function riskColor(score) {

    if (score >= 80)
        return "#ef4444";

    if (score >= 60)
        return "#f97316";

    if (score >= 35)
        return "#f59e0b";

    return "#22c55e";
}


/* =========================================================
   PREDICTION RENDERING
========================================================= */

function renderPrediction(
    target,
    place,
    weather,
    hazard,
    mlResult = null
) {

    let score;
    let band;
    let modelLabel;


    if (
        mlResult &&
        mlResult.prediction
    ) {

        score =
            Number(
                mlResult.prediction.percentage || 0
            );

        band =
            riskBand(score);

        modelLabel =
            "🧠 RakshaNet AI/ML Landslide Model";

    } else {

        const ruleRisk =
            scoreRisk(
                weather,
                hazard
            );

        score =
            ruleRisk.score;

        band =
            riskBand(score);

        modelLabel =
            "Live meteorological risk engine";
    }


    const ruleScores =
        scoreRisk(
            weather,
            hazard
        );


    const weatherText =
        weatherCodeText(
            weather.code
        );


    const mlProbability =
        mlResult?.prediction
            ?.landslideProbability;


    const probabilityPercent =
        mlProbability !== undefined &&
        mlProbability !== null
            ? Math.round(
                Number(
                    mlProbability
                ) * 100
            )
            : null;


    target.innerHTML = `

        <div class="rn-risk-score">

            <div
                class="rn-score-ring"
                style="
                    background:
                    conic-gradient(
                        ${riskColor(score)}
                        ${score * 3.6}deg,
                        rgba(255,255,255,.06) 0deg
                    )
                "
            >

                <strong>
                    ${score}
                </strong>

            </div>


            <div>

                <div
                    class="rn-risk-badge ${band.cls}"
                >
                    ${band.label}
                    ·
                    ${band.priority}
                </div>


                <h3
                    style="margin-top:9px"
                >
                    ${esc(place.name)}
                </h3>


                <div class="rn-muted">
                    Live condition:
                    ${esc(weatherText)}
                </div>


                <div
                    class="rn-small"
                    style="margin-top:6px"
                >
                    ${modelLabel}
                </div>

            </div>

        </div>


        <div class="rn-weather-grid">

            <div class="rn-metric">
                <span>
                    Temperature
                </span>

                <strong>
                    ${weather.temp.toFixed(1)}°C
                </strong>
            </div>


            <div class="rn-metric">
                <span>
                    Humidity
                </span>

                <strong>
                    ${weather.humidity}%
                </strong>
            </div>


            <div class="rn-metric">
                <span>
                    24h Rain
                </span>

                <strong>
                    ${weather.precip24.toFixed(1)} mm
                </strong>
            </div>


            <div class="rn-metric">
                <span>
                    Max Wind
                </span>

                <strong>
                    ${weather.maxWind.toFixed(0)} km/h
                </strong>
            </div>

        </div>


        ${
            mlResult
                ? `

                <div
                    class="rn-result-list"
                    style="margin-top:16px;"
                >

                    <div class="rn-result-item">

                        <strong>
                            🧠 Landslide Probability
                        </strong>

                        <span class="rn-result-meta">
                            ${probabilityPercent}%
                        </span>

                    </div>


                    <div class="rn-result-item">

                        <strong>
                            ⚠️ AI Risk Level
                        </strong>

                        <span class="rn-result-meta">
                            ${band.label}
                            ·
                            ${band.priority}
                        </span>

                    </div>


                    <div class="rn-result-item">

                        <strong>
                            📍 Coordinates
                        </strong>

                        <span class="rn-result-meta">
                            ${Number(place.lat).toFixed(4)},
                            ${Number(place.lon).toFixed(4)}
                        </span>

                    </div>

                </div>

                `
                : ""
        }


        <div
            class="rn-factor-list"
            style="margin-top:16px;"
        >

            <div class="rn-factor">

                <span>
                    Flood
                </span>

                <div class="rn-bar">

                    <span
                        style="
                            width:${ruleScores.flood}%
                        "
                    ></span>

                </div>

                <strong>
                    ${ruleScores.flood}
                </strong>

            </div>


            <div class="rn-factor">

                <span>
                    Landslide
                </span>

                <div class="rn-bar">

                    <span
                        style="
                            width:${score}%
                        "
                    ></span>

                </div>

                <strong>
                    ${score}
                </strong>

            </div>


            <div class="rn-factor">

                <span>
                    Heatwave
                </span>

                <div class="rn-bar">

                    <span
                        style="
                            width:${ruleScores.heat}%
                        "
                    ></span>

                </div>

                <strong>
                    ${ruleScores.heat}
                </strong>

            </div>


            <div class="rn-factor">

                <span>
                    Severe weather
                </span>

                <div class="rn-bar">

                    <span
                        style="
                            width:${ruleScores.severe}%
                        "
                    ></span>

                </div>

                <strong>
                    ${ruleScores.severe}
                </strong>

            </div>

        </div>


        <div class="rn-source-row">

            <span class="rn-source">
                Live Open-Meteo weather
            </span>


            ${
                mlResult
                    ? `
                    <span class="rn-source">
                        RakshaNet ML model
                    </span>
                    `
                    : ""
            }


            <span class="rn-source">
                OpenStreetMap geocoding
            </span>


            <span class="rn-source">
                IMD official warnings should be checked
            </span>

        </div>


        <div class="rn-footer-note">

            ${
                mlResult
                    ? `
                    RakshaNet's preliminary AI/ML
                    landslide risk estimate is based on
                    historical landslide-weather patterns
                    and current/recent meteorological
                    conditions.
                    `
                    : `
                    This is a preliminary
                    data-driven risk estimate.
                    `
            }

            It is not an official IMD warning.
            Follow official warnings and local
            authorities for emergency decisions.

        </div>
    `;


    return {

        score,

        ...ruleScores
    };
}


/* =========================================================
   ANALYZE PLACE
========================================================= */

async function getLiveRegionAnalysis(place, hazard = 'auto') {
    const url = new URL('/api/region-analysis', window.location.origin);
    url.search = new URLSearchParams({
        lat: String(place.lat),
        lon: String(place.lon),
        region: place.name || 'Selected area',
        hazard
    });
    const response = await fetch(url.toString(), { headers: { Accept: 'application/json' } });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) throw new Error(data.error || `Live region analysis HTTP ${response.status}`);
    return data;
}

function renderLiveRegionAnalysis(target, place, analysis, hazard) {
    const scores = analysis?.hazards?.scores || {};
    const selectedScore = Number(analysis.selectedScore ?? 0);
    const selectedBand = String(analysis.selectedBand || riskBand(selectedScore).label);
    const band = riskBand(selectedScore);
    const weather = analysis?.weather?.current || {};
    const daily = analysis?.weather?.daily || {};
    const next24Rain = Number(analysis?.hazards?.rawWeatherSignals?.next24Precip ?? Math.max(...((daily.precipitation_sum || []).map(Number).filter(Number.isFinite).slice(0,1)), 0));
    const next72Rain = Number(analysis?.hazards?.rawWeatherSignals?.next72Precip ?? (daily.precipitation_sum || []).slice(0,3).reduce((a,b)=>a+Number(b||0),0));
    const officialAlerts = Object.values(analysis?.officialHazards || {});
    const selectedOfficial = analysis?.selectedOfficialAlert;
    const hydrology = analysis?.hydrology || {};
    const factor = key => `<div class="rn-factor"><span>${esc(analysis.hazards.labels?.[key] || key)}</span><div class="rn-bar"><span style="width:${Math.round(Number(scores[key] || 0))}%"></span></div><strong>${Math.round(Number(scores[key] || 0))}</strong></div>`;
    const officialBlock = selectedOfficial ? `
        <div class="rn-alert-banner danger" style="margin-top:14px;">
            <strong>📢 Official alert matched</strong><br>
            ${esc(selectedOfficial.title || 'Official disaster alert')}<br>
            <small>${esc(selectedOfficial.area || '')} · severity ${esc(selectedOfficial.severity || 'not supplied')} · matched by ${esc(selectedOfficial.locationMatch || 'location')}</small>
            ${selectedOfficial.link ? `<br><a href="${esc(selectedOfficial.link)}" target="_blank" rel="noopener noreferrer">Open official alert source →</a>` : ''}
        </div>` : `
        <div class="rn-footer-note" style="margin-top:14px;">No matching official NDMA/SACHET alert was found for this location at this check time. This does not prove that no local incident exists.</div>`;
    target.innerHTML = `
        <div class="rn-risk-score">
            <div class="rn-score-ring" style="background:conic-gradient(${riskColor(selectedScore)} ${selectedScore * 3.6}deg,rgba(255,255,255,.06) 0deg)"><strong>${selectedScore}</strong></div>
            <div>
                <div class="rn-risk-badge ${band.cls}">${selectedBand} · ${band.priority}</div>
                <h3 style="margin-top:9px">${esc(analysis?.location?.displayName || place.name)}</h3>
                <div class="rn-muted">Live condition: ${esc(weatherCodeText(Number(weather.weather_code || 0)))}</div>
                <div class="rn-small" style="margin-top:6px">${analysis.selectedBasis === 'OFFICIAL_NDMA_SACHET' ? 'Official alert severity selected' : 'Derived live-data index selected'} · checked ${esc(new Date(analysis.checkedAt).toLocaleString())}</div>
            </div>
        </div>
        ${officialBlock}
        <div class="rn-weather-grid">
            <div class="rn-metric"><span>Temperature</span><strong>${Number(weather.temperature_2m ?? 0).toFixed(1)}°C</strong></div>
            <div class="rn-metric"><span>Humidity</span><strong>${Number(weather.relative_humidity_2m ?? 0).toFixed(0)}%</strong></div>
            <div class="rn-metric"><span>Next 24h rain</span><strong>${next24Rain.toFixed(1)} mm</strong></div>
            <div class="rn-metric"><span>Next 72h rain</span><strong>${next72Rain.toFixed(1)} mm</strong></div>
        </div>
        <div class="rn-result-list" style="margin-top:16px">
            <div class="rn-result-item"><strong>🧠 Landslide ML output</strong><span class="rn-result-meta">${analysis.landslide?.mlProbabilityPercent ?? '—'}% · preliminary model</span></div>
            <div class="rn-result-item"><strong>🌊 Hydrology guidance</strong><span class="rn-result-meta">${hydrology.available ? `${Number(hydrology.riverDischargeM3s ?? 0).toFixed(0)} m³/s · forecast peak ${Number(hydrology.forecastPeak3dM3s ?? 0).toFixed(0)} m³/s` : 'Unavailable'}</span></div>
            <div class="rn-result-item"><strong>📢 Official alerts</strong><span class="rn-result-meta">${officialAlerts.length ? `${officialAlerts.length} matched` : 'None matched'}</span></div>
            <div class="rn-result-item"><strong>⛰️ Terrain</strong><span class="rn-result-meta">${analysis.terrain?.available ? `${analysis.terrain.elevationMeters} m · slope ${analysis.terrain.slopePercent}%` : 'Unavailable'}</span></div>
            <div class="rn-result-item"><strong>📍 Coordinates</strong><span class="rn-result-meta">${Number(place.lat).toFixed(4)}, ${Number(place.lon).toFixed(4)}</span></div>
        </div>
        <div class="rn-factor-list" style="margin-top:16px">
            ${factor('flood')}${factor('heavy_rain')}${factor('landslide')}${factor('thunderstorm')}${factor('extreme_heat')}${factor('cyclone')}
        </div>
        <div class="rn-source-row">
            <span class="rn-source">Open-Meteo live weather</span>
            <span class="rn-source">GloFAS river-discharge guidance</span>
            <span class="rn-source">NDMA/SACHET official alerts</span>
            <span class="rn-source">IMD warnings when reachable</span>
            <span class="rn-source">RakshaNet historical inventory</span>
        </div>
        <div class="rn-footer-note">
            Flood, rainfall, heat, storm and cyclone numbers are <strong>derived decision-support indices, not probabilities</strong>. The landslide percentage is the deployed ML model output. A matched official alert is displayed separately and can raise the operational hazard index to the published alert severity; it is not converted into a probability. Official government warnings take precedence.
        </div>
    `;
}

async function analyzePlace(
    place,
    hazard,
    target
) {
    target.innerHTML = `<div class="rn-muted">Fetching live weather, hydrology, official warnings where available, terrain context and AI analysis for ${esc(place.name)}…</div>`;
    try {
        const analysis = await getLiveRegionAnalysis(place, hazard);
        renderLiveRegionAnalysis(target, place, analysis, hazard);
        const scores = analysis?.hazards?.scores || {};
        const selectedScore = Number(analysis?.selectedScore ?? 0);
        const result = {
            place,
            summary: {
                temp: Number(analysis?.weather?.current?.temperature_2m || 0),
                humidity: Number(analysis?.weather?.current?.relative_humidity_2m || 0),
                precip24: Number(analysis?.weather?.current?.precipitation || 0),
                maxWind: Number(analysis?.weather?.current?.wind_speed_10m || 0),
                maxTemp: Number(Math.max(...((analysis?.weather?.daily?.temperature_2m_max || []).map(Number).filter(Number.isFinite).slice(0,3)), Number(analysis?.weather?.current?.temperature_2m || 0))),
                maxProb: Number(Math.max(...((analysis?.weather?.daily?.precipitation_probability_max || []).map(Number).filter(Number.isFinite).slice(0,3)),0)),
                rain3d: Number((analysis?.weather?.daily?.rain_sum || []).slice(0,3).reduce((a,b)=>a+Number(b||0),0)),
                code: Number(analysis?.weather?.current?.weather_code || 0)
            },
            risk: { score: selectedScore, flood: Number(scores.flood||0), landslide: Number(scores.landslide||0), heat: Number(scores.extreme_heat||0), severe: Number(scores.severe||0) },
            ml: analysis?.landslide?.mlProbabilityPercent != null ? { prediction: { percentage: Number(analysis.landslide.mlProbabilityPercent), landslideProbability: Number(analysis.landslide.mlProbabilityPercent)/100, risk: analysis.landslide.mlRisk } } : null,
            analysis
        };
        localStorage.setItem('rakshanet_last_prediction', JSON.stringify({ savedAt: new Date().toISOString(), ...result }));
        localStorage.setItem('rakshanet_last_intelligence', JSON.stringify({ savedAt: new Date().toISOString(), place, hazard, intelligence: analysis }));
        localStorage.setItem('rakshanet_last_risk', String(selectedScore));
        return result;
    } catch (e) {
        console.error('RakshaNet live region analysis error:', e);
        target.innerHTML = `<div class="rn-alert-banner danger">Unable to load live multi-hazard analysis. ${esc(e.message)}</div>`;
        throw e;
    }
}



/* =========================================================
   ANALYZE CURRENT LOCATION
========================================================= */

async function analyzeFromLocation(
    target,
    hazard = "auto"
) {

    return new Promise(
        (
            resolve,
            reject
        ) => {

            RakshaLocation.getLocation(

                async loc => {

                    try {

                        const place = {

                            lat:
                                loc.latitude,

                            lon:
                                loc.longitude,

                            name:
                                `Current location (${loc.latitude.toFixed(4)}, ${loc.longitude.toFixed(4)})`
                        };


                        resolve(
                            await analyzePlace(
                                place,
                                hazard,
                                target
                            )
                        );


                    } catch (e) {

                        reject(e);
                    }

                },

                reject
            );
        }
    );
}


/* =========================================================
   HOME
========================================================= */

function initHome() {

    const status =
        $("homeLiveStatus");


    if (!status)
        return;


    const stamp =
        $("homeLiveTime");


    status.textContent =
        "LIVE";


    status.style.color =
        "#25d6a2";


    stamp.textContent =
        `Weather engine ready · ${new Date().toLocaleTimeString()}`;


    const analyzeBtn =
        $("analyzeRegionBtn");


    const myBtn =
        $("analyzeMyLocationBtn");


    const target =
        $("predictionResult");


    const hazardSelect = $("predictionHazard");
    if (hazardSelect) {
        hazardSelect.innerHTML = `
            <option value="auto">Auto detect</option>
            <option value="flood">Flood</option>
            <option value="heavy_rain">Heavy rainfall</option>
            <option value="landslide">Landslide</option>
            <option value="thunderstorm">Thunderstorm</option>
            <option value="extreme_heat">Extreme heat</option>
            <option value="cyclone">Cyclone</option>
            <option value="severe">Severe weather</option>`;
    }
    const hazard = () => hazardSelect?.value || 'auto';


    analyzeBtn?.addEventListener(
        "click",
        async () => {

            const q =
                $("predictionPlace")
                    ?.value
                    .trim();


            if (!q) {

                alert(
                    "Enter a place such as Guwahati, Assam."
                );

                return;
            }


            analyzeBtn.disabled =
                true;


            try {

                const place =
                    await geocodePlace(
                        q
                    );


                await analyzePlace(
                    place,
                    hazard(),
                    target
                );


            } catch (e) {

                target.innerHTML = `

                    <div
                        class="rn-alert-banner danger"
                    >
                        ${esc(e.message)}
                    </div>

                `;

            } finally {

                analyzeBtn.disabled =
                    false;
            }
        }
    );


    myBtn?.addEventListener(
        "click",
        async () => {

            myBtn.disabled =
                true;


            try {

                await analyzeFromLocation(
                    target,
                    hazard()
                );


            } catch (e) {

                target.innerHTML = `

                    <div
                        class="rn-alert-banner danger"
                    >
                        ${esc(e.message)}
                    </div>

                `;

            } finally {

                myBtn.disabled =
                    false;
            }
        }
    );
}


/* =========================================================
   EMERGENCY PRIORITY
========================================================= */

function calculatePriority(
    report,
    liveRisk = 0
) {

    let score =
        0;


    const typeWeights = {

        "Building Collapse":
            35,

        Fire:
            32,

        Landslide:
            30,

        Flood:
            28,

        "Road Blockage":
            15,

        Other:
            10
    };


    score +=
        typeWeights[
            report.incidentType
        ] || 10;


    score +=
        clamp(
            Number(
                report.peopleAffected
            ) * 2,
            0,
            20
        );


    score +=
        clamp(
            Number(
                report.injured
            ) * 8,
            0,
            24
        );


    score +=
        clamp(
            Number(
                report.trapped
            ) * 10,
            0,
            30
        );


    score +=
        clamp(
            liveRisk * 0.25,
            0,
            25
        );


    if (score >= 85) {

        return {

            label:
                "P1 Critical",

            score:
                Math.round(score),

            cls:
                "rn-risk-extreme"
        };
    }


    if (score >= 60) {

        return {

            label:
                "P2 High",

            score:
                Math.round(score),

            cls:
                "rn-risk-high"
        };
    }


    if (score >= 35) {

        return {

            label:
                "P3 Watch",

            score:
                Math.round(score),

            cls:
                "rn-risk-medium"
        };
    }


    return {

        label:
            "P4 Normal",

        score:
            Math.round(score),

        cls:
            "rn-risk-low"
    };
}


/* =========================================================
   FILE DATA URL
========================================================= */

function fileToDataURL(file) {

    return new Promise(
        (resolve, reject) => {

            if (!file)
                return resolve(null);


            if (
                file.size >
                4 * 1024 * 1024
            ) {

                return reject(
                    new Error(
                        "Selected file is over 4 MB for the offline browser queue."
                    )
                );
            }


            const reader =
                new FileReader();


            reader.onload =
                () =>
                    resolve({

                        name:
                            file.name,

                        type:
                            file.type,

                        size:
                            file.size,

                        dataUrl:
                            reader.result
                    });


            reader.onerror =
                reject;


            reader.readAsDataURL(
                file
            );
        }
    );
}


/* =========================================================
   EMERGENCY PAGE
========================================================= */

function initEmergencyPage() {

    const form =
        $("emergencyForm");


    if (!form)
        return;


    const submitButton =
        $("submitReportBtn");


    const incidentButtons =
        document.querySelectorAll(
            ".incident-option"
        );


    incidentButtons.forEach(
        btn => {

            btn.addEventListener(
                "click",
                () => {

                    incidentButtons.forEach(
                        b =>
                            b.classList.remove(
                                "selected",
                                "active"
                            )
                    );


                    btn.classList.add(
                        "selected",
                        "active"
                    );


                    $("incidentType").value =
                        btn.dataset.type;


                    updatePriorityPreview();
                }
            );
        }
    );


    $("locationBtn")?.addEventListener(
        "click",
        () => {

            $("locationStatus").textContent =
                "Detecting location…";


            $("locationText").textContent =
                "Please allow GPS access.";


            RakshaLocation.getLocation(

                loc => {

                    $("latitude").value =
                        loc.latitude;


                    $("longitude").value =
                        loc.longitude;


                    $("locationStatus").textContent =
                        "Location detected successfully";


                    $("locationText").textContent =
                        `Lat ${loc.latitude.toFixed(6)} · ` +
                        `Lng ${loc.longitude.toFixed(6)} · ` +
                        `±${Math.round(loc.accuracy)}m`;

                },


                msg => {

                    $("locationStatus").textContent =
                        "Location unavailable";


                    $("locationText").textContent =
                        msg;
                }
            );
        }
    );


    [
        "peopleCount",
        "injuredCount",
        "trappedCount"
    ].forEach(
        id => {

            $(id)?.addEventListener(
                "input",
                updatePriorityPreview
            );
        }
    );


    $("evidence")?.addEventListener(
        "change",
        () => {

            const file =
                $("evidence")
                    .files?.[0];


            const box =
                $("evidencePreview");


            if (!box)
                return;


            if (!file) {

                box.innerHTML =
                    "";

                return;
            }


            if (
                file.size >
                15 * 1024 * 1024
            ) {

                box.innerHTML = `

                    <div
                        class="rn-alert-banner danger"
                    >
                        Selected file is larger than 15 MB.
                    </div>

                `;

                return;
            }


            if (
                file.type.startsWith(
                    "image/"
                )
            ) {

                const url =
                    URL.createObjectURL(
                        file
                    );


                box.innerHTML = `

                    <img
                        class="rn-upload-preview"
                        src="${url}"
                        alt="Selected evidence"
                    >

                `;

            } else {

                box.innerHTML = `

                    <div class="rn-small">

                        📎 Selected file:
                        ${esc(file.name)}
                        ·
                        ${(file.size / 1024).toFixed(0)}
                        KB

                    </div>

                `;
            }
        }
    );


    form.addEventListener(
        "submit",
        async event => {

            event.preventDefault();


            if (
                submitButton?.disabled
            ) {
                return;
            }


            const report = {

                id:
                    `RN-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,

                reporterName:
                    $("reporterName")
                        ?.value
                        .trim() ||
                    "Anonymous",

                contact:
                    $("contact")
                        ?.value
                        .trim() ||
                    "",

                incidentType:
                    $("incidentType")
                        ?.value ||
                    "",

                latitude:
                    Number(
                        $("latitude")
                            ?.value
                    ),

                longitude:
                    Number(
                        $("longitude")
                            ?.value
                    ),

                peopleAffected:
                    Number(
                        $("peopleCount")
                            ?.value ||
                        0
                    ),

                injured:
                    Number(
                        $("injuredCount")
                            ?.value ||
                        0
                    ),

                trapped:
                    Number(
                        $("trappedCount")
                            ?.value ||
                        0
                    ),

                description:
                    $("description")
                        ?.value
                        .trim() ||
                    "",

                timestamp:
                    new Date()
                        .toISOString(),

                status:
                    "NEW",

                synced:
                    false
            };


            if (
                !report.incidentType
            ) {

                alert(
                    "Please select the emergency type."
                );

                return;
            }


            if (
                !Number.isFinite(
                    report.latitude
                ) ||
                !Number.isFinite(
                    report.longitude
                )
            ) {

                alert(
                    "Please use your current location first."
                );

                return;
            }


            if (
                report.peopleAffected < 1
            ) {

                alert(
                    "Enter at least 1 person affected."
                );

                return;
            }


            if (
                !report.description
            ) {

                alert(
                    "Describe the emergency."
                );

                return;
            }


            const file =
                $("evidence")
                    ?.files?.[0] ||
                null;


            if (
                file &&
                file.size >
                15 * 1024 * 1024
            ) {

                alert(
                    "Selected file is larger than 15 MB."
                );

                return;
            }


            const previewRisk =
                Number(
                    localStorage.getItem(
                        "rakshanet_last_risk"
                    ) ||
                    0
                );


            report.priority =
                calculatePriority(
                    report,
                    previewRisk
                );


            localStorage.setItem(

                "rakshanet_last_risk",

                String(
                    Math.max(
                        previewRisk,
                        report.priority.score
                    )
                )
            );


            if (
                !RN_BACKEND.available &&
                !navigator.onLine
            ) {
                try {
                    await queueEmergencyReportOffline(report, file);
                    alert(
                        "No network connection. The emergency report has been securely queued on this device and will sync automatically when connectivity returns."
                    );
                } catch (queueError) {
                    alert(
                        `Report could not be sent or queued offline.\n\n${queueError.message}`
                    );
                }
                return;
            }


            if (submitButton) {

                submitButton.disabled =
                    true;

                submitButton.textContent =
                    "⏳ SENDING REPORT...";
            }


            try {

                const fd =
                    new FormData();


                fd.append(
                    "id",
                    report.id
                );


                fd.append(
                    "reporterName",
                    report.reporterName
                );


                fd.append(
                    "contact",
                    report.contact
                );


                fd.append(
                    "incidentType",
                    report.incidentType
                );


                fd.append(
                    "latitude",
                    String(
                        report.latitude
                    )
                );


                fd.append(
                    "longitude",
                    String(
                        report.longitude
                    )
                );


                fd.append(
                    "peopleAffected",
                    String(
                        report.peopleAffected
                    )
                );


                fd.append(
                    "injured",
                    String(
                        report.injured
                    )
                );


                fd.append(
                    "trapped",
                    String(
                        report.trapped
                    )
                );


                fd.append(
                    "description",
                    report.description
                );


                fd.append(
                    "timestamp",
                    report.timestamp
                );


                fd.append(
                    "status",
                    report.status
                );


                fd.append(
                    "synced",
                    "true"
                );


                fd.append(
                    "priority",
                    JSON.stringify(
                        report.priority
                    )
                );


                if (file) {

                    fd.append(
                        "evidence",
                        file
                    );
                }


                const saved =
                    await RN_BACKEND.request(
                        "/api/reports",
                        {
                            method:
                                "POST",

                            body:
                                fd
                        }
                    );


                report.synced =
                    true;


                report.backendId =
                    saved.id;


                const lightweightReport = {

                    ...report,

                    evidence:
                        saved.evidence ||
                        null
                };


                let localReports =
                    [];


                try {

                    localReports =
                        JSON.parse(
                            localStorage.getItem(
                                "rakshanet_reports"
                            ) ||
                            "[]"
                        );


                    if (
                        !Array.isArray(
                            localReports
                        )
                    ) {

                        localReports =
                            [];
                    }

                } catch (_) {

                    localReports =
                        [];
                }


                localReports.unshift(
                    lightweightReport
                );


                localStorage.setItem(

                    "rakshanet_reports",

                    JSON.stringify(
                        localReports.slice(
                            0,
                            100
                        )
                    )
                );


                $("reportPriorityPreview")
                    .innerHTML = `

                        <div
                            class="rn-risk-badge ${report.priority.cls}"
                        >
                            ${report.priority.label}
                            · Score
                            ${report.priority.score}
                        </div>

                    `;


                alert(

                    `Emergency report sent successfully!\n\n` +

                    `Emergency ID: ${report.id}\n` +

                    `Priority: ${report.priority.label}\n\n` +

                    `The report has been saved to RakshaNet.`

                );


                form.reset();


                incidentButtons.forEach(
                    button =>
                        button.classList.remove(
                            "selected",
                            "active"
                        )
                );


                $("incidentType").value =
                    "";


                $("latitude").value =
                    "";


                $("longitude").value =
                    "";


                $("locationStatus").textContent =
                    "Location not detected";


                $("locationText").textContent =
                    "Use My Location to attach GPS.";


                $("evidencePreview").innerHTML =
                    "";


                renderMapReportsIfAvailable();


            } catch (error) {

                console.error(
                    "Emergency report submission failed:",
                    error
                );

                try {
                    await queueEmergencyReportOffline(report, file);
                    report.synced = false;
                    alert(
                        "Connection to RakshaNet was unavailable. The report has been queued offline and will be synchronized automatically when the connection is restored."
                    );
                } catch (queueError) {
                    alert(
                        `Emergency report could not be sent or queued.\n\n${queueError.message}`
                    );
                }

            } finally {

                if (submitButton) {

                    submitButton.disabled =
                        false;

                    submitButton.textContent =
                        "🚨 SEND EMERGENCY REPORT";
                }
            }
        }
    );
}


/* =========================================================
   PRIORITY PREVIEW
========================================================= */

function updatePriorityPreview() {

    const box =
        $("reportPriorityPreview");


    if (!box)
        return;


    const incidentType =
        $("incidentType")
            ?.value ||
        "Other";


    const r = {

        incidentType,

        peopleAffected:
            Number(
                $("peopleCount")
                    ?.value ||
                0
            ),

        injured:
            Number(
                $("injuredCount")
                    ?.value ||
                0
            ),

        trapped:
            Number(
                $("trappedCount")
                    ?.value ||
                0
            )
    };


    const p =
        calculatePriority(
            r,
            Number(
                localStorage.getItem(
                    "rakshanet_last_risk"
                ) ||
                0
            )
        );


    box.innerHTML = `

        <div
            class="rn-risk-badge ${p.cls}"
        >
            ${p.label}
            · Score
            ${p.score}
        </div>

    `;
}


/* =========================================================
   GIS - NEARBY SEARCH
========================================================= */

async function searchNearby(
    lat,
    lon,
    kind
) {

    const tag = {

        hospital:
            "amenity=hospital",

        shelter:
            "amenity=shelter",

        relief:
            "amenity=social_centre|amenity=community_centre",

        rescue:
            "emergency=ambulance_station|amenity=fire_station"

    }[kind];


    if (!tag)
        return [];


    const parts =
        tag
            .split("|")
            .map(
                t => {

                    const [
                        k,
                        v
                    ] =
                        t.split("=");


                    return `nwr[${k}=${JSON.stringify(v)}](around:10000,${lat},${lon});`;
                }
            )
            .join("\n");


    /*
        Smaller search area + shorter Overpass
        query timeout keeps GIS requests lighter.
    */

    const query =
        `[out:json][timeout:8];(${parts});out center tags;`;


    const data =
        await RN_BACKEND.request(
            "/api/gis/nearby",
            {

                method:
                    "POST",

                headers: {

                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        query
                    })
            }
        );


    return (

        data.elements || []

    )

        .map(
            el => {

                const c =
                    el.center ||
                    el;


                const t =
                    el.tags ||
                    {};


                return {

                    name:
                        t.name ||
                        "Unnamed facility",

                    lat:
                        Number(
                            c.lat
                        ),

                    lon:
                        Number(
                            c.lon
                        ),

                    phone:
                        t.phone ||
                        t["contact:phone"] ||
                        "",

                    address: [

                        t["addr:street"],

                        t["addr:city"],

                        t["addr:state"]

                    ]
                        .filter(Boolean)
                        .join(", ")
                };
            }
        )

        .filter(
            item =>
                Number.isFinite(
                    item.lat
                ) &&
                Number.isFinite(
                    item.lon
                )
        )

        .slice(
            0,
            12
        );
}


/* =========================================================
   DISTANCE
========================================================= */

function haversine(
    a,
    b,
    c,
    d
) {

    const R =
        6371;


    const p =
        Math.PI / 180;


    const dLat =
        (c - a) * p;


    const dLon =
        (d - b) * p;


    const x =

        Math.sin(
            dLat / 2
        ) ** 2

        +

        Math.cos(
            a * p
        )

        *

        Math.cos(
            c * p
        )

        *

        Math.sin(
            dLon / 2
        ) ** 2;


    return (

        R *

        2 *

        Math.atan2(
            Math.sqrt(x),
            Math.sqrt(1 - x)
        )

    );
}


/* =========================================================
   RENDER NEARBY RESULTS
========================================================= */

function renderNearby(
    items,
    kind,
    lat,
    lon
) {

    const box =
        $("nearbyResult");


    if (!box)
        return;


    const icon = {

        hospital:
            "🏥",

        shelter:
            "🏠",

        relief:
            "📦",

        rescue:
            "🚒"

    }[kind];


    if (!items.length) {

        box.innerHTML = `

            <div class="rn-result-item">

                <strong>
                    ${icon}
                    No mapped facilities found nearby
                </strong>

                <span class="rn-result-meta">

                    Try again or use 112
                    for immediate emergency help.

                </span>

            </div>

        `;

        return;
    }


    box.innerHTML =

        items
            .map(
                x => `

                    <div class="rn-result-item">

                        <strong>
                            ${icon}
                            ${esc(x.name)}
                        </strong>

                        <span class="rn-result-meta">

                            ${esc(
                                x.address ||
                                "Nearby facility"
                            )}

                            ·

                            ${Math.round(
                                haversine(
                                    lat,
                                    lon,
                                    x.lat,
                                    x.lon
                                ) * 100
                            ) / 100}

                            km away

                            ${
                                x.phone
                                    ? ` · ${esc(x.phone)}`
                                    : ""
                            }

                        </span>


                        <div
                            class="rn-actions"
                            style="margin-top:7px;"
                        >

                            <button
                                class="rn-btn ghost"
                                type="button"
                                onclick="
                                    window.open(
                                        'https://www.openstreetmap.org/?mlat=${x.lat}&mlon=${x.lon}#map=18/${x.lat}/${x.lon}',
                                        '_blank'
                                    )
                                "
                            >
                                View map
                            </button>


                            ${
                                x.phone

                                    ?

                                    `
                                    <a
                                        class="rn-btn"
                                        href="tel:${encodeURIComponent(x.phone)}"
                                    >
                                        Call
                                    </a>
                                    `

                                    :

                                    ""
                            }

                        </div>

                    </div>

                `
            )
            .join("");
}


/* =========================================================
   HELP PAGE
========================================================= */

function initHelpPage() {

    const btn =
        $("getHelpLocation");


    if (!btn)
        return;


    const text =
        $("helpLocationText");


    const getLoc =
        () =>
            new Promise(
                (
                    resolve,
                    reject
                ) =>
                    RakshaLocation.getLocation(
                        resolve,
                        reject
                    )
            );


    btn.addEventListener(
        "click",
        async () => {

            btn.disabled =
                true;


            text.innerHTML =
                "Detecting…";


            try {

                const l =
                    await getLoc();


                text.innerHTML = `

                    <strong>
                        Location detected
                    </strong>

                    <br>

                    Lat ${l.latitude.toFixed(6)}

                    ·

                    Lng ${l.longitude.toFixed(6)}

                    ·

                    ±${Math.round(
                        l.accuracy
                    )}m

                `;

            } catch (e) {

                text.textContent =
                    e.message;

            } finally {

                btn.disabled =
                    false;
            }
        }
    );


    [
        "findHospital",
        "findShelter",
        "findRelief",
        "findRescue"

    ].forEach(
        id => {

            $(id)?.addEventListener(
                "click",
                async () => {

                    const loc =
                        RakshaLocation
                            .getSavedLocation();


                    if (!loc) {

                        text.textContent =
                            "Detect your location first.";

                        return;
                    }


                    const kind =
                        id
                            .replace(
                                "find",
                                ""
                            )
                            .toLowerCase();


                    const box =
                        $("nearbyResult");


                    if (box) {

                        box.innerHTML = `

                            <div
                                class="rn-result-item"
                            >

                                <strong>
                                    ⏳ Searching nearby ${esc(kind)}...
                                </strong>

                                <span
                                    class="rn-result-meta"
                                >
                                    OpenStreetMap / Overpass
                                </span>

                            </div>

                        `;
                    }


                    try {

                        const items =
                            await searchNearby(
                                loc.latitude,
                                loc.longitude,
                                kind
                            );


                        renderNearby(
                            items,
                            kind,
                            loc.latitude,
                            loc.longitude
                        );


                    } catch (e) {

                        if (box) {

                            box.innerHTML = `

                                <div
                                    class="rn-result-item"
                                >

                                    <strong>
                                        ⚠️ Live search temporarily unavailable
                                    </strong>

                                    <span
                                        class="rn-result-meta"
                                    >
                                        Please try again.
                                        You can still use the map
                                        or call 112.
                                    </span>

                                </div>

                            `;
                        }


                        console.warn(
                            `${kind} search error:`,
                            e
                        );
                    }
                }
            );
        }
    );


    $("sendSos")?.addEventListener(
        "click",
        async () => {

            const description =
                $("sosDescription")
                    .value
                    .trim();


            const loc =
                RakshaLocation
                    .getSavedLocation();


            if (!loc) {

                alert(
                    "Detect your location first."
                );

                return;
            }


            const sos = {

                id:
                    `SOS-${Date.now()}`,

                latitude:
                    loc.latitude,

                longitude:
                    loc.longitude,

                description:
                    description ||
                    "Immediate rescue requested.",

                timestamp:
                    new Date()
                        .toISOString(),

                priority:
                    "P1 Critical",

                status:
                    "NEW"
            };


            const q =
                JSON.parse(
                    localStorage.getItem(
                        "rakshanet_sos"
                    ) ||
                    "[]"
                );


            q.unshift(
                sos
            );


            localStorage.setItem(

                "rakshanet_sos",

                JSON.stringify(
                    q.slice(
                        0,
                        100
                    )
                )
            );


            let backendSaved =
                false;


            if (
                RN_BACKEND.available
            ) {

                try {

                    await RN_BACKEND.request(
                        "/api/sos",
                        {

                            method:
                                "POST",

                            headers: {

                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify(
                                    sos
                                )
                        }
                    );


                    backendSaved =
                        true;

                } catch (e) {

                    console.warn(
                        "SOS backend save failed",
                        e
                    );
                }
            }


            $("sosResult").innerHTML = `

                <div
                    class="rn-alert-banner danger"
                >

                    <strong>
                        ${sos.id}
                    </strong>

                    <br>

                    SOS ${
                        backendSaved
                            ? "sent to backend"
                            : "queued locally"
                    }

                    at

                    ${loc.latitude.toFixed(5)},
                    ${loc.longitude.toFixed(5)}.

                    <br>

                    Call 112 for immediate
                    national emergency response.

                </div>

            `;
        }
    );


    initVolunteers();
}


/* =========================================================
   VOLUNTEERS
========================================================= */

function initVolunteers() {

    const list =
        $("taskList");


    if (list) {

        const sample = [

            "Check nearby shelter capacity",

            "Assist transport to nearest hospital",

            "Distribute food/water",

            "Relay local road blockage information"

        ];


        list.innerHTML =

            sample
                .map(
                    (x, i) => `

                        <div
                            class="rn-result-item"
                        >

                            <strong>
                                Task ${i + 1}
                            </strong>

                            <span
                                class="rn-result-meta"
                            >
                                ${esc(x)}
                            </span>

                        </div>

                    `
                )
                .join("");
    }


    $("registerVolunteer")?.addEventListener(
        "click",
        async () => {

            const v = {

                id:
                    `VOL-${Date.now()}`,

                name:
                    $("volName")
                        .value
                        .trim(),

                phone:
                    $("volPhone")
                        .value
                        .trim(),

                skill:
                    $("volSkill")
                        .value,

                availability:
                    $("volAvailability")
                        .value,

                timestamp:
                    new Date()
                        .toISOString()
            };


            if (
                !v.name ||
                !v.phone
            ) {

                alert(
                    "Enter volunteer name and phone."
                );

                return;
            }


            const arr =
                JSON.parse(
                    localStorage.getItem(
                        "rakshanet_volunteers"
                    ) ||
                    "[]"
                );


            arr.unshift(
                v
            );


            localStorage.setItem(

                "rakshanet_volunteers",

                JSON.stringify(
                    arr.slice(
                        0,
                        200
                    )
                )
            );


            if (
                RN_BACKEND.available
            ) {

                try {

                    const saved =
                        await RN_BACKEND.request(
                            "/api/volunteers",
                            {

                                method:
                                    "POST",

                                headers: {

                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify(v)
                            }
                        );


                    v.id =
                        saved.id;

                } catch (e) {

                    console.warn(
                        "Volunteer backend save failed",
                        e
                    );
                }
            }


            $("volunteerStatus").innerHTML = `

                <span
                    class="rn-volunteer-status"
                >
                    ✅ Registered
                    ${esc(v.id)}
                    ·
                    ${esc(v.skill)}
                    ·
                    ${esc(v.availability)}
                </span>

            `;


            $("volName").value =
                "";


            $("volPhone").value =
                "";
        }
    );
}


/* =========================================================
   MAP MARKERS
========================================================= */

function createLayerMarker(
    layer,
    item,
    emoji
) {

    const marker =
        L.circleMarker(
            [
                item.lat,
                item.lon
            ],
            {

                radius:
                    8,

                weight:
                    2,

                fillOpacity:
                    .8
            }
        );


    marker.bindPopup(`

        <strong>
            ${emoji}
            ${esc(item.name)}
        </strong>

        <br>

        ${esc(
            item.address ||
            "Nearby facility"
        )}

        ${
            item.phone
                ? `
                    <br>
                    📞
                    ${esc(item.phone)}
                `
                : ""
        }

    `);


    marker.addTo(
        layer
    );


    return marker;
}


function renderMapReportsIfAvailable() {

    if (
        window.__rnMapAPI?.renderReports
    ) {

        window.__rnMapAPI.renderReports();
    }
}




/* =========================================================
   LIVE CURRENT-SITUATION HAZARDS
========================================================= */

async function getCurrentHazardStatus(lat, lon) {
    return await RN_BACKEND.request(
        `/api/current-hazards?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`
    );
}

function hazardStatusClass(status) {
    const value = String(status || "").toUpperCase();
    if (value.includes("WARNING") || value.includes("SIGNIFICANT")) return "danger";
    if (value.includes("ACTIVITY") || value.includes("WATCH") || value.includes("COVERAGE")) return "warn";
    if (value.includes("UNAVAILABLE") || value.includes("NOT AVAILABLE") || value.includes("ERROR")) return "unknown";
    return "ok";
}

function renderCurrentSituation(target, place, weather, hazards) {
    if (!target) return;

    const hazard = hazards?.hazards || {};

    const formatUpdated = value => {
        const d = new Date(value);
        return Number.isNaN(d.getTime())
            ? "Unknown"
            : d.toLocaleTimeString(undefined, {
                hour: "2-digit",
                minute: "2-digit"
            });
    };

    const shortHazardStatus = status => {
        const s = String(status || "DATA UNAVAILABLE").toUpperCase();

        if (s.includes("NO RECENT LOCAL ACTIVITY")) return "No recent activity";
        if (s.includes("NO ACTIVE") && s.includes("WARNING")) return "No active warning";
        if (s.includes("NO CURRENT") && s.includes("BULLETIN")) return "No active bulletin";
        if (s.includes("SIGNIFICANT")) return "Significant activity";
        if (s.includes("WATCH") || s.includes("ALERT")) return "Watch / Alert";
        if (s.includes("CURRENT ACTIVITY")) return "Current activity";
        if (s.includes("DATA UNAVAILABLE")) return "Data unavailable";
        return "Check source";
    };

    const hazardClass = status => {
        const s = String(status || "").toUpperCase();
        if (s.includes("WARNING") || s.includes("SIGNIFICANT")) return "danger";
        if (s.includes("WATCH") || s.includes("ALERT") || s.includes("CURRENT ACTIVITY")) return "warn";
        if (s.includes("UNAVAILABLE")) return "unknown";
        return "ok";
    };

    const hazardRow = (icon, title, item) => {
        const status = item?.status || "DATA UNAVAILABLE";
        const cls = hazardClass(status);
        const detail = item?.detail || "No additional detail available.";
        const source = item?.source || "Official source";
        const checked = item?.checkedAt ? formatUpdated(item.checkedAt) : "Unknown";

        return `
            <details class="rn-current-hazard-mini" style="border:1px solid rgba(255,255,255,.07);border-radius:8px;padding:7px 8px;background:rgba(255,255,255,.025);">
                <summary style="list-style:none;cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:8px;">
                    <span style="font-size:.8rem;white-space:nowrap;">${icon} ${title}</span>
                    <span class="rn-current-status ${cls}" style="font-size:.66rem;padding:3px 6px;border-radius:999px;white-space:nowrap;">
                        ${esc(shortHazardStatus(status))}
                    </span>
                </summary>
                <div class="rn-small rn-muted" style="margin-top:6px;line-height:1.35;">
                    ${esc(detail)}<br>
                    Source: ${esc(source)}<br>
                    Checked: ${esc(checked)}
                </div>
            </details>
        `;
    };

    target.innerHTML = `
        <strong>📍 ${esc(place.name || "Selected location")}</strong>
        <div class="rn-small" style="margin-top:3px;">${Number(place.lat).toFixed(5)}, ${Number(place.lon).toFixed(5)}</div>

        <div style="margin-top:9px;padding:8px;border-radius:9px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.06);">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
                <strong style="font-size:.82rem;">🌤️ Current Weather</strong>
                <span class="rn-small rn-muted">${esc(formatUpdated(weather.updatedAt))}</span>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px 9px;margin-top:6px;font-size:.76rem;">
                <span>Condition</span><strong>${esc(weatherCodeText(weather.code))}</strong>
                <span>Temperature</span><strong>${weather.temp.toFixed(1)}°C</strong>
                <span>Humidity</span><strong>${weather.humidity}%</strong>
                <span>Cloud cover</span><strong>${weather.cloudCover}%</strong>
                <span>Rain now</span><strong>${weather.precipitationNow.toFixed(1)} mm</strong>
                <span>Past 24h</span><strong>${weather.precip24.toFixed(1)} mm</strong>
                <span>Wind</span><strong>${weather.currentWind.toFixed(0)} km/h</strong>
                <span>Gust</span><strong>${weather.currentWindGust.toFixed(0)} km/h</strong>
            </div>

            <div class="rn-small rn-muted" style="margin-top:6px;">Source: Open-Meteo</div>
        </div>

        <div style="margin-top:8px;padding:8px;border-radius:9px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.06);">
            <strong style="font-size:.82rem;">🌍 Current Hazard Status</strong>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:6px;">
                ${hazardRow("🌎", "Earthquake", hazard.earthquake)}
                ${hazardRow("⛰️", "Landslide", hazard.landslide)}
                ${hazardRow("🏔️", "Avalanche", hazard.avalanche)}
                ${hazardRow("🌊", "Tsunami", hazard.tsunami)}
            </div>

            <div class="rn-small rn-muted" style="margin-top:6px;text-align:right;">
                Updated ${esc(formatUpdated(hazards?.checkedAt))}
            </div>
        </div>
    `;
}

/* =========================================================
   MAP
========================================================= */

async function initMap() {

    const mapElement =
        $("disasterMap");


    if (
        !mapElement ||
        typeof L === "undefined"
    ) {

        return;
    }


    const map =
        L.map(
            mapElement
        )
        .setView(
            [26.3, 92.0],
            6
        );


    L.tileLayer(

        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",

        {

            maxZoom:
                19,

            attribution:
                "&copy; OpenStreetMap contributors"

        }

    ).addTo(
        map
    );


    const layers = {

        weather:
            L.layerGroup()
                .addTo(map),

        risk:
            L.layerGroup()
                .addTo(map),

        reports:
            L.layerGroup()
                .addTo(map),

        hospitals:
            L.layerGroup(),

        shelters:
            L.layerGroup(),

        resources:
            L.layerGroup()
    };


    let userMarker =
        null;


    let routeControl =
        null;


    let weatherCache =
        new Map();


    let destinationData = [];

let selectedFacilityKind = null;

let selectedFacilityItems = [];


    const weatherStatus =
        $("weatherStatus");


    const imdStatus =
        $("imdStatus");


    const gpsStatus =
        $("gpsStatus");


    const osmStatus =
        $("osmStatus");


    const summary =
        $("mapWeatherSummary");


    const routeStatus =
        $("routeStatus");


    const destinationSelect =
        $("destinationSelect");


    const routeChooserPanel =
    $("routeChooserPanel");

const routeEmptyState =
    $("routeEmptyState");

const routeCategoryTitle =
    $("routeCategoryTitle");

const nearestFacilityButton =
    $("nearestFacilityButton");    


    const searchInput =
        $("mapLocationSearch");


    const searchButton =
        $("mapLocationSearchBtn");


    const suggestionsBox =
        $("mapLocationSuggestions");


    const searchStatus =
        $("mapSearchStatus");


    const nearbyResult =
        $("nearbyResult");


    let searchTimer =
        null;


    let latestSearchRequest =
        0;


/* =========================================================
   SMART LOCATION SEARCH
========================================================= */

    function hideLocationSuggestions() {

        if (!suggestionsBox)
            return;


        suggestionsBox.style.display =
            "none";


        suggestionsBox.innerHTML =
            "";
    }


    function showLocationSearchLoading() {

        if (!suggestionsBox)
            return;


        suggestionsBox.innerHTML = `

            <div
                class="rn-location-result"
                style="cursor:default;"
            >

                <span
                    class="rn-location-result-title"
                >
                    🔎 Searching locations...
                </span>

                <span
                    class="rn-location-result-address"
                >
                    Looking for matching places in India
                </span>

            </div>

        `;


        suggestionsBox.style.display =
            "block";
    }


    function showLocationSearchMessage(
        title,
        description
    ) {

        if (!suggestionsBox)
            return;


        suggestionsBox.innerHTML = `

            <div
                class="rn-location-result"
                style="cursor:default;"
            >

                <span
                    class="rn-location-result-title"
                >
                    ${esc(title)}
                </span>

                <span
                    class="rn-location-result-address"
                >
                    ${esc(description)}
                </span>

            </div>

        `;


        suggestionsBox.style.display =
            "block";
    }


    function formatLocationResult(
        item
    ) {

        const address =
            item.address ||
            {};


        const parts = [

            address.suburb,

            address.neighbourhood,

            address.village,

            address.town,

            address.city,

            address.district,

            address.state

        ]
            .filter(Boolean);


        return {

            title:
                parts.length

                    ?

                    parts
                        .slice(
                            0,
                            2
                        )
                        .join(", ")

                    :

                    (
                        item.display_name ||
                        "Selected location"
                    ),


            address:
                parts.length

                    ?

                    parts
                        .slice(2)
                        .join(", ")

                    :

                    (
                        item.display_name ||
                        ""
                    )
        };
    }


    function showLocationSuggestions(
        results
    ) {

        if (!suggestionsBox)
            return;


        if (!results.length) {

            showLocationSearchMessage(

                "📍 No matching places found",

                "Try adding a city, district, village or nearby area."

            );

            return;
        }


        suggestionsBox.innerHTML =

            results

                .map(
                    (item, index) => {

                        const formatted =
                            formatLocationResult(
                                item
                            );


                        return `

                            <button
                                type="button"
                                class="rn-location-result"
                                data-index="${index}"
                            >

                                <span
                                    class="rn-location-result-title"
                                >
                                    📍
                                    ${esc(
                                        formatted.title
                                    )}
                                </span>

                                <span
                                    class="rn-location-result-address"
                                >
                                    ${esc(
                                        formatted.address
                                    )}
                                </span>

                            </button>

                        `;
                    }
                )
                .join("");


        suggestionsBox.style.display =
            "block";


        suggestionsBox
            .querySelectorAll(
                ".rn-location-result"
            )
            .forEach(
                button => {

                    button.addEventListener(
                        "click",
                        async () => {

                            const index =
                                Number(
                                    button.dataset.index
                                );


                            const selected =
                                results[index];


                            if (!selected)
                                return;


                            await selectMapLocation(
                                selected
                            );
                        }
                    );
                }
            );
    }


    async function searchMapLocations(
        query
    ) {

        const cleanQuery =
            String(
                query || ""
            )
                .trim();


        if (
            cleanQuery.length < 2
        ) {

            hideLocationSuggestions();


            if (searchStatus) {

                searchStatus.textContent =
                    "Type at least 2 characters to search.";
            }


            return;
        }


        const requestId =
            ++latestSearchRequest;


        showLocationSearchLoading();


        if (searchStatus) {

            searchStatus.textContent =
                "Searching...";
        }


        try {

            const url =
                new URL(
                    RN_CONFIG.geocode
                );


            url.search =
                new URLSearchParams({

                    q:
                        `${cleanQuery}, India`,

                    format:
                        "jsonv2",

                    addressdetails:
                        "1",

                    limit:
                        "10",

                    countrycodes:
                        "in"

                });


            const results =
                await fetchJSON(

                    url.toString(),

                    {},

                    12000

                );


            if (
                requestId !==
                latestSearchRequest
            ) {

                return;
            }


            showLocationSuggestions(
                results
            );


            if (searchStatus) {

                searchStatus.textContent =
                    results.length

                        ?

                        `${results.length} places found`

                        :

                        "No matching places found.";
            }


        } catch (error) {

            if (
                requestId !==
                latestSearchRequest
            ) {

                return;
            }


            showLocationSearchMessage(

                "⚠️ Search unavailable",

                "Check your internet connection and try again."

            );


            if (searchStatus) {

                searchStatus.textContent =
                    "Location search unavailable.";
            }


            console.warn(
                "Smart location search error:",
                error
            );
        }
    }


/* =========================================================
   SELECT MAP LOCATION
========================================================= */async function selectMapLocation(
    item
) {

    const lat = Number(item.lat);
    const lon = Number(item.lon);
    const name = item.display_name || "Selected location";

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    hideLocationSuggestions();

    if (searchInput) {
        searchInput.value = name.split(",").slice(0, 2).join(", ");
    }

    if (searchStatus) searchStatus.textContent = "Location selected.";

    map.setView([lat, lon], 13, {animate:true});

    if (window.__rnSelectedMapLocation) {
        map.removeLayer(window.__rnSelectedMapLocation);
    }

    window.__rnSelectedMapLocation =
        L.marker([lat, lon])
            .addTo(map)
            .bindPopup(`<strong>📍 Selected Location</strong><br>${esc(name)}`)
            .openPopup();

    localStorage.setItem(
        "rakshanet_map_location",
        JSON.stringify({latitude:lat, longitude:lon, name:name})
    );

    try {
        summary.innerHTML = `<strong>📍 Loading current conditions…</strong><br><span class="rn-small">${esc(name)}</span>`;
        const [raw, hazards] = await Promise.all([
            getWeather(lat, lon),
            getCurrentHazardStatus(lat, lon)
        ]);
        const weather = summarizeWeather(raw);
        renderCurrentSituation(summary, {lat, lon, name}, weather, hazards);
        await loadWeatherGrid(lat, lon);
    } catch (e) {
        console.error("Selected location current-data error:", e);
        summary.innerHTML = `<strong>📍 ${esc(name)}</strong><br><br><span class="rn-alert-banner danger">Unable to load current location data.</span><br><br><span class="rn-small">${esc(e.message)}</span>`;
    }
}



/* =========================================================
   SEARCH EVENTS
========================================================= */

    if (searchInput) {

        searchInput.addEventListener(
            "input",
            () => {

                clearTimeout(
                    searchTimer
                );


                const query =
                    searchInput.value.trim();


                if (
                    query.length < 2
                ) {

                    hideLocationSuggestions();


                    if (searchStatus) {

                        searchStatus.textContent =
                            "Start typing to search for a location.";
                    }


                    return;
                }


                searchTimer =
                    setTimeout(
                        () => {

                            searchMapLocations(
                                query
                            );

                        },
                        700
                    );
            }
        );


        searchInput.addEventListener(
            "keydown",
            event => {

                if (
                    event.key === "Enter"
                ) {

                    event.preventDefault();


                    clearTimeout(
                        searchTimer
                    );


                    searchMapLocations(
                        searchInput.value
                    );
                }


                if (
                    event.key === "Escape"
                ) {

                    hideLocationSuggestions();
                }
            }
        );
    }


    searchButton?.addEventListener(
        "click",
        () => {

            clearTimeout(
                searchTimer
            );


            searchMapLocations(
                searchInput?.value ||
                ""
            );
        }
    );


    document.addEventListener(
        "click",
        event => {

            if (

                suggestionsBox &&

                searchInput &&

                !suggestionsBox.contains(
                    event.target
                ) &&

                event.target !==
                    searchInput

            ) {

                hideLocationSuggestions();
            }
        }
    );


/* =========================================================
   WEATHER GRID
========================================================= */

    async function loadWeatherGrid(
        centerLat = 26.3,
        centerLon = 92.0
    ) {

        const pts =
            [];


        for (
            let dy = -2;
            dy <= 2;
            dy += 1
        ) {

            for (
                let dx = -2;
                dx <= 2;
                dx += 1
            ) {

                pts.push([
                    clamp(
                        centerLat +
                        dy * 1.2,
                        18,
                        33
                    ),

                    centerLon +
                    dx * 1.2
                ]);
            }
        }


        const url =
            new URL(
                RN_CONFIG.weather
            );


        url.search =
            new URLSearchParams({

                latitude:
                    pts
                        .map(
                            p => p[0]
                        )
                        .join(","),

                longitude:
                    pts
                        .map(
                            p => p[1]
                        )
                        .join(","),

                timezone:
                    "auto",

                current:
                    "temperature_2m,relative_humidity_2m,rain,precipitation,weather_code,wind_speed_10m",

                hourly:
                    "precipitation,rain,wind_speed_10m",

                daily:
                    "precipitation_sum,precipitation_probability_max,temperature_2m_max,wind_speed_10m_max",

                forecast_days:
                    "3"
            });


        try {

            const data =
                await fetchJSON(
                    url.toString()
                );


            const arr =
                Array.isArray(data)
                    ? data
                    : [data];

            const mlGrid = await rnFetchMLRiskGrid(
                arr.map((d, i) => ({
                    id: i,
                    lat: Number(d.latitude ?? pts[i][0]),
                    lon: Number(d.longitude ?? pts[i][1])
                }))
            );

            layers.weather.clearLayers();

            layers.risk.clearLayers();

            weatherCache.clear();


            arr.forEach(
                (d, i) => {

                    const lat =
                        Number(
                            d.latitude ??
                            pts[i][0]
                        );


                    const lon =
                        Number(
                            d.longitude ??
                            pts[i][1]
                        );


                    const w =
                        summarizeWeather(
                            d
                        );


                    const mlPoint =
                        mlGrid?.results?.find(
                            item => Number(item.id) === i
                        );

                    const mlRisk = Number(
                        mlPoint?.model?.percentage
                    );

                    const risk =
                        Number.isFinite(mlRisk)
                            ? mlRisk
                            : scoreRisk(
                                w,
                                "auto"
                            ).score;


                    weatherCache.set(

                        `${lat.toFixed(2)},${lon.toFixed(2)}`,

                        {

                            lat,

                            lon,

                            w,

                            risk

                        }

                    );


                    const marker =
                        L.circleMarker(
                            [
                                lat,
                                lon
                            ],
                            {

                                radius:
                                    8,

                                weight:
                                    1,

                                fillColor:
                                    riskColor(
                                        risk
                                    ),

                                color:
                                    riskColor(
                                        risk
                                    ),

                                fillOpacity:
                                    .75
                            }
                        );


                    marker.bindPopup(`

                        <strong>🌦️ Current weather</strong>

                        <br>

                        ${weatherCodeText(w.code)}

                        <br>

                        🌡️ Temp: ${w.temp.toFixed(1)}°C

                        <br>

                        💧 Humidity: ${w.humidity}%

                        <br>

                        ☁️ Cloud cover: ${w.cloudCover}%

                        <br>

                        🌧️ Rain now: ${w.precipitationNow.toFixed(1)} mm

                        <br>

                        🌧️ Past 24h: ${w.precip24.toFixed(1)} mm

                        <br>

                        💨 Wind: ${w.currentWind.toFixed(0)} km/h

                    `).addTo(
                        layers.weather
                    );


                    L.circle(
                        [
                            lat,
                            lon
                        ],
                        {

                            radius:
                                120000,

                            stroke:
                                false,

                            fillColor:
                                riskColor(
                                    risk
                                ),

                            fillOpacity:
                                Math.max(
                                    .05,
                                    Math.min(
                                        .18,
                                        risk / 600
                                    )
                                )

                        }
                    )

                        .bindPopup(
                            `Risk estimate: ${riskBand(risk).label} (${risk}/100)`
                        )

                        .addTo(
                            layers.risk
                        );

                }
            );


            if (weatherStatus) {

                weatherStatus.textContent =
                    "LIVE";


                weatherStatus.className =
                    "rn-status-ok";
            }


        } catch (error) {

            console.warn(
                "Weather grid unavailable:",
                error
            );


            if (weatherStatus) {

                weatherStatus.textContent =
                    "Unavailable";


                weatherStatus.className =
                    "rn-status-warn";
            }
        }
    }


/* =========================================================
   IMD
========================================================= */

    async function checkIMD() {

        if (!imdStatus)
            return;


        try {

            const data =
                await RN_BACKEND.request(
                    "/api/imd/district-warnings"
                );


            if (data) {

                imdStatus.textContent =
                    "LIVE";


                imdStatus.className =
                    "rn-status-ok";

            } else {

                throw new Error(
                    "No IMD data"
                );
            }


        } catch (e) {

            console.warn(
                "IMD warnings temporarily unavailable:",
                e
            );


            imdStatus.textContent =
                "Temporarily unavailable";


            imdStatus.className =
                "rn-status-warn";
        }
    }


/* =========================================================
   REPORTS
========================================================= */

    function populateReports() {

        layers.reports.clearLayers();


        const reports =
            JSON.parse(

                localStorage.getItem(
                    "rakshanet_reports"
                ) ||
                "[]"

            );


        reports.forEach(
            r => {

                if (
                    !r.latitude ||
                    !r.longitude
                ) {

                    return;
                }


                const m =
                    L.circleMarker(

                        [
                            Number(
                                r.latitude
                            ),

                            Number(
                                r.longitude
                            )
                        ],

                        {

                            radius:
                                9,

                            color:
                                r.priority?.label
                                    ?.includes("P1")
                                    ? "#ef4444"
                                    : "#f59e0b",

                            fillOpacity:
                                .85
                        }
                    );


                m.bindPopup(`

                    <strong>
                        🚨
                        ${esc(
                            r.incidentType
                        )}
                    </strong>

                    <br>

                    Priority:
                    ${esc(
                        r.priority?.label ||
                        "NEW"
                    )}

                    <br>

                    People:
                    ${r.peopleAffected}

                    <br>

                    Injured:
                    ${r.injured}

                    <br>

                    Trapped:
                    ${r.trapped}

                    <br>

                    ${esc(
                        r.description
                    )}

                    <br>

                    <small>
                        ${new Date(
                            r.timestamp
                        ).toLocaleString()}
                    </small>

                `).addTo(
                    layers.reports
                );
            }
        );
    }


/* =========================================================
   ACTIVE MAP LOCATION
========================================================= */

    function getActiveMapLocation() {

        try {

            const selected =
                JSON.parse(

                    localStorage.getItem(
                        "rakshanet_map_location"
                    ) ||
                    "null"

                );


            if (

                selected &&

                Number.isFinite(
                    Number(
                        selected.latitude
                    )
                ) &&

                Number.isFinite(
                    Number(
                        selected.longitude
                    )
                )

            ) {

                return {

                    latitude:
                        Number(
                            selected.latitude
                        ),

                    longitude:
                        Number(
                            selected.longitude
                        ),

                    name:
                        selected.name ||
                        "Selected location",

                    source:
                        "search"
                };
            }

        } catch (error) {

            console.warn(
                "Unable to read selected map location:",
                error
            );
        }


        const gps =
            RakshaLocation
                .getSavedLocation();


        if (

            gps &&

            Number.isFinite(
                Number(
                    gps.latitude
                )
            ) &&

            Number.isFinite(
                Number(
                    gps.longitude
                )
            )

        ) {

            return {

                latitude:
                    Number(
                        gps.latitude
                    ),

                longitude:
                    Number(
                        gps.longitude
                    ),

                name:
                    "My current location",

                source:
                    "gps"
            };
        }


        const center =
            map.getCenter();


        return {

            latitude:
                center.lat,

            longitude:
                center.lng,

            name:
                "Current map area",

            source:
                "map"
        };
    }


/* =========================================================
   LOAD MAP FACILITIES
========================================================= */

    async function loadFacilities(
        kind,
        layer
    ) {

        const location =
            getActiveMapLocation();


        if (!location) {

            throw new Error(
                "Please select a location first."
            );
        }


        if (searchStatus) {

            searchStatus.textContent =
                `Searching ${kind} near ${location.name}...`;
        }


        if (nearbyResult) {

            nearbyResult.innerHTML = `

                <div
                    class="rn-result-item"
                >

                    <strong>
                        ⏳ Searching nearby ${esc(kind)}...
                    </strong>

                    <span
                        class="rn-result-meta"
                    >
                        OpenStreetMap / Overpass
                    </span>

                </div>

            `;
        }


        layer.clearLayers();


        const items =
            await searchNearby(

                location.latitude,

                location.longitude,

                kind

            );


        const icon =
            kind === "hospital"

                ?

                "🏥"

                :

                kind === "shelter"

                    ?

                    "🏠"

                    :

                    kind === "rescue"

                        ?

                        "🚒"

                        :

                        "📦";


        items.forEach(
            item => {

                createLayerMarker(
                    layer,
                    item,
                    icon
                );


                destinationData.push({
    ...item,
    kind
});
            }
        );


    

        /*
            Show the actual nearby results
            in the Help & Safety card too.
        */

        if (nearbyResult) {

            renderNearby(

                items,

                kind,

                location.latitude,

                location.longitude

            );
        }


        if (searchStatus) {

            searchStatus.textContent =
                `${items.length} ${kind} locations found near ${location.name}.`;
        }


        if (osmStatus) {

            osmStatus.textContent =
                items.length
                    ? "LIVE"
                    : "No results";


            osmStatus.className =
                items.length
                    ? "rn-status-ok"
                    : "rn-status-warn";
        }


        return items;
    }


/* =========================================================
   ROUTE DESTINATIONS
========================================================= */

    function refreshDestinations(kind = selectedFacilityKind) {

    if (!destinationSelect)
        return;

    const unique =
        destinationData
            .filter(
                x =>
                    !kind ||
                    x.kind === kind
            )
            .filter(
                (x, i, a) =>
                    i ===
                    a.findIndex(
                        y =>
                            y.name === x.name &&
                            Math.abs(
                                y.lat - x.lat
                            ) < 0.0001 &&
                            Math.abs(
                                y.lon - x.lon
                            ) < 0.0001
                    )
            )
            .sort(
                (a, b) => {

                    const loc =
                        getActiveMapLocation();

                    if (!loc)
                        return 0;

                    return (
                        haversine(
                            loc.latitude,
                            loc.longitude,
                            a.lat,
                            a.lon
                        ) -
                        haversine(
                            loc.latitude,
                            loc.longitude,
                            b.lat,
                            b.lon
                        )
                    );
                }
            )
            .slice(
                0,
                30
            );


    selectedFacilityItems =
        unique;


    const icon = {

        hospital: "🏥",

        shelter: "🏠",

        rescue: "🚒",

        relief: "📦"

    }[kind] || "📍";


    const label = {

        hospital: "Hospital",

        shelter: "Shelter",

        rescue: "Rescue / Relief",

        relief: "Relief"

    }[kind] || "Destination";


    destinationSelect.innerHTML =

        `<option value="">
            Choose a ${label.toLowerCase()}
        </option>` +

        unique
            .map(
                x => `

                    <option
                        value="${x.lat},${x.lon}"
                    >
                        ${icon} ${esc(x.name)}
                        · ${haversine(
                            getActiveMapLocation().latitude,
                            getActiveMapLocation().longitude,
                            x.lat,
                            x.lon
                        ).toFixed(2)} km
                    </option>

                `
            )
            .join("");


    destinationSelect.dataset.locations =
        JSON.stringify(
            unique
        );


    if (routeChooserPanel) {

        routeChooserPanel.style.display =
            "block";
    }


    if (routeEmptyState) {

        routeEmptyState.style.display =
            "none";
    }


    if (routeCategoryTitle) {

        routeCategoryTitle.innerHTML = `

            <strong>
                ${icon} ${unique.length}
                ${label}${unique.length === 1 ? "" : "s"}
                found
            </strong>

        `;
    }


    if (nearestFacilityButton) {

        nearestFacilityButton.textContent =
            `${icon} Go to Nearest ${label}`;
    }


    if (routeStatus) {

        routeStatus.textContent =
            unique.length
                ? `Choose a ${label.toLowerCase()} from the list or use the nearest option.`
                : `No mapped ${label.toLowerCase()} locations found nearby.`;
    }
}


/* =========================================================
   MY LOCATION
========================================================= */

    $("locateUser")?.addEventListener(
        "click",
        () => {

            RakshaLocation.getLocation(
                async loc => {

                    if (userMarker) {
                        map.removeLayer(userMarker);
                    }

                    userMarker =
                        L.circleMarker(
                            [loc.latitude, loc.longitude],
                            {radius:10,color:"#25d6a2",fillColor:"#25d6a2",fillOpacity:.9}
                        )
                        .addTo(map)
                        .bindPopup("📍 You are here")
                        .openPopup();

                    map.setView([loc.latitude, loc.longitude], 12);

                    if (gpsStatus) {
                        gpsStatus.textContent = `±${Math.round(loc.accuracy)}m`;
                        gpsStatus.className = "rn-status-ok";
                    }

                    localStorage.setItem(
                        "rakshanet_map_location",
                        JSON.stringify({latitude:loc.latitude, longitude:loc.longitude, name:"My current location"})
                    );

                    if (searchInput) searchInput.value = "";

                    try {
                        summary.innerHTML = `<strong>📍 Loading current conditions…</strong><br><span class="rn-small">My current location</span>`;

                        const [raw, hazards] = await Promise.all([
                            getWeather(loc.latitude, loc.longitude),
                            getCurrentHazardStatus(loc.latitude, loc.longitude)
                        ]);

                        const weather = summarizeWeather(raw);

                        renderCurrentSituation(
                            summary,
                            {lat:loc.latitude, lon:loc.longitude, name:"My current location"},
                            weather,
                            hazards
                        );

                        await loadWeatherGrid(
                            loc.latitude,
                            loc.longitude
                        );

                    } catch (e) {
                        console.error("Current location data error:", e);
                        summary.innerHTML = `<strong>📍 My current location</strong><br><br><span class="rn-alert-banner danger">Unable to load current location data.</span><br><br><span class="rn-small">${esc(e.message)}</span>`;
                    }
                },
                msg => {
                    if (gpsStatus) {
                        gpsStatus.textContent = "Unavailable";
                        gpsStatus.className = "rn-status-bad";
                    }
                    alert(msg);
                }
            );
        }
    );



/* =========================================================
   MAP LAYERS
========================================================= */

    $("showWeather")?.addEventListener(
        "click",
        () => {

            layers.weather.addTo(
                map
            );


            layers.risk.removeFrom(
                map
            );


            layers.reports.removeFrom(
                map
            );
        }
    );


    $("showRiskZones")?.addEventListener(
        "click",
        () => {

            layers.risk.addTo(
                map
            );


            layers.weather.removeFrom(
                map
            );
        }
    );


    $("showDisasters")?.addEventListener(
        "click",
        () => {

            populateReports();


            layers.reports.addTo(
                map
            );


            layers.weather.removeFrom(
                map
            );


            layers.risk.removeFrom(
                map
            );
        }
    );


/* =========================================================
   FACILITY SEARCH BUTTONS
========================================================= */

    async function handleFacilitySearch(
        kind,
        layer,
        button
    ) {

        if (
            !button ||
            button.disabled
        ) {

            return;
        }


        const originalText =
            button.textContent;


        try {

            button.disabled =
                true;


            button.textContent =
                "⏳ Searching...";


            if (searchStatus) {

                searchStatus.textContent =
                    `Searching nearby ${kind}...`;
            }


            if (nearbyResult) {

                nearbyResult.innerHTML = `

                    <div
                        class="rn-result-item"
                    >

                        <strong>
                            ⏳ Searching nearby ${esc(kind)}...
                        </strong>

                        <span
                            class="rn-result-meta"
                        >
                            OpenStreetMap / Overpass
                        </span>

                    </div>

                `;
            }


            const items =
                await loadFacilities(
                    kind,
                    layer
                );

            selectedFacilityKind =
    kind;

selectedFacilityItems =
    items;

refreshDestinations(
    kind
);


            layer.addTo(
                map
            );


            if (searchStatus) {

                searchStatus.textContent =

                    items.length

                        ?

                        `${items.length} ${kind} locations found.`

                        :

                        `No nearby ${kind} locations found.`;
            }


        } catch (e) {

            console.warn(
                `${kind} GIS search failed:`,
                e
            );


            if (nearbyResult) {

                nearbyResult.innerHTML = `

                    <div
                        class="rn-result-item"
                    >

                        <strong>
                            ⚠️
                            ${esc(
                                kind.charAt(0).toUpperCase() +
                                kind.slice(1)
                            )}
                            search temporarily unavailable
                        </strong>

                        <span
                            class="rn-result-meta"
                        >
                            Please try again in a moment.
                            The rest of the map remains available.
                        </span>

                    </div>

                `;
            }


            if (searchStatus) {

                searchStatus.textContent =
                    `⚠️ ${kind} search temporarily unavailable. Please try again.`;
            }


            if (osmStatus) {

                osmStatus.textContent =
                    "Temporarily unavailable";


                osmStatus.className =
                    "rn-status-warn";
            }


        } finally {

            button.disabled =
                false;


            button.textContent =
                originalText;
        }
    }


    $("showHospitals")?.addEventListener(

        "click",

        () =>
            handleFacilitySearch(

                "hospital",

                layers.hospitals,

                $("showHospitals")

            )
    );


    $("showShelters")?.addEventListener(

        "click",

        () =>
            handleFacilitySearch(

                "shelter",

                layers.shelters,

                $("showShelters")

            )
    );


    $("showResources")?.addEventListener(

        "click",

        () =>
            handleFacilitySearch(

                "rescue",

                layers.resources,

                $("showResources")

            )
    );


/* =========================================================
   EVACUATION ROUTE
========================================================= */

    $("safeRouteButton")?.addEventListener(

        "click",

        () => {

            const loc =
                RakshaLocation
                    .getSavedLocation();


            if (!loc) {

                alert(
                    "Click My Location first."
                );

                return;
            }


            if (
                !destinationSelect?.value
            ) {

                alert(
                    "Choose a destination."
                );

                return;
            }


            const [
                lat,
                lon
            ] =
                destinationSelect
                    .value
                    .split(",")
                    .map(Number);


            if (routeControl) {

                map.removeControl(
                    routeControl
                );

                routeControl =
                    null;
            }


            if (routeStatus) {

                routeStatus.textContent =
                    "Calculating road route…";
            }


            routeControl =
                L.Routing.control({

                    router:
                        L.Routing.osrmv1({

                            serviceUrl:
                                RN_CONFIG.route
                        }),


                    waypoints: [

                        L.latLng(
                            loc.latitude,
                            loc.longitude
                        ),

                        L.latLng(
                            lat,
                            lon
                        )

                    ],


                    addWaypoints:
                        false,

                    draggableWaypoints:
                        false,

                    routeWhileDragging:
                        false,

                    showAlternatives:
                        true,

                    createMarker:
                        () => null,

                    fitSelectedRoutes:
                        true

                })

                    .addTo(
                        map
                    );


            routeControl.on(

                "routesfound",

                async ev => {

                    const r =
                        ev.routes[0];

                    const routeCoordinates = Array.isArray(r.coordinates)
                        ? r.coordinates
                            .filter(Boolean)
                            .map(point => ({
                                lat: Number(point.lat),
                                lon: Number(point.lng ?? point.lon)
                            }))
                            .filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon))
                        : [];

                    let riskHtml = '🧠 ML route risk: checking…';
                    try {
                        const sampled = routeCoordinates.length
                            ? routeCoordinates.filter((_, index) => index % Math.max(1, Math.ceil(routeCoordinates.length / 20)) === 0)
                            : [];
                        const risk = await RN_BACKEND.request('/api/routes/risk', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ coordinates: sampled })
                        });
                        if (Number.isFinite(Number(risk.maxRisk))) {
                            riskHtml = `🧠 ML route max risk: ${Number(risk.maxRisk)}% · mean ${Number(risk.meanRisk || 0)}%`;
                        } else {
                            riskHtml = '🧠 ML route risk unavailable';
                        }
                    } catch (error) {
                        riskHtml = '🧠 ML route risk unavailable';
                    }

                    if (routeStatus) {

                        routeStatus.innerHTML = `

                            <strong>
                                ✅ Route found
                            </strong>

                            <br>

                            ${(
                                r.summary.totalDistance /
                                1000
                            ).toFixed(1)}

                            km

                            ·

                            ${Math.round(
                                r.summary.totalTime /
                                60
                            )}

                            min

                            <br>
                            ${riskHtml}
                            <br><span class="rn-small">Use the displayed alternative routes and local road closures together with this risk signal.</span>

                        `;
                    }
                }
            );


            routeControl.on(

                "routingerror",

                () => {

                    if (routeStatus) {

                        routeStatus.innerHTML = `

                            <strong>
                                ⚠️ Route failed
                            </strong>

                            <br>

                            Check internet/routing availability.

                        `;
                    }
                }
            );
        }
    );











/* =========================================================
   NEAREST FACILITY
========================================================= */

nearestFacilityButton?.addEventListener(
    "click",
    () => {

        const loc =
            RakshaLocation
                .getSavedLocation();


        if (!loc) {

            alert(
                "Click My Location first."
            );

            return;
        }


        if (
            !selectedFacilityItems.length
        ) {

            alert(
                "No nearby facilities are available."
            );

            return;
        }


        const nearest =
            [...selectedFacilityItems]
                .sort(
                    (a, b) => {

                        const da =
                            haversine(
                                loc.latitude,
                                loc.longitude,
                                a.lat,
                                a.lon
                            );

                        const db =
                            haversine(
                                loc.latitude,
                                loc.longitude,
                                b.lat,
                                b.lon
                            );

                        return da - db;
                    }
                )[0];


        if (!nearest)
            return;


        destinationSelect.value =
            `${nearest.lat},${nearest.lon}`;


        const distance =
            haversine(
                loc.latitude,
                loc.longitude,
                nearest.lat,
                nearest.lon
            );


        if (routeStatus) {

            routeStatus.innerHTML = `

                <strong>
                    ✅ Nearest destination selected
                </strong>

                <br>

                ${esc(nearest.name)}

                ·

                ${distance.toFixed(2)}
                km away

            `;
        }

    }
);











/* =========================================================
   CLEAR ROUTE
========================================================= */

    $("clearRouteButton")?.addEventListener(

        "click",

        () => {

            if (routeControl) {

                map.removeControl(
                    routeControl
                );

                routeControl =
                    null;
            }


            if (routeStatus) {

                routeStatus.textContent =
                    "Route cleared.";
            }
        }
    );


/* =========================================================
   SEE ON MAP
========================================================= */

    $("seeOnMapButton")?.addEventListener(

        "click",

        () => {

            const mapTarget =
                $("disasterMap");


            if (!mapTarget)
                return;


            mapTarget.scrollIntoView({

                behavior:
                    "smooth",

                block:
                    "center"
            });


            /*
                If a route already exists,
                the map keeps it visible.
            */
        }
    );


/* =========================================================
   MAP API
========================================================= */

    window.__rnMapAPI = {

        renderReports:
            populateReports
    };


    populateReports();


    await loadWeatherGrid();


    await checkIMD();
}


/* =========================================================
   DOM READY
========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    async () => {

        console.log(
            "RakshaNet upgraded client loaded"
        );


        await detectBackend();

        updateAuthNavbar();

        initHome();

        initEmergencyPage();

        initHelpPage();

        initMap();
    }
);


/* =========================================================
   AUTHENTICATION NAVBAR
========================================================= */

function updateAuthNavbar() {

    const user =
        localStorage.getItem(
            "rakshanet_user"
        );


    const authButtons =
        document.querySelectorAll(
            ".rn-auth-btn"
        );


    authButtons.forEach(
        button => {

            const isPagesFolder =
                window.location.pathname.includes(
                    "/pages/"
                ) ||

                window.location.pathname.includes(
                    "\\pages\\"
                );


            if (user) {

                button.textContent =
                    "👤 My Profile";


                button.href =
                    isPagesFolder

                        ?

                        "profile.html"

                        :

                        "pages/profile.html";


            } else {

                button.textContent =
                    "🔐 Register / Login";


                button.href =
                    isPagesFolder

                        ?

                        "login.html"

                        :

                        "pages/login.html";
            }
        }
    );
}


document.addEventListener(
    "DOMContentLoaded",
    updateAuthNavbar
);









/* ============================================================
   RAKSHANET INTELLIGENCE INTEGRATION
   AI Prediction -> Unified /api/intelligence
   Existing weather + ML system remains as fallback
   ============================================================ */

async function rnGetUnifiedIntelligence(lat, lon) {
  const url =
    `${RN_BACKEND.base}/api/intelligence` +
    `?lat=${encodeURIComponent(lat)}` +
    `&lon=${encodeURIComponent(lon)}` +
    `&region=custom`;

  const response = await fetch(url, {
    method: "GET",
    headers: {
      "Accept": "application/json"
    }
  });

  if (!response.ok) {
    throw new Error(`Intelligence API returned ${response.status}`);
  }

  const data = await response.json();

  if (!data || !data.ok) {
    throw new Error(
      data?.message || "Unified Intelligence returned an invalid response."
    );
  }

  return data;
}


/* ------------------------------------------------------------
   Convert Intelligence weather into the format already used
   by RakshaNet's existing prediction renderer.
   ------------------------------------------------------------ */

function rnIntelligenceWeather(intelligence) {
  const current = intelligence?.currentWeather || {};
  const features = intelligence?.fusion?.aiFeatures || {};

  const rain24 =
    Number.isFinite(Number(features.rain_1d))
      ? Number(features.rain_1d)
      : Number(features.precip_1d || 0);

  const rain3d =
    Number.isFinite(Number(features.rain_3d))
      ? Number(features.rain_3d)
      : Number(features.precip_3d || 0);

  const windSpeed = Number(current.windSpeed || 0);

  return {
    temp: Number(current.temperature ?? 0),
    apparentTemp: Number(current.temperature ?? 0),

    humidity: Number(current.humidity ?? 0),
    cloudCover: Number(current.cloudCover ?? 0),

    visibility: null,

    rainNow: Number(current.rain ?? 0),
    precipitationNow: Number(current.precipitation ?? 0),

    precip24: rain24,
    rain24: rain24,

    next24Precip: Number(features.precip_1d ?? rain24),

    currentWind: windSpeed,
    currentWindGust: null,

    maxWind:
      Number.isFinite(Number(features.wind_max))
        ? Number(features.wind_max)
        : windSpeed,

    maxWindNext24:
      Number.isFinite(Number(features.wind_max))
        ? Number(features.wind_max)
        : windSpeed,

    maxTemp:
      Number.isFinite(Number(features.temp_max))
        ? Number(features.temp_max)
        : Number(current.temperature ?? 0),

    maxProb: null,

    rain3d: rain3d,

    code: 0,
    isDay: true,

    updatedAt:
      current.observedAt ||
      intelligence.checkedAt ||
      new Date().toISOString(),

    /* Extra Intelligence fields */
    pressure: Number(current.pressure ?? 0),
    windDirection: Number(current.windDirection ?? 0),
    soilMoisture: Number(current.soilMoisture ?? 0),

    intelligenceConnected: true,
    intelligenceSource: "RakshaNet Unified Intelligence"
  };
}


/* ------------------------------------------------------------
   Convert unified Intelligence model result into the format
   expected by renderPrediction().
   ------------------------------------------------------------ */

function rnIntelligenceModelResult(intelligence) {

  const model =
    intelligence?.fusion?.model || {};

  const features =
    intelligence?.fusion?.aiFeatures || {};

  const percentage = Math.max(
    0,
    Math.min(
      100,
      Number(
        model.percentage ??
        model.landslideProbability ??
        0
      )
    )
  );

  /*
   * IMPORTANT:
   * renderPrediction() expects landslideProbability
   * in the 0–1 range.
   *
   * Example:
   * 53% -> 0.53
   * 39% -> 0.39
   */

  const probability =
    percentage / 100;

  return {

    ok: true,

    model:
      model.model ||
      "rakshanet-landslide-v1",

    prediction: {

      /*
       * Used internally by renderPrediction()
       */
      landslideProbability:
        probability,

      /*
       * Already a percentage.
       */
      percentage:
        Math.round(percentage),

      risk:
        model.risk ||
        (
          percentage >= 80
            ? "EXTREME"
            : percentage >= 60
              ? "HIGH"
              : percentage >= 35
                ? "WATCH"
                : "LOW"
        ),

      priority:
        model.priority ||
        (
          percentage >= 80
            ? "P1 Critical"
            : percentage >= 60
              ? "P2 High"
              : percentage >= 35
                ? "P3 Watch"
                : "P4 Normal"
        ),

      predictedClass:
        Number(
          model.predictedClass ?? 0
        )
    },

    features,

    source:
      "RakshaNet Unified Intelligence"
  };
}


/* ------------------------------------------------------------
   Intelligence source / freshness information
   ------------------------------------------------------------ */

function rnGetIntelligenceSourceSummary(intelligence) {
  const sources = intelligence?.sources || {};
  const telemetry =
    intelligence?.governmentTelemetry?.feeds || [];

  let staleCount = 0;
  let connectedTelemetry = 0;

  telemetry.forEach(feed => {
    if (feed?.connected) {
      connectedTelemetry++;

      if (
        feed?.freshness?.status === "STALE"
      ) {
        staleCount++;
      }
    }
  });

  return {
    weather:
      sources.weather?.connected === true,

    satellite:
      sources.satellite?.connected === true,

    telemetry:
      sources.governmentTelemetry?.connected === true,

    connectedTelemetry,

    staleTelemetry: staleCount,

    aiReady:
      intelligence?.fusion?.aiFeaturesReady === true,

    featureCount:
      Number(intelligence?.fusion?.aiFeatureCount || 0),

    featureTotal:
      Number(intelligence?.fusion?.aiFeatureTotal || 0)
  };
}


/* ------------------------------------------------------------
   Add a small Intelligence source strip to prediction output.
   This does NOT replace the existing prediction UI.
   ------------------------------------------------------------ */

function rnRenderIntelligenceSourceStrip(target, intelligence) {
  if (!target || !intelligence) return;

  const oldStrip = target.querySelector(
    ".rn-intelligence-source-strip"
  );

  if (oldStrip) {
    oldStrip.remove();
  }

  const summary =
    rnGetIntelligenceSourceSummary(intelligence);

  const strip = document.createElement("div");

  strip.className =
    "rn-intelligence-source-strip";

  strip.style.cssText = `
    margin-top:18px;
    padding:14px 16px;
    border:1px solid rgba(0,255,180,.16);
    border-radius:12px;
    background:rgba(0,255,180,.035);
    font-size:13px;
    line-height:1.5;
  `;

  const weatherStatus =
    summary.weather
      ? "CONNECTED"
      : "OFFLINE";

  const satelliteStatus =
    summary.satellite
      ? "CONNECTED"
      : "OFFLINE";

  const telemetryStatus =
    summary.telemetry
      ? "CONNECTED"
      : "OFFLINE";

  const staleText =
    summary.staleTelemetry > 0
      ? ` · ${summary.staleTelemetry} stale`
      : "";

  strip.innerHTML = `
    <div style="
      display:flex;
      justify-content:space-between;
      gap:12px;
      flex-wrap:wrap;
      align-items:center;
    ">
      <strong style="letter-spacing:.3px;">
        🧠 Unified Intelligence
      </strong>

      <span style="opacity:.75;">
        ${summary.featureCount}/${summary.featureTotal}
        AI features ready
      </span>
    </div>

    <div style="
      margin-top:9px;
      display:flex;
      gap:14px;
      flex-wrap:wrap;
      opacity:.86;
    ">
      <span>🌦️ Weather: ${weatherStatus}</span>

      <span>🛰️ Satellite: ${satelliteStatus}</span>

      <span>
        📡 Government telemetry:
        ${telemetryStatus}${staleText}
      </span>
    </div>

    <div style="
      margin-top:8px;
      font-size:11px;
      opacity:.58;
    ">
      Data fused by RakshaNet Intelligence.
      Government observations are freshness-checked before use.
    </div>
  `;

  target.appendChild(strip);
}


/* ============================================================
   NEW analyzePlace()
   Intelligence is now PRIMARY.
   Existing weather + ML is FALLBACK.
   ============================================================ */

/* ============================================================
   Optional helper:
   Other pages can retrieve the latest Intelligence result.
   ============================================================ */

function getLastRakshaNetIntelligence() {

  try {

    const raw =
      localStorage.getItem(
        "rakshanet_last_intelligence"
      );

    if (!raw) return null;

    return JSON.parse(raw);

  } catch (error) {

    console.warn(
      "Unable to read saved Intelligence:",
      error
    );

    return null;
  }
}


/* ============================================================
   Optional helper:
   Check whether the unified Intelligence backend is alive.
   ============================================================ */

async function checkRakshaNetIntelligence() {

  try {

    const response =
      await fetch(
        `${RN_BACKEND.base}/api/intelligence?lat=26.2006&lon=92.9376&region=custom`,
        {
          method: "GET",
          headers: {
            "Accept": "application/json"
          }
        }
      );

    return response.ok;

  } catch (error) {

    console.warn(
      "Intelligence health check failed:",
      error
    );

    return false;
  }
}


/* ============================================================
   END — RAKSHANET INTELLIGENCE INTEGRATION
   ============================================================ */