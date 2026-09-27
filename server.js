require('dotenv').config({ quiet: true });

const express = require('express');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');
const ort = require('onnxruntime-node');
const jwt = require('jsonwebtoken');
const { Pool } = require('pg');
const { createClient } = require('@supabase/supabase-js');

// Optional dependency: installed in production builds for true Web Push.
// The server still boots without it so the core application remains usable.
let webPush = null;
try {
    webPush = require('web-push');
} catch (_) {
    console.warn('web-push package not installed; browser push delivery is disabled until npm install is run.');
}

const ROOT = __dirname;

const supabaseAdmin = (
    process.env.SUPABASE_URL &&
    process.env.SUPABASE_SECRET_KEY
)
    ? createClient(
        process.env.SUPABASE_URL,
        process.env.SUPABASE_SECRET_KEY,
        {
            auth: {
                autoRefreshToken: false,
                persistSession: false,
                detectSessionInUrl: false
            }
        }
    )
    : null;

const DATABASE_URL = String(process.env.DATABASE_URL || '').trim();
const DATABASE_CONFIGURED = Boolean(DATABASE_URL);

// Keep the rest of the application API-compatible even when no Postgres URL
// is configured locally. This prevents pg from silently attempting localhost:5432.
const pool = DATABASE_CONFIGURED
    ? new Pool({
        connectionString: DATABASE_URL,
        ssl: { rejectUnauthorized: false }
    })
    : {
        async query() {
            throw new Error('PostgreSQL is not configured (DATABASE_URL missing).');
        },
        on() {},
        async end() {}
    };

pool.on('error', (error) => {
    console.warn('PostgreSQL pool error:', error.message);
});

// =====================================================
// MACHINE LEARNING MODEL
// =====================================================

const ML_MODEL_FILE = path.join(ROOT, 'ml', 'landslide_model.onnx');
let landslideSessionPromise = null;

function getLandslideSession() {
    if (!landslideSessionPromise) {
        landslideSessionPromise = ort.InferenceSession.create(ML_MODEL_FILE);
    }
    return landslideSessionPromise;
}

const LANDSLIDE_FEATURES = [
    'rain_1d',
    'rain_3d',
    'rain_7d',
    'precip_1d',
    'precip_3d',
    'precip_7d',
    'precip_hours_1d',
    'precip_hours_3d',
    'precip_hours_7d',
    'temp_mean',
    'temp_max',
    'temp_min',
    'wind_max',
    'latitude',
    'longitude',
    'month'
];

const ML_MODEL_VERSION = 'rakshanet-landslide-v1';
const ML_MODEL_PROFILE_FILE = path.join(ROOT, 'ml', 'model_profile.json');
const HISTORICAL_LANDSLIDE_FILE = path.join(ROOT, 'ml', 'ner_master_landslide_inventory.csv');
let modelProfileCache = null;
let historicalInventoryCache = null;
let terrainCache = new Map();

const UPLOAD_DIR = path.join(ROOT, 'uploads');
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// =====================================================
// EXPRESS APP
// =====================================================

const app = express();

app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use(express.urlencoded({ extended: true, limit: '2mb' }));
app.use('/uploads', express.static(UPLOAD_DIR));

// =====================================================
// FILE UPLOAD
// =====================================================

const storage = multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
    filename: (_req, file, cb) => {
        const safeName = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
        cb(null, `${Date.now()}-${safeName}`);
    }
});

const upload = multer({
    storage,
    limits: { fileSize: 15 * 1024 * 1024 }
});

// =====================================================
// HEALTH
// =====================================================

app.get('/api/health', (_req, res) => {
    res.json({
        ok: true,
        service: 'RakshaNet Backend',
        time: new Date().toISOString()
    });
});

app.get('/api/db-test', async (_req, res) => {
    try {
        const result = await pool.query('SELECT NOW() AS time');
        res.json({
            ok: true,
            database: 'connected',
            time: result.rows[0].time
        });
    } catch (error) {
        console.error('Database test error:', error);
        res.status(500).json({
            ok: false,
            database: 'connection_failed',
            error: error.message
        });
    }
});

// =====================================================
// AI / ML - MODEL OUTPUT + EXPLANATION HELPERS
// =====================================================

function parseOnnxPrediction(results, outputNames = []) {
    let label = null;
    let probability = null;

    // skl2onnx with zipmap:false normally exposes outputs named
    // `label` and `probabilities`. Prefer those names instead of assuming
    // that the second arbitrary output is the positive-class probability.
    const labelOutput = results?.label;
    if (labelOutput?.data?.length) {
        const candidate = Number(labelOutput.data[0]);
        if (Number.isFinite(candidate)) label = candidate;
    }

    const probabilityOutput = results?.probabilities;
    if (probabilityOutput?.data?.length >= 2) {
        const candidate = Number(probabilityOutput.data[1]);
        if (Number.isFinite(candidate)) probability = candidate;
    }

    // Defensive fallback for exported models whose output names differ.
    for (const name of outputNames || []) {
        const output = results?.[name];
        if (!output?.data || output.data.length === 0) continue;
        const normalized = String(name).toLowerCase();
        if (probability === null && output.data.length >= 2 && normalized.includes('prob')) {
            const candidate = Number(output.data[1]);
            if (Number.isFinite(candidate)) probability = candidate;
        }
        if (label === null && output.data.length === 1 && normalized.includes('label')) {
            const candidate = Number(output.data[0]);
            if (Number.isFinite(candidate)) label = candidate;
        }
    }

    if (probability === null) {
        // Last-resort shape-based fallback.
        for (const name of outputNames || []) {
            const output = results?.[name];
            if (!output?.data) continue;
            if (output.data.length >= 2) {
                const candidate = Number(output.data[1]);
                if (Number.isFinite(candidate)) {
                    probability = candidate;
                    break;
                }
            }
        }
    }

    if (label === null) {
        for (const name of outputNames || []) {
            const output = results?.[name];
            if (!output?.data || output.data.length !== 1) continue;
            const candidate = Number(output.data[0]);
            if (Number.isFinite(candidate)) {
                label = candidate;
                break;
            }
        }
    }

    return { label, probability };
}

function loadModelProfile() {
    if (modelProfileCache) return modelProfileCache;
    try {
        if (!fs.existsSync(ML_MODEL_PROFILE_FILE)) return null;
        modelProfileCache = JSON.parse(
            fs.readFileSync(ML_MODEL_PROFILE_FILE, 'utf8')
        );
        return modelProfileCache;
    } catch (error) {
        console.warn('ML model profile could not be loaded:', error.message);
        return null;
    }
}

function loadHistoricalInventory() {
    if (historicalInventoryCache) return historicalInventoryCache;
    try {
        if (!fs.existsSync(HISTORICAL_LANDSLIDE_FILE)) {
            historicalInventoryCache = [];
            return historicalInventoryCache;
        }
        const rows = parseCsv(fs.readFileSync(HISTORICAL_LANDSLIDE_FILE, 'utf8'))
            .map(row => ({
                latitude: Number(row.latitude),
                longitude: Number(row.longitude),
                source: String(row.source || 'Unknown'),
                eventYear: Number(row.event_year)
            }))
            .filter(row => Number.isFinite(row.latitude) && Number.isFinite(row.longitude));
        historicalInventoryCache = rows;
        return rows;
    } catch (error) {
        console.warn('Historical landslide inventory could not be loaded:', error.message);
        historicalInventoryCache = [];
        return historicalInventoryCache;
    }
}

function haversineKm(lat1, lon1, lat2, lon2) {
    const toRad = value => value * Math.PI / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
    return 6371.0088 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function getHistoricalContext(latitude, longitude) {
    const inventory = loadHistoricalInventory();
    if (!inventory.length) {
        return {
            available: false,
            eventsNearby: 0,
            radiusKm: 25,
            nearestDistanceKm: null,
            susceptibilityScore: null,
            sourceCounts: {}
        };
    }

    const radiusKm = 25;
    let eventsNearby = 0;
    let nearestDistanceKm = Infinity;
    const sourceCounts = {};
    const nearbyEvents = [];

    for (const event of inventory) {
        const distance = haversineKm(latitude, longitude, event.latitude, event.longitude);
        if (distance < nearestDistanceKm) nearestDistanceKm = distance;
        if (distance <= radiusKm) {
            eventsNearby += 1;
            sourceCounts[event.source] = (sourceCounts[event.source] || 0) + 1;
            if (nearbyEvents.length < 60) nearbyEvents.push({ ...event, distanceKm: Number(distance.toFixed(2)) });
        }
    }

    nearbyEvents.sort((a, b) => a.distanceKm - b.distanceKm);
    const densityScore = Math.min(100, Math.round((eventsNearby / 40) * 100));
    return {
        available: true,
        eventsNearby,
        radiusKm,
        nearestDistanceKm: Number.isFinite(nearestDistanceKm) ? Number(nearestDistanceKm.toFixed(2)) : null,
        susceptibilityScore: densityScore,
        sourceCounts,
        nearbyEvents: nearbyEvents.slice(0, 25),
        inventoryRecords: inventory.length
    };
}

async function fetchTerrainContext(latitude, longitude) {
    const key = `${Number(latitude).toFixed(3)},${Number(longitude).toFixed(3)}`;
    if (terrainCache.has(key)) return terrainCache.get(key);

    const offsets = [
        [0, 0],
        [0.005, 0],
        [-0.005, 0],
        [0, 0.005],
        [0, -0.005]
    ];

    const locations = offsets
        .map(([dLat, dLon]) => `${latitude + dLat},${longitude + dLon}`)
        .join('|');

    try {
        const url = `https://api.opentopodata.org/v1/aster30m?locations=${encodeURIComponent(locations)}`;
        const response = await fetch(url, {
            headers: { 'User-Agent': 'RakshaNet/2.0', 'Accept': 'application/json' },
            signal: AbortSignal.timeout(9000)
        });
        if (!response.ok) throw new Error(`OpenTopoData HTTP ${response.status}`);
        const data = await response.json();
        const elevations = (data.results || [])
            .map(item => Number(item.elevation))
            .filter(Number.isFinite);
        if (!elevations.length) throw new Error('No elevation value returned.');

        const center = elevations[0];
        const north = elevations[1] ?? center;
        const south = elevations[2] ?? center;
        const east = elevations[3] ?? center;
        const west = elevations[4] ?? center;
        const verticalGradient = Math.abs(north - south) / 1110;
        const horizontalGradient = Math.abs(east - west) / 1110;
        const slopePercent = Math.min(100, Math.sqrt(verticalGradient ** 2 + horizontalGradient ** 2) * 100);

        const result = {
            available: true,
            elevationMeters: Number(center.toFixed(1)),
            slopePercent: Number(slopePercent.toFixed(2)),
            source: 'OpenTopoData / ASTER 30m proxy',
            sampledPoints: elevations.length
        };
        terrainCache.set(key, result);
        return result;
    } catch (error) {
        const result = {
            available: false,
            elevationMeters: null,
            slopePercent: null,
            source: 'OpenTopoData unavailable',
            error: error.message
        };
        terrainCache.set(key, result);
        return result;
    }
}

function buildSignalExplanations(features, historicalContext, terrain) {
    const profile = loadModelProfile();
    const importance = profile?.featureImportance || {};
    const quantiles = profile?.featureQuantiles || {};
    const signals = [];

    for (const feature of LANDSLIDE_FEATURES) {
        const value = Number(features?.[feature]);
        if (!Number.isFinite(value)) continue;
        const profileFeature = quantiles?.[feature];
        if (!profileFeature) continue;
        const q75 = Number(profileFeature.q75);
        const q25 = Number(profileFeature.q25);
        let state = 'NORMAL_RANGE';
        if (Number.isFinite(q75) && value >= q75) state = 'ELEVATED_RANGE';
        else if (Number.isFinite(q25) && value <= q25) state = 'LOWER_RANGE';
        signals.push({
            feature,
            value,
            importance: Number(importance[feature] || 0),
            state
        });
    }

    signals.sort((a, b) => (b.importance - a.importance));
    const topSignals = signals.slice(0, 5).map(signal => ({
        feature: signal.feature,
        value: signal.value,
        importance: Number(signal.importance.toFixed(4)),
        state: signal.state
    }));

    return {
        modelSignals: topSignals,
        historical: historicalContext,
        terrain: terrain,
        note: 'Signal ranges are contextual indicators from the training profile; they are not causal explanations of an individual landslide.'
    };
}

function dataConfidenceScore({ aiFeatureCount, aiFeatureTotal, telemetry, terrain, historical }) {
    const featureScore = aiFeatureTotal ? (aiFeatureCount / aiFeatureTotal) * 60 : 0;
    const sourceScore = (telemetry ? 15 : 5) + (terrain?.available ? 10 : 0) + (historical?.available ? 15 : 0);
    return Math.round(Math.min(100, featureScore + sourceScore));
}

function computeCompositeRisk({ model, currentWeather, historical, terrain, telemetry }) {
    const modelPercent = Number(model?.percentage);
    const base = Number.isFinite(modelPercent) ? modelPercent : null;

    const signals = [];
    let contextScore = 0;

    if (historical?.available) {
        const history = Math.max(0, Math.min(100, Number(historical.susceptibilityScore || 0)));
        const contribution = history * 0.15;
        contextScore += contribution;
        signals.push({ key: 'historical_susceptibility', value: history, contribution: Number(contribution.toFixed(1)) });
    }

    if (terrain?.available && Number.isFinite(Number(terrain.slopePercent))) {
        const slope = Math.max(0, Math.min(100, Number(terrain.slopePercent)));
        const slopeSignal = Math.min(100, slope * 2.5);
        const contribution = slopeSignal * 0.10;
        contextScore += contribution;
        signals.push({ key: 'terrain_slope', value: slope, contribution: Number(contribution.toFixed(1)) });
    }

    const rain24 = Number(currentWeather?.rain ?? currentWeather?.precipitation);
    if (Number.isFinite(rain24)) {
        const rainSignal = Math.max(0, Math.min(100, (rain24 / 50) * 100));
        const contribution = rainSignal * 0.10;
        contextScore += contribution;
        signals.push({ key: 'current_rain', value: rain24, contribution: Number(contribution.toFixed(1)) });
    }

    const telemetryReady = Boolean(telemetry?.ok);
    if (telemetryReady) {
        const contribution = 5;
        contextScore += contribution;
        signals.push({ key: 'government_telemetry', value: 1, contribution });
    }

    if (base === null) {
        return {
            available: false,
            score: Math.round(Math.min(100, contextScore)),
            basis: 'Context-only operational index; ML prediction unavailable.',
            signals
        };
    }

    const score = Math.round(Math.max(0, Math.min(100, (base * 0.60) + contextScore)));
    let band = 'LOW';
    if (score >= 80) band = 'EXTREME';
    else if (score >= 60) band = 'HIGH';
    else if (score >= 35) band = 'WATCH';

    return {
        available: true,
        score,
        band,
        mlProbabilityPercent: base,
        basis: '60% current ML estimate + contextual signals. This composite score is not a calibrated probability.',
        signals
    };
}

// =====================================================
// AI / ML - LANDSLIDE PREDICTION
// =====================================================

app.post('/api/predict/landslide', async (req, res) => {
    try {
        const body = req.body || {};
        const values = LANDSLIDE_FEATURES.map(feature => Number(body[feature]));
        const invalid = values.some(value => !Number.isFinite(value));

        if (invalid) {
            return res.status(400).json({
                ok: false,
                error: 'All landslide prediction features are required and must be numeric.',
                requiredFeatures: LANDSLIDE_FEATURES
            });
        }

        const session = await getLandslideSession();
        const inputTensor = new ort.Tensor(
            'float32',
            Float32Array.from(values),
            [1, values.length]
        );

        const inputName = session.inputNames[0];
        const feeds = {};
        feeds[inputName] = inputTensor;

        const results = await session.run(feeds);
        const outputNames = session.outputNames;

        const parsedPrediction = parseOnnxPrediction(results, outputNames);
        const label = parsedPrediction.label;
        let probability = parsedPrediction.probability;

        if (probability === null || !Number.isFinite(probability)) {
            probability = label === 1 ? 1 : 0;
        }

        probability = Math.max(0, Math.min(1, probability));
        const percentage = Math.round(probability * 100);

        let risk;
        let priority;

        if (percentage >= 80) {
            risk = 'EXTREME';
            priority = 'P1 Critical';
        } else if (percentage >= 60) {
            risk = 'HIGH';
            priority = 'P2 High';
        } else if (percentage >= 35) {
            risk = 'WATCH';
            priority = 'P3 Watch';
        } else {
            risk = 'LOW';
            priority = 'P4 Normal';
        }

        return res.json({
            ok: true,
            model: ML_MODEL_VERSION,
            prediction: {
                landslideProbability: Number(probability.toFixed(4)),
                percentage,
                risk,
                priority,
                predictedClass: percentage >= 50 ? 1 : 0
            },
            features: Object.fromEntries(
                LANDSLIDE_FEATURES.map((feature, index) => [feature, values[index]])
            ),
            message: 'Preliminary ML-based landslide risk estimate. Follow official warnings and local authorities for emergency decisions.'
        });
    } catch (error) {
        console.error('Landslide ML prediction error:', error);
        return res.status(500).json({
            ok: false,
            error: 'Unable to run landslide prediction.',
            details: error.message
        });
    }
});

// =====================================================
// AUTHENTICATION
// =====================================================

app.post('/api/auth/register', async (req, res) => {
    try {
        const {
            name,
            mobile,
            email,
            password,
            permanentLocation,
            latitude,
            longitude,
            alertPreferences,
            preferredLanguage
        } = req.body || {};

        if (!name || !mobile || !password) {
            return res.status(400).json({
                error: 'Name, mobile number and password are required.'
            });
        }

        if (password.length < 6) {
            return res.status(400).json({
                error: 'Password must contain at least 6 characters.'
            });
        }

        const normalizedMobile = String(mobile).replace(/\s+/g, '');
        const normalizedEmail = email ? String(email).trim().toLowerCase() : null;

        const existingUser = await pool.query(
            `
            SELECT id
            FROM public.users
            WHERE mobile = $1
               OR (
                    $2::text IS NOT NULL
                    AND email IS NOT NULL
                    AND LOWER(email) = LOWER($2)
               )
            LIMIT 1
            `,
            [normalizedMobile, normalizedEmail]
        );

        if (existingUser.rows.length > 0) {
            return res.status(409).json({
                error: 'An account with this mobile number or email already exists.'
            });
        }

        const passwordHash = await bcrypt.hash(password, 12);
        const userId = `USER-${Date.now()}-${crypto.randomBytes(3).toString('hex')}`;

        const location = {
            name: permanentLocation?.name || '',
            latitude: Number.isFinite(Number(latitude)) ? Number(latitude) : null,
            longitude: Number.isFinite(Number(longitude)) ? Number(longitude) : null
        };

        const preferences = Array.isArray(alertPreferences)
            ? alertPreferences.map(value => String(value).toLowerCase()).filter(value => ALERT_HAZARD_KEYS.includes(value))
            : ['heavy_rain', 'flood', 'landslide', 'thunderstorm'];
        const language = ['en', 'hi', 'as', 'bn'].includes(String(preferredLanguage || '').toLowerCase())
            ? String(preferredLanguage).toLowerCase()
            : 'en';

        await pool.query(`ALTER TABLE public.users ADD COLUMN IF NOT EXISTS preferred_language TEXT NOT NULL DEFAULT 'en'`);
        await pool.query(
            `
            INSERT INTO public.users (
                id, name, mobile, email, password_hash,
                permanent_location, alert_preferences, preferred_language, status, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $10)
            `,
            [
                userId,
                String(name).trim(),
                normalizedMobile,
                normalizedEmail,
                passwordHash,
                JSON.stringify(location),
                JSON.stringify(preferences),
                language,
                'ACTIVE',
                new Date().toISOString()
            ]
        );

        return res.status(201).json({
            ok: true,
            user: {
                id: userId,
                name: String(name).trim(),
                mobile: normalizedMobile,
                email: normalizedEmail,
                permanentLocation: location,
                alertPreferences: preferences,
                preferredLanguage: language
            }
        });
    } catch (error) {
        console.error('Registration error:', error);
        return res.status(500).json({ error: 'Unable to create account.' });
    }
});

app.post('/api/auth/login', async (req, res) => {
    try {
        const { identifier, password } = req.body || {};

        if (!identifier || !password) {
            return res.status(400).json({
                error: 'Mobile/email and password are required.'
            });
        }

        const value = String(identifier).trim().toLowerCase();

        const result = await pool.query(
            `
            SELECT id, name, mobile, email, password_hash,
                   permanent_location, alert_preferences, preferred_language
            FROM public.users
            WHERE mobile = $1
               OR (email IS NOT NULL AND LOWER(email) = LOWER($1))
            LIMIT 1
            `,
            [value]
        );

        if (result.rows.length === 0) {
            return res.status(401).json({ error: 'Invalid login credentials.' });
        }

        const user = result.rows[0];
        const valid = await bcrypt.compare(password, user.password_hash);

        if (!valid) {
            return res.status(401).json({ error: 'Invalid login credentials.' });
        }

        return res.json({
            ok: true,
            user: {
                id: user.id,
                name: user.name,
                mobile: user.mobile,
                email: user.email,
                permanentLocation: user.permanent_location,
                alertPreferences: user.alert_preferences,
                preferredLanguage: user.preferred_language || 'en'
            }
        });
    } catch (error) {
        console.error('Login error:', error);
        return res.status(500).json({ error: 'Unable to login.' });
    }
});

app.put('/api/auth/profile/:id', async (req, res) => {
    try {
        const {
            name,
            email,
            permanentLocation,
            latitude,
            longitude,
            alertPreferences,
            preferredLanguage
        } = req.body || {};

        const userResult = await pool.query(
            `
            SELECT id, name, mobile, email,
                   permanent_location, alert_preferences, preferred_language
            FROM public.users
            WHERE id = $1
            LIMIT 1
            `,
            [req.params.id]
        );

        if (userResult.rows.length === 0) {
            return res.status(404).json({ error: 'User not found.' });
        }

        const currentUser = userResult.rows[0];

        const newName = name !== undefined
            ? String(name).trim()
            : currentUser.name;

        const newEmail = email !== undefined
            ? (String(email).trim() ? String(email).trim().toLowerCase() : null)
            : currentUser.email;

        let newLocation = currentUser.permanent_location;

        if (permanentLocation !== undefined) {
            newLocation = {
                name: permanentLocation?.name || currentUser.permanent_location?.name || '',
                latitude: Number.isFinite(Number(latitude))
                    ? Number(latitude)
                    : currentUser.permanent_location?.latitude ?? null,
                longitude: Number.isFinite(Number(longitude))
                    ? Number(longitude)
                    : currentUser.permanent_location?.longitude ?? null
            };
        }

        const newPreferences = Array.isArray(alertPreferences)
            ? alertPreferences.map(value => String(value).toLowerCase()).filter(value => ALERT_HAZARD_KEYS.includes(value))
            : currentUser.alert_preferences;
        const newLanguage = preferredLanguage !== undefined
            ? (['en', 'hi', 'as', 'bn'].includes(String(preferredLanguage).toLowerCase()) ? String(preferredLanguage).toLowerCase() : 'en')
            : (currentUser.preferred_language || 'en');
        if (!Array.isArray(newPreferences) || !newPreferences.length) {
            return res.status(400).json({ error: 'Select at least one alert preference.' });
        }

        const updated = await pool.query(
            `
            UPDATE public.users
            SET name = $1,
                email = $2,
                permanent_location = $3::jsonb,
                alert_preferences = $4::jsonb,
                preferred_language = $5,
                updated_at = $6
            WHERE id = $7
            RETURNING id, name, mobile, email,
                      permanent_location, alert_preferences, preferred_language
            `,
            [
                newName,
                newEmail,
                JSON.stringify(newLocation),
                JSON.stringify(newPreferences),
                newLanguage,
                new Date().toISOString(),
                req.params.id
            ]
        );

        const user = updated.rows[0];

        return res.json({
            ok: true,
            user: {
                id: user.id,
                name: user.name,
                mobile: user.mobile,
                email: user.email,
                permanentLocation: user.permanent_location,
                alertPreferences: user.alert_preferences,
                preferredLanguage: user.preferred_language || 'en'
            }
        });
    } catch (error) {
        console.error('Profile update error:', error);
        return res.status(500).json({ error: 'Unable to update profile.' });
    }
});

// =====================================================
// ADMIN LOGIN
// =====================================================

app.post('/api/admin/login', async (req, res) => {
    try {
        const { identifier, password } = req.body || {};

        if (!identifier || !password) {
            return res.status(400).json({
                error: 'Mobile/email and password are required.'
            });
        }

        const value = String(identifier).trim().toLowerCase();

        const result = await pool.query(
            `
            SELECT id, name, mobile, email, password_hash, role, status
            FROM public.users
            WHERE mobile = $1
               OR (email IS NOT NULL AND LOWER(email) = LOWER($1))
            LIMIT 1
            `,
            [value]
        );

        if (result.rows.length === 0) {
            return res.status(401).json({ error: 'Invalid admin credentials.' });
        }

        const user = result.rows[0];
        const passwordValid = await bcrypt.compare(password, user.password_hash);

        if (!passwordValid) {
            return res.status(401).json({ error: 'Invalid admin credentials.' });
        }

        const status = String(user.status || '').trim().toUpperCase();
        if (status !== 'ACTIVE') {
            return res.status(403).json({ error: 'This account is not active.' });
        }

        const role = String(user.role || '').trim().toUpperCase();
        if (role !== 'ADMIN') {
            return res.status(403).json({ error: 'Administrator access is required.' });
        }

        const token = jwt.sign(
            { userId: user.id, role: 'ADMIN' },
            process.env.JWT_SECRET,
            {
                expiresIn: '8h',
                issuer: 'RakshaNet',
                audience: 'RakshaNet-Admin'
            }
        );

        return res.json({
            ok: true,
            token,
            user: {
                id: user.id,
                name: user.name,
                mobile: user.mobile,
                email: user.email,
                role: 'ADMIN'
            }
        });
    } catch (error) {
        console.error('Admin login error:', error);
        return res.status(500).json({
            error: 'Unable to process admin login.',
            details: error.message
        });
    }
});

// =====================================================
// ADMIN AUTHENTICATION MIDDLEWARE
// =====================================================

function requireAdmin(req, res, next) {
    try {
        const authHeader = req.headers.authorization || '';

        if (!authHeader.startsWith('Bearer ')) {
            return res.status(401).json({
                error: 'Admin authentication required.'
            });
        }

        const token = authHeader.slice(7).trim();
        if (!token) {
            return res.status(401).json({
                error: 'Admin authentication required.'
            });
        }

        const decoded = jwt.verify(
            token,
            process.env.JWT_SECRET,
            {
                issuer: 'RakshaNet',
                audience: 'RakshaNet-Admin'
            }
        );

        if (!decoded || decoded.role !== 'ADMIN') {
            return res.status(403).json({
                error: 'Administrator access required.'
            });
        }

        req.admin = decoded;
        next();
    } catch (error) {
        // Invalid/expired browser tokens are expected during local development;
        // return 401 without flooding the server console.
        return res.status(401).json({
            error: 'Invalid or expired admin token.'
        });
    }
}

// =====================================================
// REPORTS
// =====================================================

function readLocalReportsStore() {
    const file = path.join(ROOT, 'data', 'reports.json');
    try {
        if (!fs.existsSync(file)) return [];
        const value = JSON.parse(fs.readFileSync(file, 'utf8'));
        return Array.isArray(value) ? value : [];
    } catch (_) {
        return [];
    }
}

function writeLocalReportsStore(rows) {
    const file = path.join(ROOT, 'data', 'reports.json');
    try {
        fs.mkdirSync(path.dirname(file), { recursive: true });
        fs.writeFileSync(file, JSON.stringify(Array.isArray(rows) ? rows : [], null, 2));
        return true;
    } catch (_) {
        return false;
    }
}

app.get('/api/reports', async (_req, res) => {
    if (!DATABASE_CONFIGURED) {
        return res.json(readLocalReportsStore());
    }
    try {
        const result = await pool.query(
            `
            SELECT
                id, reporter_name AS "reporterName", contact, incident_type AS "incidentType", latitude, longitude,
                people_affected AS "peopleAffected", injured, trapped, description, priority, timestamp, status, synced, evidence
            FROM public.emergency_reports ORDER BY timestamp DESC
            `
        );
        return res.json(result.rows);
    } catch (error) {
        const local = readLocalReportsStore();
        if (local.length) return res.json(local);
        return res.status(500).json({ error: 'Unable to load emergency reports.' });
    }
});

app.delete('/api/reports/:id', requireAdmin, async (req, res) => {
    if (!DATABASE_CONFIGURED) {
        const rows = readLocalReportsStore();
        const index = rows.findIndex(row => String(row?.id) === String(req.params.id));
        if (index < 0) return res.status(404).json({ error: 'Report not found.' });
        const deleted = rows.splice(index, 1)[0];
        writeLocalReportsStore(rows);
        return res.json({ ok: true, deleted });
    }
    try {
        const result = await pool.query(
            `
            DELETE FROM public.emergency_reports
            WHERE id = $1
            RETURNING
                id,
                reporter_name AS "reporterName",
                contact,
                incident_type AS "incidentType",
                latitude,
                longitude,
                people_affected AS "peopleAffected",
                injured,
                trapped,
                description,
                priority,
                timestamp,
                status,
                synced,
                evidence
            `,
            [req.params.id]
        );

        if (result.rows.length === 0) {
            return res.status(404).json({ error: 'Report not found.' });
        }

        return res.json({ ok: true, deleted: result.rows[0] });
    } catch (error) {
        console.error('Delete report error:', error);
        return res.status(500).json({
            error: 'Unable to delete report.',
            details: error.message
        });
    }
});

app.delete('/api/reports', requireAdmin, async (_req, res) => {
    if (!DATABASE_CONFIGURED) {
        const rows = readLocalReportsStore();
        const remaining = rows.filter(row => String(row?.status || '').toUpperCase() !== 'RESOLVED');
        writeLocalReportsStore(remaining);
        return res.json({ ok: true, removed: rows.length - remaining.length });
    }
    try {
        const result = await pool.query(
            `
            DELETE FROM public.emergency_reports
            WHERE UPPER(COALESCE(status, '')) = 'RESOLVED'
            RETURNING id
            `
        );

        return res.json({ ok: true, removed: result.rowCount });
    } catch (error) {
        console.error('Delete resolved reports error:', error);
        return res.status(500).json({
            error: 'Unable to delete resolved reports.',
            details: error.message
        });
    }
});

app.patch('/api/reports/:id/resolve', requireAdmin, async (req, res) => {
    if (!DATABASE_CONFIGURED) {
        const rows = readLocalReportsStore();
        const index = rows.findIndex(row => String(row?.id) === String(req.params.id));
        if (index < 0) return res.status(404).json({ error: 'Report not found.' });
        rows[index] = { ...rows[index], status: 'RESOLVED', resolvedAt: new Date().toISOString(), synced: false };
        writeLocalReportsStore(rows);
        return res.json({ ok: true, report: rows[index] });
    }
    try {
        const result = await pool.query(
            `
            UPDATE public.emergency_reports
            SET status = 'RESOLVED',
                resolved_at = NOW()
            WHERE id = $1
            RETURNING
                id,
                reporter_name AS "reporterName",
                contact,
                incident_type AS "incidentType",
                latitude,
                longitude,
                people_affected AS "peopleAffected",
                injured,
                trapped,
                description,
                priority,
                timestamp,
                status,
                synced,
                evidence,
                resolved_at AS "resolvedAt"
            `,
            [req.params.id]
        );

        if (result.rows.length === 0) {
            return res.status(404).json({ error: 'Report not found.' });
        }

        return res.json({ ok: true, report: result.rows[0] });
    } catch (error) {
        console.error('Resolve report error:', error);
        return res.status(500).json({
            error: 'Unable to resolve report.',
            details: error.message
        });
    }
});

app.post('/api/reports', upload.single('evidence'), async (req, res) => {
    try {
        const body = req.body || {};
        let priority = null;

        try {
            priority = body.priority ? JSON.parse(body.priority) : null;
        } catch (_error) {
            priority = null;
        }

        const report = {
            id: body.id || `RN-${Date.now()}-${crypto.randomBytes(3).toString('hex')}`,
            reporterName: body.reporterName || 'Anonymous',
            contact: body.contact || '',
            incidentType: body.incidentType || 'Other',
            latitude: Number(body.latitude),
            longitude: Number(body.longitude),
            peopleAffected: Number(body.peopleAffected || 0),
            injured: Number(body.injured || 0),
            trapped: Number(body.trapped || 0),
            description: body.description || '',
            priority,
            timestamp: body.timestamp || new Date().toISOString(),
            status: 'NEW',
            synced: true
        };

        // Evidence remains traceable even without PostgreSQL/Supabase in local prototype mode.
        if (req.file) {
            if (!DATABASE_CONFIGURED) {
                report.evidence = {
                    name: req.file.originalname,
                    url: `/uploads/${encodeURIComponent(path.basename(req.file.path))}`,
                    path: path.basename(req.file.path),
                    size: req.file.size,
                    type: req.file.mimetype,
                    storage: 'LOCAL'
                };
            } else {
                try {
                    if (!supabaseAdmin) {
                        throw new Error('Supabase Storage is not configured. Set SUPABASE_URL and SUPABASE_SECRET_KEY.');
                    }
                const safeFileName = String(req.file.filename)
                    .replace(/[^a-zA-Z0-9._-]/g, '_');

                const storagePath = `reports/${report.id}/${safeFileName}`;
                const fileBuffer = fs.readFileSync(req.file.path);

                console.log(
                    'Uploading evidence to Supabase Storage:',
                    {
                        bucket: 'rakshanet-evidence',
                        path: storagePath,
                        size: fileBuffer.length,
                        type: req.file.mimetype
                    }
                );

                const uploadResult = await supabaseAdmin.storage
                    .from('rakshanet-evidence')
                    .upload(storagePath, fileBuffer, {
                        contentType: req.file.mimetype,
                        cacheControl: '3600',
                        upsert: true
                    });

                if (uploadResult.error) {
                    console.error('Supabase Storage upload error:', uploadResult.error);
                    return res.status(500).json({
                        error: 'Evidence file could not be uploaded to cloud storage.',
                        details: uploadResult.error.message
                    });
                }

                const publicUrlResult = supabaseAdmin.storage
                    .from('rakshanet-evidence')
                    .getPublicUrl(storagePath);

                report.evidence = {
                    name: req.file.originalname,
                    url: publicUrlResult.data.publicUrl,
                    path: storagePath,
                    size: req.file.size,
                    type: req.file.mimetype
                };

                    try {
                        fs.unlinkSync(req.file.path);
                    } catch (cleanupError) {
                        console.warn('Temporary evidence cleanup failed:', cleanupError.message);
                    }
                } catch (storageError) {
                    console.error('Evidence storage error:', storageError);
                    try { if (req.file?.path && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path); } catch (_) {}
                    return res.status(500).json({ error: 'Unable to store evidence file.', details: storageError.message });
                }
            }
        } else {
            report.evidence = null;
        }

        if (
            !Number.isFinite(report.latitude) ||
            !Number.isFinite(report.longitude)
        ) {
            return res.status(400).json({
                error: 'Valid latitude and longitude are required.'
            });
        }

        if (!DATABASE_CONFIGURED) {
            report.synced = false;
            const rows = readLocalReportsStore();
            const existingIndex = rows.findIndex(row => String(row?.id) === String(report.id));
            if (existingIndex >= 0) rows[existingIndex] = report;
            else rows.unshift(report);
            if (!writeLocalReportsStore(rows)) return res.status(500).json({ error: 'Unable to save local citizen report.' });
            return res.status(201).json(report);
        }

        await pool.query(
            `
            INSERT INTO public.emergency_reports (
                id,
                reporter_name,
                contact,
                incident_type,
                latitude,
                longitude,
                people_affected,
                injured,
                trapped,
                description,
                priority,
                timestamp,
                status,
                synced,
                evidence
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11::jsonb, $12, $13, $14, $15::jsonb
            )
            `,
            [
                report.id,
                report.reporterName,
                report.contact,
                report.incidentType,
                report.latitude,
                report.longitude,
                report.peopleAffected,
                report.injured,
                report.trapped,
                report.description,
                JSON.stringify(report.priority),
                report.timestamp,
                report.status,
                report.synced,
                JSON.stringify(report.evidence)
            ]
        );

        
        return res.status(201).json(report);
    } catch (error) {
        console.error('Create report error:', error);
        return res.status(500).json({
            error: 'Unable to save emergency report.'
        });
    }
});

// =====================================================
// SOS
// =====================================================

app.get('/api/sos', async (_req, res) => {
    try {
        const result = await pool.query(
            `
            SELECT id, latitude, longitude, description,
                   timestamp, priority, status, synced
            FROM public.sos_alerts
            ORDER BY timestamp DESC
            `
        );
        return res.json(result.rows);
    } catch (error) {
        console.error('Get SOS alerts error:', error);
        return res.status(500).json({
            error: 'Unable to load SOS alerts.'
        });
    }
});

app.post('/api/sos', async (req, res) => {
    try {
        const body = req.body || {};
        const sos = {
            id: body.id || `SOS-${Date.now()}-${crypto.randomBytes(3).toString('hex')}`,
            latitude: Number(body.latitude),
            longitude: Number(body.longitude),
            description: body.description || 'Immediate rescue requested.',
            timestamp: body.timestamp || new Date().toISOString(),
            priority: 'P1 Critical',
            status: 'NEW',
            synced: true
        };

        if (!Number.isFinite(sos.latitude) || !Number.isFinite(sos.longitude)) {
            return res.status(400).json({ error: 'Valid location is required.' });
        }

        await pool.query(
            `
            INSERT INTO public.sos_alerts (
                id, latitude, longitude, description,
                timestamp, priority, status, synced
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            `,
            [
                sos.id,
                sos.latitude,
                sos.longitude,
                sos.description,
                sos.timestamp,
                sos.priority,
                sos.status,
                sos.synced
            ]
        );

        return res.status(201).json(sos);
    } catch (error) {
        console.error('Create SOS error:', error);
        return res.status(500).json({
            error: 'Unable to save SOS alert.',
            details: error.message
        });
    }
});

// =====================================================
// VOLUNTEERS
// =====================================================

app.get('/api/volunteers', async (_req, res) => {
    try {
        const result = await pool.query(
            `
            SELECT id, name, phone, skill, availability,
                   latitude, longitude, timestamp, status, synced
            FROM public.volunteers
            ORDER BY timestamp DESC
            `
        );
        return res.json(result.rows);
    } catch (error) {
        console.error('Get volunteers error:', error);
        return res.status(500).json({
            error: 'Unable to load volunteers.'
        });
    }
});

app.post('/api/volunteers', async (req, res) => {
    try {
        const body = req.body || {};

        if (!body.name || !body.phone) {
            return res.status(400).json({
                error: 'Volunteer name and phone are required.'
            });
        }

        const volunteer = {
            id: body.id || `VOL-${Date.now()}-${crypto.randomBytes(3).toString('hex')}`,
            name: String(body.name).trim(),
            phone: String(body.phone).trim(),
            skill: body.skill || 'General',
            availability: body.availability || 'Available now',
            latitude: body.latitude !== undefined && body.latitude !== ''
                ? Number(body.latitude)
                : null,
            longitude: body.longitude !== undefined && body.longitude !== ''
                ? Number(body.longitude)
                : null,
            timestamp: body.timestamp || new Date().toISOString(),
            status: 'REGISTERED',
            synced: true
        };

        if (volunteer.latitude !== null && !Number.isFinite(volunteer.latitude)) {
            volunteer.latitude = null;
        }

        if (volunteer.longitude !== null && !Number.isFinite(volunteer.longitude)) {
            volunteer.longitude = null;
        }

        await pool.query(
            `
            INSERT INTO public.volunteers (
                id, name, phone, skill, availability,
                latitude, longitude, timestamp, status, synced
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            `,
            [
                volunteer.id,
                volunteer.name,
                volunteer.phone,
                volunteer.skill,
                volunteer.availability,
                volunteer.latitude,
                volunteer.longitude,
                volunteer.timestamp,
                volunteer.status,
                volunteer.synced
            ]
        );

        return res.status(201).json(volunteer);
    } catch (error) {
        console.error('Create volunteer error:', error);
        return res.status(500).json({
            error: 'Unable to save volunteer.',
            details: error.message
        });
    }
});

// =====================================================
// DASHBOARD
// =====================================================

app.get('/api/dashboard', requireAdmin, async (_req, res) => {
    try {
        const reportsResult = await pool.query(
            `
            SELECT id,
                   reporter_name AS "reporterName",
                   contact,
                   incident_type AS "incidentType",
                   latitude,
                   longitude,
                   people_affected AS "peopleAffected",
                   injured,
                   trapped,
                   description,
                   priority,
                   timestamp,
                   status,
                   synced,
                   evidence
            FROM public.emergency_reports
            ORDER BY timestamp DESC
            `
        );

        const sosResult = await pool.query(
            `
            SELECT id, latitude, longitude, description,
                   timestamp, priority, status, synced
            FROM public.sos_alerts
            ORDER BY timestamp DESC
            `
        );

        const volunteersResult = await pool.query(
            `
            SELECT id, name, phone, skill, availability,
                   latitude, longitude, timestamp, status, synced
            FROM public.volunteers
            ORDER BY timestamp DESC
            `
        );

        const reports = reportsResult.rows;
        const sos = sosResult.rows;
        const volunteers = volunteersResult.rows;

        const summary = {
            reports: reports.length,
            criticalReports: reports.filter(
                report => report.priority?.label === 'P1 Critical'
            ).length,
            activeSOS: sos.filter(
                alert => String(alert.status || '').toUpperCase() !== 'RESOLVED'
            ).length,
            volunteers: volunteers.length,
            availableVolunteers: volunteers.filter(
                volunteer => String(volunteer.availability || '').toLowerCase() !== 'not available'
            ).length
        };

        return res.json({ summary, reports, sos, volunteers });
    } catch (error) {
        console.error('Dashboard error:', error);
        return res.status(500).json({
            error: 'Unable to load dashboard data.',
            details: error.message
        });
    }
});

// =====================================================
// LIVE IMD PROXY
// =====================================================

app.get('/api/imd/district-warnings', async (_req, res) => {
    try {
        const imdUrl = 'https://mausam.imd.gov.in/api/warnings_district_api.php';

        const response = await fetch(imdUrl, {
            method: 'GET',
            headers: {
                'User-Agent': 'RakshaNet/1.0',
                'Accept': 'application/json,text/plain,*/*'
            }
        });

        const text = await response.text();

        if (!response.ok) {
            console.warn(`INFO: IMD district warnings unavailable (HTTP ${response.status}). ${response.status === 401 ? 'The IMD API requires the server IP to be whitelisted.' : 'Upstream request was rejected.'}`);
            return res.status(response.status).json({
                error: 'IMD request failed.',
                upstreamStatus: response.status
            });
        }

        return res.status(200).type('application/json').send(text);
    } catch (error) {
        console.warn('INFO: IMD district warning connector unavailable:', error.message);
        return res.status(502).json({
            error: 'Unable to reach IMD service.'
        });
    }
});

// =====================================================
// LIVE CURRENT HAZARD STATUS
// =====================================================

const currentHazardCache = {
    earthquakes: { timestamp: 0, data: null },
    tsunami: { timestamp: 0, data: null },
    landslide: { timestamp: 0, data: null },
    avalanche: { timestamp: 0, data: null }
};

const HAZARD_CACHE_MS = {
    earthquakes: 60 * 1000,
    tsunami: 2 * 60 * 1000,
    landslide: 10 * 60 * 1000,
    avalanche: 10 * 60 * 1000
};

function cacheFresh(entry, ttl) {
    return Boolean(entry?.data && Date.now() - entry.timestamp < ttl);
}

async function fetchTextWithTimeout(url, timeout = 8000) {
    const response = await fetch(url, {
        headers: {
            'User-Agent': 'RakshaNet/1.0',
            'Accept': 'text/html,application/json;q=0.9,*/*;q=0.8'
        },
        signal: AbortSignal.timeout(timeout)
    });

    const text = await response.text();
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return text;
}

function safeDistanceKm(lat1, lon1, lat2, lon2) {
    const toRad = v => v * Math.PI / 180;
    const R = 6371;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(lat1)) *
        Math.cos(toRad(lat2)) *
        Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

async function getLiveEarthquakes() {
    const cache = currentHazardCache.earthquakes;
    if (cacheFresh(cache, HAZARD_CACHE_MS.earthquakes)) return cache.data;

    const response = await fetch(
        'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson',
        {
            headers: {
                'User-Agent': 'RakshaNet/1.0',
                'Accept': 'application/geo+json,application/json'
            },
            signal: AbortSignal.timeout(8000)
        }
    );

    if (!response.ok) throw new Error(`USGS HTTP ${response.status}`);
    const data = await response.json();

    const events = Array.isArray(data?.features)
        ? data.features.map(event => {
            const coords = event?.geometry?.coordinates || [];
            return {
                id: event.id || '',
                magnitude: Number(event?.properties?.mag),
                time: Number(event?.properties?.time),
                place: event?.properties?.place || 'Unknown region',
                latitude: Number(coords[1]),
                longitude: Number(coords[0]),
                depth: Number(coords[2])
            };
        }).filter(e =>
            Number.isFinite(e.latitude) &&
            Number.isFinite(e.longitude) &&
            Number.isFinite(e.magnitude)
        )
        : [];

    const result = {
        source: 'USGS Earthquake Hazards Program',
        checkedAt: new Date().toISOString(),
        events
    };

    cache.timestamp = Date.now();
    cache.data = result;
    return result;
}

async function getLiveTsunamiStatus() {
    const cache = currentHazardCache.tsunami;
    if (cacheFresh(cache, HAZARD_CACHE_MS.tsunami)) return cache.data;

    const html = await fetchTextWithTimeout('https://tsunami.incois.gov.in/', 8000);
    const plain = html
        .replace(/<script[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style[\s\S]*?<\/style>/gi, ' ')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

    const section = plain.match(/Tsunami Updates\s+(.+?)\s+Storm Surge Updates/i)?.[1]?.trim() || '';

    let status = 'DATA UNAVAILABLE';
    if (/no tsunami/i.test(section)) status = 'NO ACTIVE WARNING';
    else if (/warning/i.test(section)) status = 'WARNING';
    else if (/watch|alert/i.test(section)) status = 'WATCH / ALERT';

    const result = {
        status,
        detail: section || 'INCOIS tsunami status could not be read.',
        source: 'INCOIS Indian Tsunami Early Warning System',
        checkedAt: new Date().toISOString()
    };

    cache.timestamp = Date.now();
    cache.data = result;
    return result;
}

async function getGsiLandslideCoverage() {
    const cache = currentHazardCache.landslide;
    if (cacheFresh(cache, HAZARD_CACHE_MS.landslide)) return cache.data;

    const html = await fetchTextWithTimeout('https://bhusanket.gsi.gov.in/', 8000);
    const plain = html
        .replace(/<script[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style[\s\S]*?<\/style>/gi, ' ')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

    const result = {
        portalAvailable: /National Landslide Forecasting Centre/i.test(plain),
        source: 'Geological Survey of India · National Landslide Forecasting Centre',
        checkedAt: new Date().toISOString()
    };

    cache.timestamp = Date.now();
    cache.data = result;
    return result;
}

async function getDgreAvalancheStatus() {
    const cache = currentHazardCache.avalanche;
    if (cacheFresh(cache, HAZARD_CACHE_MS.avalanche)) return cache.data;

    const html = await fetchTextWithTimeout(
        'https://drdo.gov.in/drdo/en/documents/avalanche-warning-bulletin-archive',
        8000
    );

    const plain = html
        .replace(/<script[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style[\s\S]*?<\/style>/gi, ' ')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

    const closed = /Closing of Operational Avalanche Forecasting Services/i.test(plain);

    const result = {
        status: closed ? 'NO CURRENT OPERATIONAL BULLETIN' : 'BULLETIN SOURCE AVAILABLE',
        detail: closed
            ? 'DRDO/DGRE archive records closure of the 2025-26 operational avalanche forecasting season; no current operational bulletin was identified.'
            : 'Use the current DGRE/DRDO avalanche bulletin for location-specific status.',
        source: 'DRDO · Defence Geoinformatics Research Establishment (DGRE)',
        checkedAt: new Date().toISOString()
    };

    cache.timestamp = Date.now();
    cache.data = result;
    return result;
}

app.get('/api/current-hazards', async (req, res) => {
    try {
        const lat = Number(req.query.lat);
        const lon = Number(req.query.lon);

        if (!Number.isFinite(lat) || !Number.isFinite(lon) ||
            lat < -90 || lat > 90 || lon < -180 || lon > 180) {
            return res.status(400).json({
                error: 'Valid latitude and longitude are required.'
            });
        }

        const [eqRes, tsRes, lsRes, avRes, floodRes] = await Promise.allSettled([
            getLiveEarthquakes(),
            getLiveTsunamiStatus(),
            getGsiLandslideCoverage(),
            getDgreAvalancheStatus(),
            fetchOpenMeteoFlood(lat, lon)
        ]);

        let earthquake;

        if (eqRes.status === 'fulfilled') {
            const events = eqRes.value.events
                .map(e => ({
                    ...e,
                    distanceKm: safeDistanceKm(lat, lon, e.latitude, e.longitude)
                }))
                .sort((a, b) => a.distanceKm - b.distanceKm);

            const local = events.find(e => e.distanceKm <= 100);
            const significant = events.find(
                e => e.distanceKm <= 250 && e.magnitude >= 4.5
            );

            if (local) {
                earthquake = {
                    status: local.magnitude >= 4.5 ? 'SIGNIFICANT ACTIVITY' : 'CURRENT ACTIVITY',
                    detail: `M${local.magnitude.toFixed(1)} · ${local.distanceKm.toFixed(0)} km away · ${local.place}`,
                    source: 'USGS live earthquake feed',
                    checkedAt: eqRes.value.checkedAt,
                    event: local
                };
            } else if (significant) {
                earthquake = {
                    status: 'DISTANT SIGNIFICANT ACTIVITY',
                    detail: `M${significant.magnitude.toFixed(1)} · ${significant.distanceKm.toFixed(0)} km away · ${significant.place}`,
                    source: 'USGS live earthquake feed',
                    checkedAt: eqRes.value.checkedAt,
                    event: significant
                };
            } else {
                earthquake = {
                    status: 'NO RECENT LOCAL ACTIVITY',
                    detail: 'No USGS earthquake in the last 24 hours was found within 100 km of the selected point.',
                    source: 'USGS live earthquake feed · NCS official portal available for corroboration',
                    checkedAt: eqRes.value.checkedAt
                };
            }
        } else {
            earthquake = {
                status: 'DATA UNAVAILABLE',
                detail: eqRes.reason?.message || 'USGS earthquake feed unavailable.',
                source: 'USGS live earthquake feed',
                checkedAt: new Date().toISOString()
            };
        }

        const landslide =
            lsRes.status === 'fulfilled' && lsRes.value.portalAvailable
                ? {
                    status: 'OFFICIAL BULLETIN COVERAGE VARIES',
                    detail: 'GSI publishes live/experimental landslide forecasts for selected districts. A public machine-readable location-specific bulletin was not identified here, so RakshaNet does not invent a local risk level.',
                    source: lsRes.value.source,
                    checkedAt: lsRes.value.checkedAt
                }
                : {
                    status: 'DATA UNAVAILABLE',
                    detail: lsRes.reason?.message || 'GSI NLFC portal unavailable.',
                    source: 'GSI National Landslide Forecasting Centre',
                    checkedAt: new Date().toISOString()
                };

        const avalanche = avRes.status === 'fulfilled'
            ? avRes.value
            : {
                status: 'DATA UNAVAILABLE',
                detail: avRes.reason?.message || 'DGRE/DRDO bulletin archive unavailable.',
                source: 'DRDO / DGRE',
                checkedAt: new Date().toISOString()
            };

        const tsunami = tsRes.status === 'fulfilled'
            ? tsRes.value
            : {
                status: 'DATA UNAVAILABLE',
                detail: tsRes.reason?.message || 'INCOIS service unavailable.',
                source: 'INCOIS Indian Tsunami Early Warning System',
                checkedAt: new Date().toISOString()
            };

        const flood = floodRes.status === 'fulfilled'
            ? buildHydrologySummary(floodRes.value)
            : {
                available: false,
                status: 'DATA UNAVAILABLE',
                detail: floodRes.reason?.message || 'Flood hydrology feed unavailable.',
                source: 'GloFAS v4 via Open-Meteo Flood API',
                checkedAt: new Date().toISOString()
            };

        const weatherForHazard = await fetchOpenMeteoIntelligence(lat, lon).catch(() => null);
        const geo = await reverseGeocodeLocation(lat, lon).catch(() => ({ district: null, state: null, city: null, displayName: null }));
        const sachetFeed = await fetchSachetAlerts();
        const sachet = buildSachetHazardSummary(sachetFeed, geo, 'auto', lat, lon);
        const weatherHazards = summarizeWeatherHazards(weatherForHazard, flood);
        const riskBundle = buildMultiHazardRisk({ hazardSignals: { ...weatherHazards, landslide: 0, cyclone: 0 }, officialWarning: null, nationalAlerts: sachet });

        return res.json({
            ok: true,
            latitude: lat,
            longitude: lon,
            checkedAt: new Date().toISOString(),
            hazards: { earthquake, landslide, avalanche, tsunami, flood, official: sachet, derived: { floodIndex: riskBundle.scores.flood, heavyRainIndex: riskBundle.scores.heavy_rain } },
            location: { ...geo },
            sourceStatus: { sachet: { available: sachet.available, matched: sachet.matched, checkedAt: sachet.checkedAt, error: sachet.error || null } }
        });
    } catch (error) {
        console.error('Current hazard status error:', error);
        return res.status(502).json({
            error: 'Unable to load current hazard status.',
            details: error.message
        });
    }
});

// =====================================================
// RAKSHANET LIVE SENSOR INTELLIGENCE
// National Water Data Portal -> RakshaNet
// Current connected feeds:
// 1. Assam River Water Level - Telemetry Hourly
// 2. Assam Rainfall - Telemetry Hourly
// 3. Assam Temperature - Telemetry Hourly
// 4. Assam Relative Humidity - Telemetry Hourly
// 5. Assam Wind Speed - Telemetry Hourly
// 6. Assam Wind Direction - Telemetry Hourly
// 7. Assam Atmospheric Pressure - Telemetry Hourly
// =====================================================

const SENSOR_CACHE = {
    timestamp: 0,
    data: null
};

const SENSOR_CACHE_MS = 5 * 60 * 1000;

const NWDP_SENSOR_SOURCES = {
    riverLevel:
        'https://nwdp.nwic.gov.in/dataset/6273c426-32f9-4fdf-b67f-e4e7a46d8554/resource/847f5630-f231-46c0-922d-0f2f379a5cb8/download/rwl_tel_hr_assam_999_2026_2030.csv',

    rainfall:
        'https://nwdp.nwic.gov.in/dataset/7c8adfd6-2f7e-4bcc-b23d-f22a815e4b3c/resource/e72d7889-0ab9-494f-8e6c-e5a85b71c9fb/download/rainfall_tel_hr_assam_as_2026_2030.csv',

    temperature:
        'https://nwdp.nwic.gov.in/dataset/a85d971b-17d3-4c5f-8a6c-0b5aec66896d/resource/676e0e2c-27bc-4f63-8d28-55b7c9d683c4/download/temprature_tel_hr_assam_as_2026_2030.csv',

    humidity:
        'https://nwdp.nwic.gov.in/dataset/3d4f12bf-7cd4-49ce-b332-b2a11414485c/resource/c592b2e4-858e-4a5d-b037-702441b162c8/download/humid_tel_hr_assam_as_2026_2030.csv',

    windSpeed:
        'https://nwdp.nwic.gov.in/dataset/74de7025-f99f-43d4-b80d-b5b5c57e2bf0/resource/57d0b524-bf71-4b54-b67e-63b0434c31d2/download/wind_speed_tel_hr_assam_as_2026_2030.csv',

    windDirection:
        'https://nwdp.nwic.gov.in/dataset/72c4e99f-d18a-44ba-86c4-f2b6f029e612/resource/ce0535a4-5869-40ef-971f-9d9f3a321c85/download/wind_direction_tel_hr_assam_as_2026_2030.csv',

    pressure:
        'https://nwdp.nwic.gov.in/dataset/93697d51-5922-4102-9624-12b3d4d7d914/resource/684b5322-2ccd-4d76-b87d-6d2f6ba03d17/download/pressure_tel_hr_assam_as_2026_2030.csv'
};

function parseCsvLine(line) {
    const values = [];
    let current = '';
    let insideQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];

        if (char === '"') {
            if (insideQuotes && line[i + 1] === '"') {
                current += '"';
                i++;
            } else {
                insideQuotes = !insideQuotes;
            }
            continue;
        }

        if (char === ',' && !insideQuotes) {
            values.push(current.trim());
            current = '';
            continue;
        }

        current += char;
    }

    values.push(current.trim());
    return values;
}

function parseCsv(text) {
    const cleaned = String(text || '')
        .replace(/^\uFEFF/, '')
        .replace(/\r/g, '');

    const lines = cleaned
        .split('\n')
        .filter(line => line.trim());

    if (lines.length < 2) {
        return [];
    }

    const headers = parseCsvLine(lines[0]);
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
        const values = parseCsvLine(lines[i]);
        if (values.length === 0) continue;

        const row = {};

        headers.forEach((header, index) => {
            row[String(header).trim()] = values[index] ?? '';
        });

        rows.push(row);
    }

    return rows;
}

function findCsvColumn(headers, possibleNames) {
    const normalizedHeaders = headers.map(header => ({
        original: header,
        normalized: String(header)
            .toLowerCase()
            .replace(/[^a-z0-9]/g, '')
    }));

    for (const possibleName of possibleNames) {
        const target = String(possibleName)
            .toLowerCase()
            .replace(/[^a-z0-9]/g, '');

        const exact = normalizedHeaders.find(
            item => item.normalized === target
        );

        if (exact) return exact.original;
    }

    for (const possibleName of possibleNames) {
        const target = String(possibleName)
            .toLowerCase()
            .replace(/[^a-z0-9]/g, '');

        const partial = normalizedHeaders.find(
            item =>
                item.normalized.includes(target) ||
                target.includes(item.normalized)
        );

        if (partial) return partial.original;
    }

    return null;
}

function numericValue(value) {
    if (value === null || value === undefined) return null;

    const raw = String(value)
        .replace(/,/g, '')
        .trim();

    if (!raw) return null;

    const direct = Number(raw);
    if (Number.isFinite(direct)) return direct;

    // Accept measurements that include units or short labels.
    const match = raw.match(
        /[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/
    );

    if (!match) return null;

    const extracted = Number(match[0]);
    return Number.isFinite(extracted) ? extracted : null;
}

function parseObservationDate(value) {
    if (!value) return null;

    const raw = String(value).trim();

    // DD-MM-YYYY HH:MM[:SS]
    const dmy = raw.match(
        /^(\d{1,2})-(\d{1,2})-(\d{4})(?:\s+(\d{1,2}):?(\d{2})(?::?(\d{2}))?)?$/
    );

    if (dmy) {
        const day = Number(dmy[1]);
        const month = Number(dmy[2]) - 1;
        const year = Number(dmy[3]);
        const hour = Number(dmy[4] || 0);
        const minute = Number(dmy[5] || 0);
        const second = Number(dmy[6] || 0);

        const utcMs = Date.UTC(year, month, day, hour, minute, second) - (5 * 60 + 30) * 60 * 1000;
        const date = new Date(utcMs);

        if (!Number.isNaN(date.getTime())) return date;
    }

    const parsed = new Date(raw);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function buildTimestamp(row, dateColumn, timeColumn) {
    let timestamp = dateColumn
        ? String(row[dateColumn] ?? '').trim()
        : '';

    if (timeColumn && row[timeColumn]) {
        timestamp = `${timestamp} ${String(row[timeColumn]).trim()}`.trim();
    }

    return timestamp || null;
}

async function fetchSensorCsv(url) {
    const response = await fetch(url, {
        method: 'GET',
        headers: {
            'User-Agent': 'RakshaNet/1.0',
            'Accept': 'text/csv,text/plain,*/*'
        },
        signal: AbortSignal.timeout(30000)
    });

    if (!response.ok) {
        throw new Error(`Sensor source returned HTTP ${response.status}`);
    }

    return await response.text();
}

// -----------------------------------------------------
// Process River Water Level
// -----------------------------------------------------

function processRiverRows(rows) {

    if (
        !Array.isArray(rows) ||
        rows.length === 0
    ) {
        return {
            latest: null,
            stations: 0
        };
    }

    const headers =
        Object.keys(rows[0] || {});

    if (process.env.RAKSHANET_DEBUG === 'true') console.log('River sensor CSV headers:', headers);

    const normalizedHeaders =
        headers.map(header => ({
            original: header,
            normalized:
                String(header)
                    .toLowerCase()
                    .replace(/[^a-z0-9]/g, '')
        }));

    const dateColumn =
        normalizedHeaders.find(item =>
            [
                'date',
                'datetime',
                'timestamp',
                'observationdate',
                'observationdatetime'
            ].some(name =>
                item.normalized === name ||
                item.normalized.includes(name)
            )
        )?.original || null;

    const timeColumn =
        normalizedHeaders.find(item =>
            item.normalized === 'time' ||
            item.normalized === 'observationtime'
        )?.original || null;

    const stationColumn =
        normalizedHeaders.find(item => {

            const h =
                item.normalized;

            return (
                h.includes('stationname') ||
                h === 'station' ||
                h.includes('stationid') ||
                h.includes('stationidentifier') ||
                h.includes('stationcode') ||
                h.includes('observationstation')
            );

        })?.original || null;

    const districtColumn =
        normalizedHeaders.find(item => {

            const h =
                item.normalized;

            return (
                h === 'district' ||
                h.includes('districtname')
            );

        })?.original || null;

    const stateColumn = normalizedHeaders.find(item => ['state','statename'].includes(item.normalized))?.original || null;
    const latitudeColumn = normalizedHeaders.find(item => ['latitude','lat'].includes(item.normalized))?.original || null;
    const longitudeColumn = normalizedHeaders.find(item => ['longitude','lon','lng'].includes(item.normalized))?.original || null;
    const acquisitionColumn = normalizedHeaders.find(item => item.normalized.includes('dataacquisitiontime'))?.original || null;

    const riverColumn =
        normalizedHeaders.find(item => {

            const h =
                item.normalized;

            return (
                h === 'river' ||
                h.includes('rivername')
            );

        })?.original || null;

    let valueColumn =
        normalizedHeaders.find(item => {

            const h =
                item.normalized;

            return (
                h.includes('riverwaterlevel') ||
                h.includes('waterlevel') ||
                h.includes('waterheight') ||
                h.includes('rwl') ||
                h === 'level' ||
                h.includes('levelm')
            );

        })?.original || null;

    if (!valueColumn) {

        const excludedNames =
            new Set(
                [
                    dateColumn,
                    timeColumn,
                    stationColumn,
                    districtColumn,
                    riverColumn
                ].filter(Boolean)
            );

        const numericCandidates =
            headers
                .filter(
                    header =>
                        !excludedNames.has(header)
                )
                .map(header => {

                    let numericCount = 0;

                    for (const row of rows) {

                        const value =
                            numericValue(
                                row[header]
                            );

                        if (
                            value !== null
                        ) {
                            numericCount++;
                        }
                    }

                    return {
                        header,
                        numericCount
                    };

                })
                .filter(
                    candidate =>
                        candidate.numericCount > 0
                )
                .sort(
                    (a, b) =>
                        b.numericCount -
                        a.numericCount
                );

        if (
            numericCandidates.length > 0
        ) {

            valueColumn =
                numericCandidates[0].header;

            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('River level column fallback:', valueColumn);
        }
    }

    if (process.env.RAKSHANET_DEBUG === 'true') console.log('River parser columns:', { dateColumn, timeColumn, stationColumn, districtColumn, riverColumn, valueColumn });

    const processed =
        rows
            .map(row => {

                let timestamp = '';

                if (dateColumn) {

                    timestamp =
                        String(
                            row[dateColumn] ?? ''
                        ).trim();
                }

                if (
                    timeColumn &&
                    row[timeColumn]
                ) {

                    timestamp =
                        `${timestamp} ${String(
                            row[timeColumn]
                        ).trim()}`;
                }

                if (!timestamp && acquisitionColumn) timestamp = String(row[acquisitionColumn] ?? '').trim();

                const waterLevel =
                    valueColumn
                        ? numericValue(
                            row[valueColumn]
                        )
                        : null;

                return {

                    timestamp:
                        timestamp || null,

                    station:
                        stationColumn
                            ? String(
                                row[stationColumn] ?? ''
                            ).trim()
                            : 'Unknown station',

                    district:
                        districtColumn
                            ? String(
                                row[districtColumn] ?? ''
                            ).trim()
                            : '',

                    state: stateColumn ? String(row[stateColumn] ?? '').trim() : '',
                    latitude: latitudeColumn ? numericValue(row[latitudeColumn]) : null,
                    longitude: longitudeColumn ? numericValue(row[longitudeColumn]) : null,

                    river:
                        riverColumn
                            ? String(
                                row[riverColumn] ?? ''
                            ).trim()
                            : '',

                    waterLevel
                };

            })
            .filter(
                row =>
                    row.waterLevel !== null
            );

    processed.sort((a, b) => {
        const aDate = parseObservationDate(a.timestamp);
        const bDate = parseObservationDate(b.timestamp);
        const aTime = aDate ? aDate.getTime() : 0;
        const bTime = bDate ? bDate.getTime() : 0;
        return bTime - aTime;
    });

    const latest =
        processed[0] || null;

    const stationSet =
        new Set();

    processed.forEach(
        row => {

            const station =
                String(
                    row.station || ''
                ).trim();

            if (
                station &&
                station !== 'Unknown station'
            ) {

                stationSet.add(
                    station
                );
            }
        }
    );

    return {

        latest,

        stations:
            stationSet.size
    };
}

// -----------------------------------------------------
// Process Rainfall
// -----------------------------------------------------

function processRainfallRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        return { latest: null, stations: 0 };
    }

    const headers = Object.keys(rows[0] || {});

    const dateColumn = findCsvColumn(headers, [
        'date',
        'datetime',
        'timestamp',
        'observation date',
        'observation datetime'
    ]);

    const timeColumn = findCsvColumn(headers, [
        'time',
        'observation time'
    ]);

    const stationColumn = findCsvColumn(headers, [
        'station name',
        'station',
        'station id',
        'station identifier',
        'station code',
        'observation station',
        'location'
    ]);

    const districtColumn = findCsvColumn(headers, [
        'district',
        'district name'
    ]);

    let valueColumn = findCsvColumn(headers, [
        'rainfall',
        'rainfall mm',
        'rain',
        'rain mm'
    ]);

    if (!valueColumn) {
        const excludedNames = new Set(
            [dateColumn, timeColumn, stationColumn, districtColumn]
                .filter(Boolean)
        );

        const numericCandidates = headers
            .filter(header => !excludedNames.has(header))
            .map(header => ({
                header,
                numericCount: rows.filter(
                    row => numericValue(row[header]) !== null
                ).length
            }))
            .filter(candidate => candidate.numericCount > 0)
            .sort((a, b) => b.numericCount - a.numericCount);

        if (numericCandidates.length > 0) {
            valueColumn = numericCandidates[0].header;
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Rainfall column was not matched by name. Using numeric fallback:', valueColumn);
        }
    }

    const processed = rows
        .map(row => ({
            timestamp: buildTimestamp(row, dateColumn, timeColumn),
            station: stationColumn
                ? String(row[stationColumn] ?? '').trim()
                : 'Unknown station',
            district: districtColumn
                ? String(row[districtColumn] ?? '').trim()
                : '',
            rainfall: valueColumn
                ? numericValue(row[valueColumn])
                : null
        }))
        .filter(row => row.rainfall !== null);

    processed.sort((a, b) => {
        const aTime = parseObservationDate(a.timestamp);
        const bTime = parseObservationDate(b.timestamp);
        return (bTime?.getTime() || 0) - (aTime?.getTime() || 0);
    });

    const stationSet = new Set();
    processed.forEach(row => {
        const station = String(row.station || '').trim();
        if (station) stationSet.add(station);
    });

    return {
        latest: processed[0] || null,
        stations: stationSet.size
    };
}

// -----------------------------------------------------
// Process Temperature
// -----------------------------------------------------

function processTemperatureRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        return { latest: null, stations: 0 };
    }

    const headers = Object.keys(rows[0] || {});

    const dateColumn = findCsvColumn(headers, [
        'date',
        'datetime',
        'timestamp',
        'observation date',
        'observation datetime'
    ]);

    const timeColumn = findCsvColumn(headers, [
        'time',
        'observation time'
    ]);

    const stationColumn = findCsvColumn(headers, [
        'station name',
        'station',
        'station id',
        'station identifier',
        'station code',
        'observation station',
        'location'
    ]);

    const districtColumn = findCsvColumn(headers, [
        'district',
        'district name'
    ]);

    let valueColumn = findCsvColumn(headers, [
        'temperature',
        'temperature c',
        'temperature (c)',
        'temperature °c',
        'temp',
        'temp c',
        'temp (c)'
    ]);

    if (!valueColumn) {
        const excludedNames = new Set(
            [dateColumn, timeColumn, stationColumn, districtColumn]
                .filter(Boolean)
        );

        const numericCandidates = headers
            .filter(header => !excludedNames.has(header))
            .map(header => ({
                header,
                numericCount: rows.filter(
                    row => numericValue(row[header]) !== null
                ).length
            }))
            .filter(candidate => candidate.numericCount > 0)
            .sort((a, b) => b.numericCount - a.numericCount);

        if (numericCandidates.length > 0) {
            valueColumn = numericCandidates[0].header;
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Temperature column was not matched by name. Using numeric fallback:', valueColumn);
        }
    }

    const processed = rows
        .map(row => ({
            timestamp: buildTimestamp(row, dateColumn, timeColumn),
            station: stationColumn
                ? String(row[stationColumn] ?? '').trim()
                : 'Unknown station',
            district: districtColumn
                ? String(row[districtColumn] ?? '').trim()
                : '',
            temperature: valueColumn
                ? numericValue(row[valueColumn])
                : null
        }))
        .filter(row => row.temperature !== null);

    processed.sort((a, b) => {
        const aTime = parseObservationDate(a.timestamp);
        const bTime = parseObservationDate(b.timestamp);
        return (bTime?.getTime() || 0) - (aTime?.getTime() || 0);
    });

    const stationSet = new Set();
    processed.forEach(row => {
        const station = String(row.station || '').trim();
        if (station) stationSet.add(station);
    });

    return {
        latest: processed[0] || null,
        stations: stationSet.size
    };
}

// -----------------------------------------------------
// Process Relative Humidity
// -----------------------------------------------------

function processHumidityRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        return { latest: null, stations: 0 };
    }

    const headers = Object.keys(rows[0] || {});

    const dateColumn = findCsvColumn(headers, [
        'date',
        'datetime',
        'timestamp',
        'observation date',
        'observation datetime'
    ]);

    const timeColumn = findCsvColumn(headers, [
        'time',
        'observation time'
    ]);

    const stationColumn = findCsvColumn(headers, [
        'station name',
        'station',
        'station id',
        'station identifier',
        'station code',
        'observation station',
        'location'
    ]);

    const districtColumn = findCsvColumn(headers, [
        'district',
        'district name'
    ]);

    let valueColumn = findCsvColumn(headers, [
        'relative humidity',
        'relative humidity (%)',
        'humidity',
        'humidity (%)',
        'rh',
        'rh (%)',
        'relative humidty'
    ]);

    if (!valueColumn) {
        const excludedNames = new Set(
            [dateColumn, timeColumn, stationColumn, districtColumn]
                .filter(Boolean)
        );

        const numericCandidates = headers
            .filter(header => !excludedNames.has(header))
            .map(header => ({
                header,
                numericCount: rows.filter(
                    row => numericValue(row[header]) !== null
                ).length
            }))
            .filter(candidate => candidate.numericCount > 0)
            .sort((a, b) => b.numericCount - a.numericCount);

        if (numericCandidates.length > 0) {
            valueColumn = numericCandidates[0].header;
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Humidity column was not matched by name. Using numeric fallback:', valueColumn);
        }
    }

    const processed = rows
        .map(row => ({
            timestamp: buildTimestamp(row, dateColumn, timeColumn),
            station: stationColumn
                ? String(row[stationColumn] ?? '').trim()
                : 'Unknown station',
            district: districtColumn
                ? String(row[districtColumn] ?? '').trim()
                : '',
            humidity: valueColumn
                ? numericValue(row[valueColumn])
                : null
        }))
        .filter(row => row.humidity !== null);

    processed.sort((a, b) => {
        const aTime = parseObservationDate(a.timestamp);
        const bTime = parseObservationDate(b.timestamp);
        return (bTime?.getTime() || 0) - (aTime?.getTime() || 0);
    });

    const stationSet = new Set();
    processed.forEach(row => {
        const station = String(row.station || '').trim();
        if (station) stationSet.add(station);
    });

    return {
        latest: processed[0] || null,
        stations: stationSet.size
    };
}


// -----------------------------------------------------
// Generic metadata helpers for meteorological feeds
// -----------------------------------------------------

function findStationColumns(headers) {
    const dateColumn = findCsvColumn(headers, [
        'date',
        'datetime',
        'timestamp',
        'observation date',
        'observation datetime'
    ]);

    const timeColumn = findCsvColumn(headers, [
        'time',
        'observation time'
    ]);

    const stationColumn = findCsvColumn(headers, [
        'station name',
        'station',
        'station id',
        'station identifier',
        'station code',
        'observation station',
        'location'
    ]);

    const districtColumn = findCsvColumn(headers, [
        'district',
        'district name'
    ]);

    return {
        dateColumn,
        timeColumn,
        stationColumn,
        districtColumn
    };
}

function uniqueStations(rows) {
    const stationSet = new Set();

    rows.forEach(row => {
        const station = String(row.station || '').trim();
        if (station && station !== 'Unknown station') {
            stationSet.add(station);
        }
    });

    return stationSet.size;
}

function sortNewestFirst(rows) {
    rows.sort((a, b) => {
        const aTime = parseObservationDate(a.timestamp);
        const bTime = parseObservationDate(b.timestamp);
        return (bTime?.getTime() || 0) - (aTime?.getTime() || 0);
    });

    return rows;
}

// -----------------------------------------------------
// Process Wind Speed
// NWDP generally reports wind speed in km/h.
// -----------------------------------------------------

function processWindSpeedRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        return { latest: null, stations: 0 };
    }

    const headers = Object.keys(rows[0] || {});
    const meta = findStationColumns(headers);

    let valueColumn = findCsvColumn(headers, [
        'wind speed',
        'wind speed km/h',
        'wind speed (km/h)',
        'wind speed(km/h)',
        'windspeed',
        'windspeedkmh',
        'speed'
    ]);

    if (!valueColumn) {
        const excluded = new Set([
            meta.dateColumn,
            meta.timeColumn,
            meta.stationColumn,
            meta.districtColumn
        ].filter(Boolean));

        const candidates = headers
            .filter(header => !excluded.has(header))
            .map(header => ({
                header,
                count: rows.filter(
                    row => numericValue(row[header]) !== null
                ).length
            }))
            .filter(item => item.count > 0)
            .sort((a, b) => b.count - a.count);

        if (candidates.length) {
            valueColumn = candidates[0].header;
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Wind speed column fallback:', valueColumn);
        }
    }

    const processed = rows
        .map(row => ({
            timestamp:
                buildTimestamp(
                    row,
                    meta.dateColumn,
                    meta.timeColumn
                ),

            station:
                meta.stationColumn
                    ? String(
                        row[meta.stationColumn] ?? ''
                    ).trim()
                    : 'Unknown station',

            district:
                meta.districtColumn
                    ? String(
                        row[meta.districtColumn] ?? ''
                    ).trim()
                    : '',

            windSpeed:
                valueColumn
                    ? numericValue(
                        row[valueColumn]
                    )
                    : null
        }))
        .filter(row => row.windSpeed !== null);

    sortNewestFirst(processed);

    return {
        latest: processed[0] || null,
        stations: uniqueStations(processed)
    };
}

// -----------------------------------------------------
// Process Wind Direction
// Direction may be degrees or a textual compass value.
// -----------------------------------------------------

function processWindDirectionRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        return { latest: null, stations: 0 };
    }

    const headers = Object.keys(rows[0] || {});
    const meta = findStationColumns(headers);

    let directionColumn = findCsvColumn(headers, [
        'wind direction',
        'wind direction (degrees)',
        'wind direction(degrees)',
        'wind direction degree',
        'wind direction degrees',
        'winddirection',
        'direction',
        'dir'
    ]);

    if (!directionColumn) {
        const excluded = new Set([
            meta.dateColumn,
            meta.timeColumn,
            meta.stationColumn,
            meta.districtColumn
        ].filter(Boolean));

        const candidates = headers
            .filter(header => !excluded.has(header))
            .map(header => ({
                header,
                textCount: rows.filter(row => {
                    const value = String(
                        row[header] ?? ''
                    ).trim();
                    return value.length > 0;
                }).length
            }))
            .filter(item => item.textCount > 0)
            .sort((a, b) => b.textCount - a.textCount);

        if (candidates.length) {
            directionColumn = candidates[0].header;
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Wind direction column fallback:', directionColumn);
        }
    }

    const processed = rows
        .map(row => {
            const rawDirection =
                directionColumn
                    ? String(
                        row[directionColumn] ?? ''
                    ).trim()
                    : '';

            const numericDirection =
                numericValue(rawDirection);

            return {
                timestamp:
                    buildTimestamp(
                        row,
                        meta.dateColumn,
                        meta.timeColumn
                    ),

                station:
                    meta.stationColumn
                        ? String(
                            row[meta.stationColumn] ?? ''
                        ).trim()
                        : 'Unknown station',

                district:
                    meta.districtColumn
                        ? String(
                            row[meta.districtColumn] ?? ''
                        ).trim()
                        : '',

                windDirection:
                    rawDirection || null,

                windDirectionDegrees:
                    numericDirection
            };
        })
        .filter(row => row.windDirection !== null);

    sortNewestFirst(processed);

    return {
        latest: processed[0] || null,
        stations: uniqueStations(processed)
    };
}

// -----------------------------------------------------
// Process Atmospheric Pressure
// NWDP reports pressure in Millibar (Mb).
// -----------------------------------------------------

function processPressureRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        return { latest: null, stations: 0 };
    }

    const headers = Object.keys(rows[0] || {});
    const meta = findStationColumns(headers);

    let valueColumn = findCsvColumn(headers, [
        'atmospheric pressure',
        'atmospheric pressure (mb)',
        'atmospheric pressure(mb)',
        'pressure',
        'pressure mb',
        'pressure (mb)',
        'pressuremb',
        'barometric pressure'
    ]);

    if (!valueColumn) {
        const excluded = new Set([
            meta.dateColumn,
            meta.timeColumn,
            meta.stationColumn,
            meta.districtColumn
        ].filter(Boolean));

        const candidates = headers
            .filter(header => !excluded.has(header))
            .map(header => ({
                header,
                count: rows.filter(
                    row => numericValue(row[header]) !== null
                ).length
            }))
            .filter(item => item.count > 0)
            .sort((a, b) => b.count - a.count);

        if (candidates.length) {
            valueColumn = candidates[0].header;
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pressure column fallback:', valueColumn);
        }
    }

    const processed = rows
        .map(row => ({
            timestamp:
                buildTimestamp(
                    row,
                    meta.dateColumn,
                    meta.timeColumn
                ),

            station:
                meta.stationColumn
                    ? String(
                        row[meta.stationColumn] ?? ''
                    ).trim()
                    : 'Unknown station',

            district:
                meta.districtColumn
                    ? String(
                        row[meta.districtColumn] ?? ''
                    ).trim()
                    : '',

            pressure:
                valueColumn
                    ? numericValue(
                        row[valueColumn]
                    )
                    : null
        }))
        .filter(row => row.pressure !== null);

    sortNewestFirst(processed);

    return {
        latest: processed[0] || null,
        stations: uniqueStations(processed)
    };
}


// -----------------------------------------------------
// MAIN SENSOR LOADER
// -----------------------------------------------------

async function getAssamSensorData() {
    if (
        SENSOR_CACHE.data &&
        Date.now() - SENSOR_CACHE.timestamp < SENSOR_CACHE_MS
    ) {
        return SENSOR_CACHE.data;
    }

    const [
        riverResult,
        rainfallResult,
        temperatureResult,
        humidityResult,
        windSpeedResult,
        windDirectionResult,
        pressureResult
    ] = await Promise.allSettled([
        fetchSensorCsv(NWDP_SENSOR_SOURCES.riverLevel),
        fetchSensorCsv(NWDP_SENSOR_SOURCES.rainfall),
        fetchSensorCsv(NWDP_SENSOR_SOURCES.temperature),
        fetchSensorCsv(NWDP_SENSOR_SOURCES.humidity),
        fetchSensorCsv(NWDP_SENSOR_SOURCES.windSpeed),
        fetchSensorCsv(NWDP_SENSOR_SOURCES.windDirection),
        fetchSensorCsv(NWDP_SENSOR_SOURCES.pressure)
    ]);

    let river = null;
    let rainfall = null;
    let temperature = null;
    let humidity = null;
    let windSpeed = null;
    let windDirection = null;
    let pressure = null;

    if (riverResult.status === 'fulfilled') {
        try {
            river = processRiverRows(parseCsv(riverResult.value));
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('River CSV processing error:', error.message);
        }
    } else {
        if (process.env.RAKSHANET_DEBUG === 'true') console.warn('River sensor source failed:', riverResult.reason?.message || riverResult.reason);
    }

    if (rainfallResult.status === 'fulfilled') {
        try {
            rainfall = processRainfallRows(parseCsv(rainfallResult.value));
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Rainfall CSV processing error:', error.message);
        }
    } else {
        if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Rainfall sensor source failed:', rainfallResult.reason?.message || rainfallResult.reason);
    }

    if (temperatureResult.status === 'fulfilled') {
        try {
            temperature = processTemperatureRows(parseCsv(temperatureResult.value));
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Temperature CSV processing error:', error.message);
        }
    } else {
        if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Temperature sensor source failed:', temperatureResult.reason?.message || temperatureResult.reason);
    }

    if (humidityResult.status === 'fulfilled') {
        try {
            humidity = processHumidityRows(parseCsv(humidityResult.value));
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Humidity CSV processing error:', error.message);
        }
    } else {
        if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Humidity sensor source failed:', humidityResult.reason?.message || humidityResult.reason);
    }

    if (windSpeedResult.status === 'fulfilled') {
        try {
            windSpeed = processWindSpeedRows(
                parseCsv(windSpeedResult.value)
            );
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Wind speed CSV processing error:', error.message);
        }
    } else {
        if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Wind speed sensor source failed:', windSpeedResult.reason?.message || windSpeedResult.reason);
    }

    if (windDirectionResult.status === 'fulfilled') {
        try {
            windDirection = processWindDirectionRows(
                parseCsv(windDirectionResult.value)
            );
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Wind direction CSV processing error:', error.message || error);
        }
    } else {
        if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Wind direction sensor source failed:', windDirectionResult.reason?.message || windDirectionResult.reason);
    }

    if (pressureResult.status === 'fulfilled') {
        try {
            pressure = processPressureRows(
                parseCsv(pressureResult.value)
            );
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pressure CSV processing error:', error.message);
        }
    } else {
        if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pressure sensor source failed:', pressureResult.reason?.message || pressureResult.reason);
    }

    const data = {
        ok: Boolean(
            river?.latest ||
            rainfall?.latest ||
            temperature?.latest ||
            humidity?.latest ||
            windSpeed?.latest ||
            windDirection?.latest ||
            pressure?.latest
        ),
        source: 'National Water Data Portal · Assam Water Department',
        updatedAt: new Date().toISOString(),
        feedCount: 7,
        feeds: {
            riverLevel: {
                connected: Boolean(river?.latest),
                latest: river?.latest || null,
                stationCount: river?.stations || 0
            },
            rainfall: {
                connected: Boolean(rainfall?.latest),
                latest: rainfall?.latest || null,
                stationCount: rainfall?.stations || 0
            },
            temperature: {
                connected: Boolean(temperature?.latest),
                latest: temperature?.latest || null,
                stationCount: temperature?.stations || 0
            },
            humidity: {
                connected: Boolean(humidity?.latest),
                latest: humidity?.latest || null,
                stationCount: humidity?.stations || 0
            },
            windSpeed: {
                connected: Boolean(windSpeed?.latest),
                latest: windSpeed?.latest || null,
                stationCount: windSpeed?.stations || 0
            },
            windDirection: {
                connected: Boolean(windDirection?.latest),
                latest: windDirection?.latest || null,
                stationCount: windDirection?.stations || 0
            },
            pressure: {
                connected: Boolean(pressure?.latest),
                latest: pressure?.latest || null,
                stationCount: pressure?.stations || 0
            },
            soilMoisture: {
                connected: false,
                message: 'No direct soil-moisture telemetry feed connected yet.'
            },
            groundMovement: {
                connected: false,
                message: 'Ground-movement telemetry will be connected later.'
            }
        }
    };

    SENSOR_CACHE.timestamp = Date.now();
    SENSOR_CACHE.data = data;
    return data;
}

app.get('/api/sensors/assam', async (_req, res) => {
    try {
        const data = await getAssamSensorData();
        return res.json(data);
    } catch (error) {
        console.error('Assam sensor API error:', error);
        return res.status(502).json({
            ok: false,
            error: 'Unable to load Assam telemetry data.',
            details: error.message
        });
    }
});

app.get('/api/sensors/status', async (_req, res) => {
    try {
        const data = await getAssamSensorData();
        const feeds = data.feeds || {};

        const stationCount = [
            feeds.riverLevel?.stationCount,
            feeds.rainfall?.stationCount,
            feeds.temperature?.stationCount,
            feeds.humidity?.stationCount,
            feeds.windSpeed?.stationCount,
            feeds.windDirection?.stationCount,
            feeds.pressure?.stationCount
        ].reduce((sum, value) => sum + Number(value || 0), 0);

        const connectedFeeds = Object.values(feeds)
            .filter(feed => feed && typeof feed === 'object')
            .filter(feed => Boolean(feed.connected))
            .length;

        return res.json({
            ok: data.ok,
            source: data.source,
            updatedAt: data.updatedAt,
            feedCount: 7,
            connectedFeeds,
            feeds: {
                riverLevel: Boolean(feeds.riverLevel?.connected),
                rainfall: Boolean(feeds.rainfall?.connected),
                temperature: Boolean(feeds.temperature?.connected),
                humidity: Boolean(feeds.humidity?.connected),
                windSpeed: Boolean(feeds.windSpeed?.connected),
                windDirection: Boolean(feeds.windDirection?.connected),
                pressure: Boolean(feeds.pressure?.connected)
            },
            stationCount
        });
    } catch (error) {
        return res.status(502).json({
            ok: false,
            error: error.message
        });
    }
});

// =====================================================
// LIVE OVERPASS GIS PROXY
// Browser -> RakshaNet -> Overpass
// =====================================================

app.post('/api/gis/nearby', async (req, res) => {
    try {
        const query = req.body?.query;

        if (!query || typeof query !== 'string') {
            return res.status(400).json({
                error: 'Valid Overpass query is required.'
            });
        }

        const overpassServers = [
            'https://overpass-api.de/api/interpreter',
            'https://overpass.kumi.systems/api/interpreter',
            'https://overpass.private.coffee/api/interpreter'
        ];

        async function requestOverpass(endpoint) {
            console.log('Trying Overpass:', endpoint);

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': 'RakshaNet/1.0',
                    'Accept': 'application/json'
                },
                body: new URLSearchParams({ data: query }),
                signal: AbortSignal.timeout(10000)
            });

            const text = await response.text();

            if (!response.ok) {
                throw new Error(
                    `HTTP ${response.status}: ${text.slice(0, 300)}`
                );
            }

            if (!text.trim()) {
                throw new Error('Empty response from Overpass provider.');
            }

            console.log('Overpass success:', endpoint);
            return text;
        }

        const requests = overpassServers.map(endpoint => requestOverpass(endpoint));

        try {
            const text = await Promise.any(requests);
            return res.status(200).type('application/json').send(text);
        } catch (aggregateError) {
            console.error('All Overpass providers failed.');

            if (aggregateError && Array.isArray(aggregateError.errors)) {
                aggregateError.errors.forEach((error, index) => {
                    console.warn(
                        `Overpass provider ${index + 1} failed:`,
                        error?.message
                    );
                });
            }

            return res.status(502).json({
                error: 'All GIS services are currently unavailable.',
                details: 'All Overpass providers timed out or returned an error.'
            });
        }
    } catch (error) {
        console.error('GIS proxy error:', error);
        return res.status(500).json({
            error: 'GIS proxy failed.',
            details: error.message
        });
    }
});


// =====================================================
// RAKSHANET UNIFIED INTELLIGENCE API
//
// Region-agnostic intelligence layer:
// - Open-Meteo provides coordinate-based current + hourly weather
// - Assam NWDP telemetry is included when region=assam (or
//   coordinates fall within the Assam/North-East focus area)
// - Existing hazard/GIS/ML systems remain separate and reusable
// - Returns normalized observations + derived AI-ready features
// =====================================================

const INTELLIGENCE_DEFAULT_LOCATION = {
    region: 'India',
    latitude: 22.9734,
    longitude: 78.6569
};

const INTELLIGENCE_REGION_PRESETS = {
    india: { name: 'India', latitude: 22.9734, longitude: 78.6569 },
    assam: { name: 'Assam', latitude: 26.2006, longitude: 92.9376 },
    sikkim: { name: 'Sikkim', latitude: 27.5330, longitude: 88.5122 },
    arunachal: { name: 'Arunachal Pradesh', latitude: 28.2180, longitude: 94.7278 },
    meghalaya: { name: 'Meghalaya', latitude: 25.4670, longitude: 91.3662 },
    uttarakhand: { name: 'Uttarakhand', latitude: 30.0668, longitude: 79.0193 },
    himachal: { name: 'Himachal Pradesh', latitude: 31.1048, longitude: 77.1734 },
    maharashtra: { name: 'Maharashtra', latitude: 19.7515, longitude: 75.7139 },
    kerala: { name: 'Kerala', latitude: 10.8505, longitude: 76.2711 },
    odisha: { name: 'Odisha', latitude: 20.2961, longitude: 85.8245 }
};

function normalizeRegionKey(value) {
    return String(value || '')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]/g, '');
}

function resolveIntelligenceLocation(req) {
    const requestedRegion = normalizeRegionKey(req.query?.region);

    let latitude = Number(req.query?.lat);
    let longitude = Number(req.query?.lon);
    let region = req.query?.region ? String(req.query.region).trim() : '';

    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
        const preset =
            INTELLIGENCE_REGION_PRESETS[requestedRegion] ||
            INTELLIGENCE_DEFAULT_LOCATION;

        latitude = preset.latitude;
        longitude = preset.longitude;
        region = preset.name;
    }

    if (
        latitude < -90 || latitude > 90 ||
        longitude < -180 || longitude > 180
    ) {
        throw new Error('Latitude or longitude is outside a valid range.');
    }

    if (!region) region = 'Custom location';

    return { region, latitude, longitude };
}

function parseIsoOrDmyObservation(value) {
    if (!value) return null;

    const raw = String(value).trim();

    const dmy = raw.match(
        /^(\d{1,2})-(\d{1,2})-(\d{4})(?:\s+(\d{1,2}):?(\d{2})(?::?(\d{2}))?)?$/
    );

    if (dmy) {
        const date = new Date(
            Number(dmy[3]),
            Number(dmy[2]) - 1,
            Number(dmy[1]),
            Number(dmy[4] || 0),
            Number(dmy[5] || 0),
            Number(dmy[6] || 0)
        );

        if (!Number.isNaN(date.getTime())) return date;
    }

    const parsed = new Date(raw);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function observationFreshness(timestamp) {
    const parsed = parseIsoOrDmyObservation(timestamp);

    if (!parsed) {
        return {
            status: 'TIME_UNKNOWN',
            ageHours: null,
            label: 'Observation time unavailable'
        };
    }

    const ageHours =
        Math.max(0, Date.now() - parsed.getTime()) /
        (60 * 60 * 1000);

    if (ageHours <= 3) {
        return {
            status: 'FRESH',
            ageHours,
            label: 'Recent hourly observation'
        };
    }

    if (ageHours <= 24) {
        return {
            status: 'RECENT',
            ageHours,
            label: 'Within the last 24 hours'
        };
    }

    if (ageHours <= 72) {
        return {
            status: 'STALE',
            ageHours,
            label: 'Older than 24 hours'
        };
    }

    return {
        status: 'STALE',
        ageHours,
        label: 'Older than 72 hours'
    };
}

const OPEN_METEO_WEATHER_CACHE = new Map();
const OPEN_METEO_WEATHER_INFLIGHT = new Map();
let OPEN_METEO_WEATHER_QUEUE = Promise.resolve();
const OPEN_METEO_WEATHER_CACHE_TTL_MS = 10 * 60 * 1000;
const OPEN_METEO_WEATHER_STALE_TTL_MS = 2 * 60 * 60 * 1000;
const OPEN_METEO_WEATHER_MIN_INTERVAL_MS = 650;

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function openMeteoWeatherCacheKey(latitude, longitude) {
    return `${Number(latitude).toFixed(4)},${Number(longitude).toFixed(4)}`;
}

async function queueOpenMeteoWeatherRequest(task) {
    let release;
    const previous = OPEN_METEO_WEATHER_QUEUE;
    OPEN_METEO_WEATHER_QUEUE = new Promise(resolve => { release = resolve; });
    await previous;
    try {
        return await task();
    } finally {
        setTimeout(release, OPEN_METEO_WEATHER_MIN_INTERVAL_MS);
    }
}

function unavailableOpenMeteoWeather(error, cached = null) {
    return {
        available: false,
        source: 'Open-Meteo',
        error: String(error?.message || error || 'Weather service unavailable'),
        current: {},
        hourly: {},
        daily: {},
        units: {},
        cached: Boolean(cached),
        stale: Boolean(cached),
        fetchedAt: cached?.fetchedAt || null
    };
}

async function fetchOpenMeteoIntelligence(latitude, longitude) {
    const key = openMeteoWeatherCacheKey(latitude, longitude);
    const now = Date.now();
    const cached = OPEN_METEO_WEATHER_CACHE.get(key);

    if (cached && now - cached.timestamp < OPEN_METEO_WEATHER_CACHE_TTL_MS) {
        return { ...cached.data, available: true, cached: true, stale: false, fetchedAt: new Date(cached.timestamp).toISOString() };
    }

    if (OPEN_METEO_WEATHER_INFLIGHT.has(key)) {
        return OPEN_METEO_WEATHER_INFLIGHT.get(key);
    }

    const request = (async () => {
        const params = new URLSearchParams({
            latitude: String(latitude),
            longitude: String(longitude),
            timezone: 'UTC',
            past_days: '7',
            forecast_days: '7',
            current: [
                'temperature_2m',
                'relative_humidity_2m',
                'apparent_temperature',
                'precipitation',
                'rain',
                'cloud_cover',
                'visibility',
                'weather_code',
                'wind_speed_10m',
                'wind_gusts_10m',
                'wind_direction_10m',
                'surface_pressure',
                'soil_moisture_0_to_1cm'
            ].join(','),
            hourly: [
                'temperature_2m',
                'precipitation',
                'rain',
                'weather_code',
                'wind_speed_10m',
                'wind_gusts_10m',
                'precipitation_probability'
            ].join(','),
            daily: [
                'precipitation_sum',
                'rain_sum',
                'precipitation_hours',
                'precipitation_probability_max',
                'weather_code',
                'temperature_2m_max',
                'temperature_2m_min',
                'wind_speed_10m_max',
                'wind_gusts_10m_max'
            ].join(',')
        });

        try {
            const response = await queueOpenMeteoWeatherRequest(() => fetch(
                `https://api.open-meteo.com/v1/forecast?${params.toString()}`,
                {
                    headers: {
                        'User-Agent': 'RakshaNet/1.1',
                        'Accept': 'application/json'
                    },
                    signal: AbortSignal.timeout(15000)
                }
            ));

            if (!response.ok) {
                const error = new Error(`Open-Meteo HTTP ${response.status}`);
                error.status = response.status;
                error.retryAfter = response.headers.get('retry-after');
                throw error;
            }

            const data = await response.json();
            const enriched = { ...data, available: true, cached: false, stale: false, fetchedAt: new Date().toISOString() };
            OPEN_METEO_WEATHER_CACHE.set(key, { timestamp: Date.now(), data: enriched });
            return enriched;
        } catch (error) {
            if (cached && now - cached.timestamp < OPEN_METEO_WEATHER_STALE_TTL_MS) {
                console.warn(
                    `Open-Meteo unavailable for ${key}; using cached weather from ${new Date(cached.timestamp).toISOString()}. ${error.message}`
                );
                return {
                    ...cached.data,
                    available: true,
                    cached: true,
                    stale: true,
                    staleReason: error.message,
                    fetchedAt: new Date(cached.timestamp).toISOString()
                };
            }

            console.warn(`Open-Meteo unavailable for ${key}: ${error.message}`);
            return unavailableOpenMeteoWeather(error);
        } finally {
            OPEN_METEO_WEATHER_INFLIGHT.delete(key);
        }
    })();

    OPEN_METEO_WEATHER_INFLIGHT.set(key, request);
    return request;
}

function utcTimeToDate(value) {
    if (!value) return null;

    const raw = String(value);
    const date = new Date(
        /Z$|[+-]\d\d:\d\d$/.test(raw)
            ? raw
            : `${raw}Z`
    );

    return Number.isNaN(date.getTime()) ? null : date;
}

function aggregateWeatherFeatures(hourly) {
    const times = Array.isArray(hourly?.time)
        ? hourly.time
        : [];

    const precipitation = Array.isArray(hourly?.precipitation)
        ? hourly.precipitation
        : [];

    const rain = Array.isArray(hourly?.rain)
        ? hourly.rain
        : [];

    const temperature = Array.isArray(hourly?.temperature_2m)
        ? hourly.temperature_2m
        : [];

    const wind = Array.isArray(hourly?.wind_speed_10m)
        ? hourly.wind_speed_10m
        : [];

    const now = Date.now();

    function valuesWithin(hours, source) {
        const cutoff = now - hours * 60 * 60 * 1000;
        const values = [];

        for (let i = 0; i < times.length; i++) {
            const date = utcTimeToDate(times[i]);
            if (!date || date.getTime() < cutoff || date.getTime() > now) continue;

            const value = Number(source[i]);
            if (Number.isFinite(value)) values.push(value);
        }

        return values;
    }

    function sum(values) {
        return values.reduce((total, value) => total + value, 0);
    }

    const rain1 = valuesWithin(24, rain);
    const rain3 = valuesWithin(72, rain);
    const rain7 = valuesWithin(168, rain);

    const precip1 = valuesWithin(24, precipitation);
    const precip3 = valuesWithin(72, precipitation);
    const precip7 = valuesWithin(168, precipitation);

    const temp1 = valuesWithin(24, temperature);
    const wind1 = valuesWithin(24, wind);

    function mean(values) {
        return values.length
            ? sum(values) / values.length
            : null;
    }

    function max(values) {
        return values.length ? Math.max(...values) : null;
    }

    function min(values) {
        return values.length ? Math.min(...values) : null;
    }

    return {
        rain_1d: Number(sum(rain1).toFixed(3)),
        rain_3d: Number(sum(rain3).toFixed(3)),
        rain_7d: Number(sum(rain7).toFixed(3)),
        precip_1d: Number(sum(precip1).toFixed(3)),
        precip_3d: Number(sum(precip3).toFixed(3)),
        precip_7d: Number(sum(precip7).toFixed(3)),
        precip_hours_1d: precip1.filter(value => value > 0).length,
        precip_hours_3d: precip3.filter(value => value > 0).length,
        precip_hours_7d: precip7.filter(value => value > 0).length,
        temp_mean: mean(temp1) === null ? null : Number(mean(temp1).toFixed(3)),
        temp_max: max(temp1),
        temp_min: min(temp1),
        wind_max: max(wind1),
        sourceHours: times.length
    };
}

async function runLandslideModelForFeatures(features) {
    const orderedValues = LANDSLIDE_FEATURES.map(
        feature => Number(features[feature])
    );

    if (orderedValues.some(value => !Number.isFinite(value))) {
        return {
            available: false,
            reason: 'Not all model features are available.'
        };
    }

    try {
        const session = await getLandslideSession();
        const inputTensor = new ort.Tensor(
            'float32',
            Float32Array.from(orderedValues),
            [1, orderedValues.length]
        );

        const inputName = session.inputNames[0];
        const results = await session.run({
            [inputName]: inputTensor
        });

        const parsedPrediction = parseOnnxPrediction(results, session.outputNames);
        const label = parsedPrediction.label;
        let probability = parsedPrediction.probability;

        if (probability === null) {
            probability = label === 1 ? 1 : 0;
        }

        probability = Math.max(0, Math.min(1, probability));
        const percentage = Math.round(probability * 100);

        let risk = 'LOW';
        let priority = 'P4 Normal';

        if (percentage >= 80) {
            risk = 'EXTREME';
            priority = 'P1 Critical';
        } else if (percentage >= 60) {
            risk = 'HIGH';
            priority = 'P2 High';
        } else if (percentage >= 35) {
            risk = 'WATCH';
            priority = 'P3 Watch';
        }

        return {
            available: true,
            model: ML_MODEL_VERSION,
            percentage,
            risk,
            priority,
            predictedClass: percentage >= 50 ? 1 : 0
        };
    } catch (error) {
        console.error('Unified intelligence ML error:', error);
        return {
            available: false,
            reason: error.message
        };
    }
}

function isAssamFocus(region, latitude, longitude) {
    const key = normalizeRegionKey(region);

    if (key === 'assam') return true;

    // Broad Assam bounding area; intentionally used only to decide
    // whether to display Assam-specific NWDP telemetry.
    return (
        latitude >= 24.0 &&
        latitude <= 28.5 &&
        longitude >= 89.5 &&
        longitude <= 96.5
    );
}


const FLOOD_API_CACHE = new Map();
const REVERSE_GEOCODE_CACHE = new Map();
const IMD_WARNING_CACHE = { timestamp: 0, data: null };
const IMD_CYCLONE_CACHE = { timestamp: 0, data: null };
const SACHET_CACHE = { timestamp: 0, data: null, etag: null };
const FLOOD_CACHE_TTL_MS = 5 * 60 * 1000;
const IMD_CACHE_TTL_MS = 10 * 60 * 1000;
const SACHET_CACHE_TTL_MS = 2 * 60 * 1000;
const REVERSE_CACHE_TTL_MS = 6 * 60 * 60 * 1000;
const SACHET_RSS_URL = process.env.SACHET_RSS_URL || 'https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml';

function clampScore(value) {
    return Math.max(0, Math.min(100, Number.isFinite(Number(value)) ? Number(value) : 0));
}

function median(values) {
    const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!clean.length) return null;
    const mid = Math.floor(clean.length / 2);
    return clean.length % 2 ? clean[mid] : (clean[mid - 1] + clean[mid]) / 2;
}

async function fetchOpenMeteoFlood(latitude, longitude) {
    const key = `${Number(latitude).toFixed(3)},${Number(longitude).toFixed(3)}`;
    const cached = FLOOD_API_CACHE.get(key);
    if (cached && Date.now() - cached.timestamp < FLOOD_CACHE_TTL_MS) return cached.data;

    const params = new URLSearchParams({
        latitude: String(latitude),
        longitude: String(longitude),
        timezone: 'UTC',
        past_days: '7',
        forecast_days: '7',
        daily: [
            'river_discharge',
            'river_discharge_mean',
            'river_discharge_median',
            'river_discharge_max',
            'river_discharge_min',
            'river_discharge_p25',
            'river_discharge_p75'
        ].join(',')
    });
    const response = await fetch(`https://flood-api.open-meteo.com/v1/flood?${params.toString()}`, {
        headers: { 'User-Agent': 'RakshaNet/3.0', 'Accept': 'application/json' },
        signal: AbortSignal.timeout(15000)
    });
    if (!response.ok) throw new Error(`GloFAS/Open-Meteo Flood HTTP ${response.status}`);
    const data = await response.json();
    FLOOD_API_CACHE.set(key, { timestamp: Date.now(), data });
    return data;
}

function buildHydrologySummary(floodData) {
    const daily = floodData?.daily || {};
    const times = Array.isArray(daily.time) ? daily.time : [];
    const discharge = Array.isArray(daily.river_discharge) ? daily.river_discharge.map(Number) : [];
    const p25 = Array.isArray(daily.river_discharge_p25) ? daily.river_discharge_p25.map(Number) : [];
    const p75 = Array.isArray(daily.river_discharge_p75) ? daily.river_discharge_p75.map(Number) : [];
    const mean = Array.isArray(daily.river_discharge_mean) ? daily.river_discharge_mean.map(Number) : [];
    const maxEnsemble = Array.isArray(daily.river_discharge_max) ? daily.river_discharge_max.map(Number) : [];

    if (!times.length || !discharge.length) {
        return { available: false, source: 'GloFAS v4 via Open-Meteo Flood API' };
    }

    const todayKey = new Date().toISOString().slice(0, 10);
    let nowIndex = times.findIndex(value => String(value).slice(0, 10) === todayKey);
    if (nowIndex < 0) nowIndex = Math.max(0, Math.min(discharge.length - 1, 7));
    const past = discharge.slice(0, Math.min(nowIndex + 1, 8)).filter(Number.isFinite);
    const recentBaseline = median(past.slice(-7, -1)) ?? median(past) ?? null;
    const today = Number(discharge[nowIndex]);
    const forecastWindow = discharge.slice(nowIndex, nowIndex + 3).filter(Number.isFinite);
    const forecastPeak = forecastWindow.length ? Math.max(...forecastWindow) : today;
    const currentP25 = Number(p25[nowIndex]);
    const currentP75 = Number(p75[nowIndex]);
    const currentMean = Number(mean[nowIndex]);
    const ensembleMax = Number(maxEnsemble[nowIndex]);

    const ratio = Number.isFinite(recentBaseline) && recentBaseline > 0 ? today / recentBaseline : null;
    const peakRatio = Number.isFinite(recentBaseline) && recentBaseline > 0 ? forecastPeak / recentBaseline : null;

    let ensembleSignal = 0;
    if (Number.isFinite(today) && Number.isFinite(currentP75) && today >= currentP75) ensembleSignal += 35;
    if (Number.isFinite(today) && Number.isFinite(ensembleMax) && ensembleMax > 0 && today >= ensembleMax * 0.9) ensembleSignal += 25;
    if (Number.isFinite(today) && Number.isFinite(currentP25) && today >= currentP25) ensembleSignal += 10;

    const ratioSignal = Number.isFinite(peakRatio)
        ? clampScore((peakRatio - 1) * 55)
        : 0;

    const riverStress = Math.round(clampScore(Math.max(ensembleSignal + ratioSignal, 0)));

    return {
        available: true,
        source: 'GloFAS v4 via Open-Meteo Flood API',
        riverDischargeM3s: Number.isFinite(today) ? Number(today.toFixed(2)) : null,
        forecastPeak3dM3s: Number.isFinite(forecastPeak) ? Number(forecastPeak.toFixed(2)) : null,
        recentBaselineM3s: Number.isFinite(recentBaseline) ? Number(recentBaseline.toFixed(2)) : null,
        currentP25M3s: Number.isFinite(currentP25) ? Number(currentP25.toFixed(2)) : null,
        currentP75M3s: Number.isFinite(currentP75) ? Number(currentP75.toFixed(2)) : null,
        currentMeanM3s: Number.isFinite(currentMean) ? Number(currentMean.toFixed(2)) : null,
        ensembleMaxM3s: Number.isFinite(ensembleMax) ? Number(ensembleMax.toFixed(2)) : null,
        peakToBaselineRatio: Number.isFinite(peakRatio) ? Number(peakRatio.toFixed(2)) : null,
        forecastDailyDischarge: discharge.slice(nowIndex, nowIndex + 7).map(value => Number.isFinite(value) ? Number(value.toFixed(2)) : null),
        riverStressIndex: riverStress,
        observationDate: times[nowIndex] || null,
        note: 'Hydrological stress index derived from GloFAS river-discharge forecasts and recent local discharge baseline. It is not a flood probability or official flood warning.'
    };
}

function summarizeWeatherHazards(weather, hydrology) {
    const current = weather?.current || {};
    const hourly = weather?.hourly || {};
    const daily = weather?.daily || {};
    const currentTime = utcTimeToDate(current.time) || new Date();
    const nowMs = currentTime.getTime();
    const times = Array.isArray(hourly.time) ? hourly.time : [];

    const series = {
        precipitation: Array.isArray(hourly.precipitation) ? hourly.precipitation : [],
        rain: Array.isArray(hourly.rain) ? hourly.rain : [],
        weatherCode: Array.isArray(hourly.weather_code) ? hourly.weather_code : [],
        gusts: Array.isArray(hourly.wind_gusts_10m) ? hourly.wind_gusts_10m : []
    };

    function valuesInWindow(source, fromMs, toMs) {
        const out = [];
        for (let i = 0; i < times.length; i++) {
            const date = utcTimeToDate(times[i]);
            if (!date) continue;
            const t = date.getTime();
            if (t < fromMs || t > toMs) continue;
            const value = Number(source[i]);
            if (Number.isFinite(value)) out.push(value);
        }
        return out;
    }

    function sum(values) { return values.reduce((a, b) => a + b, 0); }
    function max(values, fallback = 0) { return values.length ? Math.max(...values) : fallback; }

    const observed24 = valuesInWindow(series.rain, nowMs - 24 * 3600e3, nowMs);
    const observed72 = valuesInWindow(series.rain, nowMs - 72 * 3600e3, nowMs);
    const observed168 = valuesInWindow(series.rain, nowMs - 168 * 3600e3, nowMs);
    const observed24Precip = valuesInWindow(series.precipitation, nowMs - 24 * 3600e3, nowMs);
    const forecast24 = valuesInWindow(series.precipitation, nowMs, nowMs + 24 * 3600e3);
    const forecast72 = valuesInWindow(series.precipitation, nowMs, nowMs + 72 * 3600e3);
    const forecastCodes = valuesInWindow(series.weatherCode, nowMs, nowMs + 72 * 3600e3);
    const observedCodes = valuesInWindow(series.weatherCode, nowMs - 24 * 3600e3, nowMs);
    const forecastGusts = valuesInWindow(series.gusts, nowMs, nowMs + 72 * 3600e3);

    const dailyProb = (daily.precipitation_probability_max || []).map(Number).filter(Number.isFinite).slice(0, 3);
    const maxProb = dailyProb.length ? Math.max(...dailyProb) : 0;
    const maxTemp = Math.max(...((daily.temperature_2m_max || []).map(Number).filter(Number.isFinite).slice(0, 3)), Number(current.temperature_2m || 0));
    const maxWindGust = Math.max(max(forecastGusts), Number(current.wind_gusts_10m || 0));

    const hasThunder = [...observedCodes, ...forecastCodes].some(code => [95, 96, 99].includes(Number(code)));
    const thunderSignal = hasThunder ? 85 : 0;

    // Flood stress combines upstream hydrology, observed accumulated rain and near-term forecast.
    // It remains a decision-support index, not a probability.
    const riverStress = Number(hydrology?.riverStressIndex || 0);
    const riverP75Ratio = Number.isFinite(Number(hydrology?.currentP75M3s)) && Number(hydrology?.currentP75M3s) > 0
        ? Number(hydrology?.riverDischargeM3s || 0) / Number(hydrology.currentP75M3s)
        : null;
    const forecastP75Ratio = Number.isFinite(Number(hydrology?.currentP75M3s)) && Number(hydrology?.currentP75M3s) > 0
        ? Number(hydrology?.forecastPeak3dM3s || 0) / Number(hydrology.currentP75M3s)
        : null;
    const riverLevelSignal = clampScore(Math.max(
        riverStress,
        Number.isFinite(riverP75Ratio) ? riverP75Ratio * 80 : 0,
        Number.isFinite(forecastP75Ratio) ? forecastP75Ratio * 90 : 0
    ));
    const observedRainSignal = clampScore(
        (sum(observed72) / 180) * 45 +
        (sum(observed168) / 350) * 20
    );
    const incomingRainSignal = clampScore(
        (sum(forecast24) / 80) * 20 +
        (sum(forecast72) / 180) * 25 +
        (maxProb / 100) * 15
    );
    const floodScore = hydrology?.available
        ? Math.round(clampScore(riverLevelSignal * 0.50 + observedRainSignal * 0.30 + incomingRainSignal * 0.20))
        : Math.round(clampScore(observedRainSignal * 0.50 + incomingRainSignal * 0.50));

    const observed24Rain = sum(observed24);
    const next24Rain = sum(forecast24);
    const next72Rain = sum(forecast72);
    const heavyRain = Math.round(clampScore(
        (max(observed24) / 20) * 30 +
        (observed24Rain / 80) * 20 +
        (next24Rain / 80) * 30 +
        (next72Rain / 180) * 10 +
        (maxProb / 100) * 10
    ));

    const thunderstorm = Math.round(clampScore(
        thunderSignal * 0.65 +
        (maxWindGust / 90) * 20 +
        (maxProb / 100) * 15
    ));

    const heatwave = Math.round(clampScore(((maxTemp - 32) / 10) * 100));

    const severe = Math.round(clampScore(
        thunderSignal * 0.40 +
        (maxWindGust / 100) * 35 +
        (maxProb / 100) * 25
    ));

    return {
        flood: floodScore,
        heavyRain,
        thunderstorm,
        heatwave,
        severeWeather: severe,
        raw: {
            observed24hRain: Number(observed24Rain.toFixed(2)),
            observed72hRain: Number(sum(observed72).toFixed(2)),
            observed7dRain: Number(sum(observed168).toFixed(2)),
            forecast24hPrecip: Number(next24Rain.toFixed(2)),
            forecast72hPrecip: Number(next72Rain.toFixed(2)),
            maxObservedHourlyRain: Number(max(observed24).toFixed(2)),
            maxPrecipProbability: Math.round(maxProb),
            maxTemperature: Number(maxTemp.toFixed(1)),
            maxWindGust: Number(maxWindGust.toFixed(1)),
            weatherCode: Number(current.weather_code),
            floodMethod: hydrology?.available ? 'GloFAS + observed/forecast rainfall' : 'Observed/forecast rainfall only',
            dataBoundary: current.time || null
        }
    };
}

async function reverseGeocodeLocation(latitude, longitude) {
    const key = `${Number(latitude).toFixed(3)},${Number(longitude).toFixed(3)}`;
    const cached = REVERSE_GEOCODE_CACHE.get(key);
    if (cached && Date.now() - cached.timestamp < REVERSE_CACHE_TTL_MS) return cached.data;

    try {
        const url = new URL('https://nominatim.openstreetmap.org/reverse');
        url.search = new URLSearchParams({
            lat: String(latitude), lon: String(longitude), format: 'jsonv2', zoom: '10', addressdetails: '1'
        });
        const response = await fetch(url, {
            headers: { 'User-Agent': 'RakshaNet/3.0 contact=local-project', 'Accept': 'application/json' },
            signal: AbortSignal.timeout(9000)
        });
        if (!response.ok) throw new Error(`Nominatim reverse HTTP ${response.status}`);
        const data = await response.json();
        const address = data.address || {};
        const result = {
            displayName: data.display_name || null,
            district: address.state_district || address.district || address.county || null,
            state: address.state || null,
            city: address.city || address.town || address.village || address.municipality || null,
            country: address.country || null
        };
        REVERSE_GEOCODE_CACHE.set(key, { timestamp: Date.now(), data: result });
        return result;
    } catch (error) {
        const result = { displayName: null, district: null, state: null, city: null, country: null, error: error.message };
        REVERSE_GEOCODE_CACHE.set(key, { timestamp: Date.now(), data: result });
        return result;
    }
}

function imdHeaders() {
    const headers = { 'User-Agent': 'RakshaNet/3.0', 'Accept': 'application/json' };
    const key = String(process.env.IMD_API_KEY || '').trim();
    if (key) {
        headers['X-API-Key'] = key;
        headers.Authorization = `Bearer ${key}`;
    }
    return headers;
}

async function fetchImdDistrictWarnings() {
    if (IMD_WARNING_CACHE.data && Date.now() - IMD_WARNING_CACHE.timestamp < IMD_CACHE_TTL_MS) return IMD_WARNING_CACHE.data;
    try {
        const response = await fetch('https://api.imd.gov.in/api/v1/districtwarning', {
            headers: imdHeaders(), signal: AbortSignal.timeout(12000)
        });
        if (!response.ok) throw new Error(`IMD district warning HTTP ${response.status}`);
        const data = await response.json();
        const rows = Array.isArray(data) ? data : Array.isArray(data?.data) ? data.data : [];
        const result = { available: true, rows, source: 'India Meteorological Department (IMD) district warnings', checkedAt: new Date().toISOString() };
        IMD_WARNING_CACHE.timestamp = Date.now();
        IMD_WARNING_CACHE.data = result;
        return result;
    } catch (error) {
        const result = { available: false, rows: [], source: 'India Meteorological Department (IMD) district warnings', error: error.message, checkedAt: new Date().toISOString() };
        IMD_WARNING_CACHE.timestamp = Date.now();
        IMD_WARNING_CACHE.data = result;
        return result;
    }
}

function normalizeLoose(value) {
    return String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

function matchImdDistrictWarning(rows, geo) {
    const candidates = [geo?.district, geo?.city, geo?.state].map(normalizeLoose).filter(Boolean);
    if (!candidates.length) return null;
    let best = null;
    let bestRank = Infinity;
    for (const row of rows || []) {
        const district = normalizeLoose(row?.District || row?.district);
        const state = normalizeLoose(row?.State || row?.state);
        if (!district && !state) continue;
        let rank = Infinity;
        if (district && candidates.includes(district)) rank = 0;
        else if (district && candidates.some(c => c.includes(district) || district.includes(c))) rank = 1;
        else if (state && candidates.includes(state)) rank = 2;
        if (rank < bestRank) { best = row; bestRank = rank; }
    }
    return best;
}

const IMD_WARNING_MAP = {
    heavy_rain: new Set(['2', '16', '17']),
    thunderstorm: new Set(['4', '5', '6', '14', '15', '17']),
    heatwave: new Set(['9', '10']),
    severe: new Set(['4', '5', '6', '8', '9', '10', '14', '15', '16', '17'])
};

function analyzeImdWarning(row) {
    if (!row) return { available: true, matched: false, official: false, severity: 'NONE', codes: [], hazards: [], message: null };
    const codes = [];
    for (const day of ['Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'day1_warning', 'day2_warning', 'day3_warning', 'day4_warning', 'day5_warning']) {
        const raw = row?.[day];
        if (raw === undefined || raw === null || raw === '') continue;
        String(raw).split(',').map(x => x.trim()).filter(Boolean).forEach(code => codes.push(code));
    }
    const warningsText = [row?.message, row?.Day_1, row?.Day_2, row?.Day_3, row?.Day_4, row?.Day_5, row?.event, row?.Event].filter(Boolean).join(' | ');
    const color = [row?.Day1_Color, row?.day1_color, row?.Day2_Color, row?.day2_color].map(Number).filter(Number.isFinite);
    const redOrOrange = color.some(v => v <= 2);
    const hazards = new Set();
    for (const [hazard, codeSet] of Object.entries(IMD_WARNING_MAP)) {
        if (codes.some(code => codeSet.has(String(code).trim()))) hazards.add(hazard === 'heatwave' ? 'extreme_heat' : hazard);
    }
    const text = String(warningsText).toLowerCase();
    if (/very heavy rain|extremely heavy rain|heavy rainfall|heavy rain|rainfall warning/.test(text)) hazards.add('heavy_rain');
    if (/thunderstorm|lightning|squall|hail/.test(text)) hazards.add('thunderstorm');
    if (/heat wave|heatwave|hot day|severe heat/.test(text)) hazards.add('extreme_heat');
    if (/cyclone|depression|deep depression|tropical storm/.test(text)) hazards.add('cyclone');
    if (/flood|inundation|waterlogging|river level|flash flood/.test(text)) hazards.add('flood');
    if (/landslide|slope failure|mudslide|rockfall/.test(text)) hazards.add('landslide');
    return {
        available: true,
        matched: true,
        official: true,
        district: row?.District || row?.district || null,
        state: row?.State || row?.state || null,
        date: row?.Date || row?.date_obs || null,
        severity: redOrOrange ? 'HIGH' : 'WATCH',
        codes,
        hazards: Array.from(hazards),
        message: warningsText || null,
        raw: row
    };
}


function decodeXmlEntities(value) {
    return String(value || '')
        .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/gi, '$1')
        .replace(/&amp;/gi, '&')
        .replace(/&lt;/gi, '<')
        .replace(/&gt;/gi, '>')
        .replace(/&quot;/gi, '"')
        .replace(/&apos;/gi, "'")
        .replace(/&#(\d+);/g, (_m, n) => String.fromCharCode(Number(n)))
        .replace(/&#x([0-9a-f]+);/gi, (_m, n) => String.fromCharCode(parseInt(n, 16)));
}

function stripHtml(value) {
    return decodeXmlEntities(value)
        .replace(/<script[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style[\s\S]*?<\/style>/gi, ' ')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function extractXmlTag(block, tagNames) {
    for (const tagName of tagNames) {
        const re = new RegExp(`<(?:(?:[A-Za-z0-9_-]+):)?${tagName}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/(?:(?:[A-Za-z0-9_-]+):)?${tagName}>`, 'i');
        const match = String(block || '').match(re);
        if (match) return decodeXmlEntities(match[1]).trim();
    }
    return '';
}

function classifySachetHazard(text) {
    const value = String(text || '').toLowerCase();
    if (/landslide|slope failure|mudslide|rockfall/.test(value)) return 'landslide';
    if (/flood|inundation|waterlogging|river level|flash flood/.test(value)) return 'flood';
    if (/cyclone|depression|deep depression|tropical storm/.test(value)) return 'cyclone';
    if (/heat wave|heatwave|hot day|severe heat/.test(value)) return 'extreme_heat';
    if (/thunderstorm|lightning|squall|hail/.test(value)) return 'thunderstorm';
    if (/very heavy rain|extremely heavy rain|heavy rainfall|heavy rain|intense rainfall|rainfall warning/.test(value)) return 'heavy_rain';
    return /severe weather|weather warning|warning/.test(value) ? 'severe' : 'other';
}

function sachetSeverityScore(alert) {
    const severity = String(alert?.severity || alert?.color || '').toLowerCase();
    const urgency = String(alert?.urgency || '').toLowerCase();
    const certainty = String(alert?.certainty || '').toLowerCase();
    let score = 40;
    if (/extreme|red/.test(severity)) score = 90;
    else if (/severe|orange/.test(severity)) score = 75;
    else if (/moderate|yellow/.test(severity)) score = 50;
    else if (/minor|green/.test(severity)) score = 25;
    if (/immediate/.test(urgency)) score += 8;
    else if (/expected/.test(urgency)) score += 4;
    if (/observed|likely/.test(certainty)) score += 3;
    return Math.round(clampScore(score));
}

function parseSachetRss(xml) {
    const items = [];
    const blocks = String(xml || '').match(/<item\b[\s\S]*?<\/item>/gi) || [];
    for (const block of blocks) {
        const title = extractXmlTag(block, ['title', 'headline']);
        const description = stripHtml(extractXmlTag(block, ['description', 'summary', 'text']));
        const areaDesc = extractXmlTag(block, ['areaDesc', 'area', 'location', 'district']);
        const pubDate = extractXmlTag(block, ['pubDate', 'effective', 'sent']);
        const expires = extractXmlTag(block, ['expires']);
        const severity = extractXmlTag(block, ['severity', 'Severity']);
        const urgency = extractXmlTag(block, ['urgency']);
        const certainty = extractXmlTag(block, ['certainty']);
        const link = extractXmlTag(block, ['link']);
        const guid = extractXmlTag(block, ['guid', 'identifier']);
        const category = extractXmlTag(block, ['category', 'event']);
        const polygon = parseCapPolygon(extractXmlTag(block, ['polygon']));
        const circle = parseCapCircle(extractXmlTag(block, ['circle']));
        const text = [title, description, areaDesc, category].filter(Boolean).join(' | ');
        const hazard = classifySachetHazard(text);
        const publishedAt = pubDate ? new Date(pubDate) : null;
        const expiresAt = expires ? new Date(expires) : null;
        if (expiresAt && Number.isFinite(expiresAt.getTime()) && expiresAt.getTime() < Date.now()) continue;
        items.push({
            title: title || 'NDMA/SACHET alert',
            description,
            areaDesc,
            pubDate: pubDate || null,
            expires: expires || null,
            publishedAt: publishedAt && Number.isFinite(publishedAt.getTime()) ? publishedAt.toISOString() : null,
            expiresAt: expiresAt && Number.isFinite(expiresAt.getTime()) ? expiresAt.toISOString() : null,
            severity: severity || '',
            urgency: urgency || '',
            certainty: certainty || '',
            category,
            link,
            guid,
            polygon,
            circle,
            hazard,
            severityScore: sachetSeverityScore({ severity: severity || category || title })
        });
    }
    return items.sort((a, b) => Number(b.severityScore || 0) - Number(a.severityScore || 0) || String(b.publishedAt || '').localeCompare(String(a.publishedAt || '')));
}

async function fetchSachetAlerts() {
    if (SACHET_CACHE.data && Date.now() - SACHET_CACHE.timestamp < SACHET_CACHE_TTL_MS) return SACHET_CACHE.data;
    try {
        const headers = { 'User-Agent': 'RakshaNet/4.0', 'Accept': 'application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8' };
        if (SACHET_CACHE.etag) headers['If-None-Match'] = SACHET_CACHE.etag;
        const response = await fetch(SACHET_RSS_URL, { headers, signal: AbortSignal.timeout(12000) });
        if (response.status === 304 && SACHET_CACHE.data) {
            SACHET_CACHE.timestamp = Date.now();
            return SACHET_CACHE.data;
        }
        if (!response.ok) throw new Error(`SACHET RSS HTTP ${response.status}`);
        const xml = await response.text();
        const result = {
            available: true,
            source: 'NDMA SACHET India CAP/RSS feed',
            url: SACHET_RSS_URL,
            alerts: parseSachetRss(xml),
            checkedAt: new Date().toISOString(),
            etag: response.headers.get('etag') || null
        };
        SACHET_CACHE.timestamp = Date.now();
        SACHET_CACHE.data = result;
        SACHET_CACHE.etag = result.etag || SACHET_CACHE.etag;
        return result;
    } catch (error) {
        const result = {
            available: false,
            source: 'NDMA SACHET India CAP/RSS feed',
            url: SACHET_RSS_URL,
            alerts: SACHET_CACHE.data?.alerts || [],
            checkedAt: new Date().toISOString(),
            error: error.message,
            cached: Boolean(SACHET_CACHE.data?.alerts?.length)
        };
        SACHET_CACHE.timestamp = Date.now();
        SACHET_CACHE.data = result;
        return result;
    }
}

function normalizeLocationText(value) {
    return normalizeLoose(String(value || '').replace(/\([^)]*\)/g, ' '));
}

const STATE_ALIASES = {
    andhrapradesh: ['andhrapradesh'],
    arunachalpradesh: ['arunachalpradesh', 'arunachal'],
    assam: ['assam'],
    bihar: ['bihar'],
    chhattisgarh: ['chhattisgarh', 'chattisgarh'],
    goa: ['goa'],
    gujarat: ['gujarat'],
    haryana: ['haryana'],
    himachalpradesh: ['himachalpradesh', 'himachal'],
    jharkhand: ['jharkhand'],
    karnataka: ['karnataka'],
    kerala: ['kerala'],
    madhyapradesh: ['madhyapradesh'],
    maharashtra: ['maharashtra'],
    manipur: ['manipur'],
    meghalaya: ['meghalaya'],
    mizoram: ['mizoram'],
    nagaland: ['nagaland'],
    odisha: ['odisha', 'orissa'],
    punjab: ['punjab'],
    rajasthan: ['rajasthan'],
    sikkim: ['sikkim'],
    tamilnadu: ['tamilnadu'],
    telangana: ['telangana'],
    tripura: ['tripura'],
    uttarakhand: ['uttarakhand', 'uttrakhand'],
    uttarpradesh: ['uttarpradesh'],
    westbengal: ['westbengal']
};

function stateLocationAliases(state) {
    const key = normalizeLocationText(state);
    return STATE_ALIASES[key] || (key ? [key] : []);
}


function parseCapPolygon(value) {
    const raw = String(value || '').trim();
    if (!raw) return [];
    const points = raw.split(/\s+/).map(pair => {
        const [lat, lon] = pair.split(',').map(Number);
        return Number.isFinite(lat) && Number.isFinite(lon) ? [lat, lon] : null;
    }).filter(Boolean);
    return points.length >= 3 ? points : [];
}

function parseCapCircle(value) {
    const raw = String(value || '').trim();
    if (!raw) return null;
    const match = raw.match(/^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$/);
    if (!match) return null;
    const lat = Number(match[1]);
    const lon = Number(match[2]);
    const radiusKm = Number(match[3]);
    return Number.isFinite(lat) && Number.isFinite(lon) && Number.isFinite(radiusKm)
        ? { latitude: lat, longitude: lon, radiusKm }
        : null;
}

function pointInPolygon(latitude, longitude, polygon) {
    if (!Array.isArray(polygon) || polygon.length < 3) return false;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const yi = Number(polygon[i][0]);
        const xi = Number(polygon[i][1]);
        const yj = Number(polygon[j][0]);
        const xj = Number(polygon[j][1]);
        const intersect = ((yi > latitude) !== (yj > latitude)) &&
            (longitude < (xj - xi) * (latitude - yi) / ((yj - yi) || 1e-12) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

function capGeometryMatch(alert, latitude, longitude) {
    if (!Number.isFinite(Number(latitude)) || !Number.isFinite(Number(longitude))) return false;
    if (pointInPolygon(Number(latitude), Number(longitude), alert?.polygon)) return true;
    const circle = alert?.circle;
    if (circle) {
        return haversineKm(Number(latitude), Number(longitude), circle.latitude, circle.longitude) <= circle.radiusKm;
    }
    return false;
}

function matchSachetAlerts(alerts, geo, requestedHazard = 'auto', latitude = null, longitude = null) {
    const candidates = [geo?.district, geo?.city, geo?.state].filter(Boolean).map(normalizeLocationText);
    const stateAliases = stateLocationAliases(geo?.state);
    const requested = String(requestedHazard || 'auto').toLowerCase();
    const rows = [];
    for (const alert of alerts || []) {
        const hazard = alert.hazard;
        if (requested !== 'auto' && requested !== 'severe' && hazard !== requested) continue;
        const text = [alert.title, alert.description, alert.areaDesc, alert.category].filter(Boolean).join(' | ');
        const normalizedText = normalizeLocationText(text);
        const area = normalizeLocationText(alert.areaDesc);
        const geometryMatch = capGeometryMatch(alert, latitude, longitude);
        const districtMatch = candidates.find(token => token && (normalizedText.includes(token) || area.includes(token)));
        const stateMatch = stateAliases.find(token => token && (normalizedText.includes(token) || area.includes(token)));
        const allIndia = /(^|\b)all\s*india(\b|$)/.test(area) || /^india(\b|$)/.test(area) || /allindia/.test(normalizedText);
        if (!geometryMatch && !districtMatch && !stateMatch && !allIndia) continue;
        const locationMatch = geometryMatch ? 'CAP geometry' : districtMatch ? 'district/city' : stateMatch ? 'state' : 'national';
        rows.push({ ...alert, locationMatch });
    }
    return rows.sort((a, b) => Number(b.severityScore || 0) - Number(a.severityScore || 0));
}

function buildSachetHazardSummary(sachet, geo, requestedHazard = 'auto', latitude = null, longitude = null) {
    const matched = matchSachetAlerts(sachet?.alerts || [], geo, requestedHazard, latitude, longitude);
    const byHazard = {};
    for (const alert of matched) {
        const key = alert.hazard === 'other' ? 'severe' : alert.hazard;
        if (!byHazard[key] || Number(alert.severityScore) > Number(byHazard[key].severityScore)) byHazard[key] = alert;
    }
    return {
        available: Boolean(sachet?.available),
        source: sachet?.source || 'NDMA SACHET India CAP/RSS feed',
        url: sachet?.url || SACHET_RSS_URL,
        checkedAt: sachet?.checkedAt || new Date().toISOString(),
        matched: matched.length > 0,
        count: matched.length,
        alerts: matched.slice(0, 20),
        byHazard,
        error: sachet?.error || null,
        cached: Boolean(sachet?.cached)
    };
}

async function buildOfficialWeatherWarning(latitude, longitude) {
    const [geo, imd] = await Promise.all([
        reverseGeocodeLocation(latitude, longitude),
        fetchImdDistrictWarnings()
    ]);
    const row = imd.available ? matchImdDistrictWarning(imd.rows, geo) : null;
    const warning = analyzeImdWarning(row);
    const hazardMatches = {};
    for (const hazard of ALERT_HAZARD_KEYS) {
        if (warning.hazards?.includes(hazard)) {
            hazardMatches[hazard] = {
                source: imd.source,
                district: warning.district || geo.district || null,
                state: warning.state || geo.state || null,
                severity: warning.severity,
                codes: warning.codes,
                message: warning.message,
                date: warning.date || null
            };
        }
    }
    return {
        available: imd.available,
        source: imd.source,
        district: geo.district,
        state: geo.state,
        matched: warning.matched,
        hazardMatches,
        warning,
        checkedAt: new Date().toISOString(),
        error: imd.error || null
    };
}


async function fetchImdCycloneTrack() {
    if (IMD_CYCLONE_CACHE.data && Date.now() - IMD_CYCLONE_CACHE.timestamp < IMD_CACHE_TTL_MS) return IMD_CYCLONE_CACHE.data;
    try {
        const response = await fetch('https://api.imd.gov.in/api/v1/cyclone_track', { headers: imdHeaders(), signal: AbortSignal.timeout(12000) });
        if (!response.ok) throw new Error(`IMD cyclone track HTTP ${response.status}`);
        const data = await response.json();
        const result = { available: true, source: 'India Meteorological Department cyclone track', data, checkedAt: new Date().toISOString() };
        IMD_CYCLONE_CACHE.timestamp = Date.now();
        IMD_CYCLONE_CACHE.data = result;
        return result;
    } catch (error) {
        const result = { available: false, source: 'India Meteorological Department cyclone track', error: error.message, checkedAt: new Date().toISOString() };
        IMD_CYCLONE_CACHE.timestamp = Date.now();
        IMD_CYCLONE_CACHE.data = result;
        return result;
    }
}

function computeCycloneIndex(cycloneData, latitude, longitude) {
    const observed = Array.isArray(cycloneData?.data?.observed) ? cycloneData.data.observed : [];
    const forecast = Array.isArray(cycloneData?.data?.forecast) ? cycloneData.data.forecast : [];
    const tracks = [...observed, ...forecast];
    let best = null;
    for (const item of tracks) {
        const lat = Number(item.lat);
        const lon = Number(item.lon);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
        const distanceKm = haversineKm(latitude, longitude, lat, lon);
        if (!best || distanceKm < best.distanceKm) best = { item, distanceKm };
    }
    if (!best) return { available: cycloneData?.available === true, index: 0, active: false, detail: cycloneData?.available ? 'No cyclone track point available.' : 'IMD cyclone track unavailable.' };
    const wind = Number(best.item['Mean MSW (kmph)'] || String(best.item['MSW range (kmph)'] || '').split('-').pop() || 0);
    if (best.distanceKm > 700) return { available: true, index: 0, active: false, detail: `Nearest IMD track is ${best.distanceKm.toFixed(0)} km away.` };
    const distanceSignal = clampScore(((700 - best.distanceKm) / 700) * 70);
    const windSignal = clampScore((wind / 180) * 30);
    return { available: true, index: Math.round(clampScore(distanceSignal + windSignal)), active: true, distanceKm: Number(best.distanceKm.toFixed(1)), windKmph: wind, name: best.item.CYCLONE_NAME || best.item.Cyclone_Name || 'Cyclone', detail: `${best.item.CYCLONE_NAME || 'Cyclone'} · ${best.distanceKm.toFixed(0)} km · mean wind ${Number.isFinite(wind) ? wind : 0} km/h` };
}

function officialScoreForAlert(alert) {
    return clampScore(alert?.severityScore ?? 0);
}

function buildMultiHazardRisk({ hazardSignals, officialWarning, nationalAlerts }) {
    const imdMatches = officialWarning?.hazardMatches || {};
    const sachetMatches = nationalAlerts?.byHazard || {};
    const officialByHazard = {};
    const scores = {
        flood: clampScore(Number(hazardSignals?.flood || 0)),
        heavy_rain: clampScore(Number(hazardSignals?.heavyRain || 0)),
        landslide: clampScore(Number(hazardSignals?.landslide || 0)),
        thunderstorm: clampScore(Number(hazardSignals?.thunderstorm || 0)),
        extreme_heat: clampScore(Number(hazardSignals?.heatwave || 0)),
        cyclone: clampScore(Number(hazardSignals?.cyclone || 0)),
        severe: clampScore(Number(hazardSignals?.severeWeather || 0))
    };

    for (const hazard of ALERT_HAZARD_KEYS.concat('severe')) {
        const sachet = sachetMatches[hazard];
        const imd = imdMatches[hazard];
        const candidates = [];

        if (sachet) {
            const severityScore = officialScoreForAlert(sachet);
            candidates.push({
                source: 'NDMA SACHET India CAP/RSS official alert',
                title: sachet.title,
                severity: sachet.severity || null,
                severityScore,
                area: sachet.areaDesc || null,
                locationMatch: sachet.locationMatch || null,
                publishedAt: sachet.publishedAt || sachet.pubDate || null,
                expiresAt: sachet.expiresAt || sachet.expires || null,
                link: sachet.link || 'https://sachet.ndma.gov.in/'
            });
        }

        if (imd) {
            const severityScore = /EXTREME/i.test(imd.severity) ? 90 : /HIGH/i.test(imd.severity) ? 75 : 50;
            candidates.push({
                source: officialWarning?.source || 'India Meteorological Department district warning',
                title: `${hazard.replace('_', ' ')} district warning`,
                severity: imd.severity,
                severityScore,
                area: imd.district || officialWarning?.district || null,
                locationMatch: 'IMD district warning',
                message: imd.message || null,
                publishedAt: imd.date || null,
                expiresAt: null,
                link: 'https://mausam.imd.gov.in/'
            });
        }

        if (candidates.length) {
            candidates.sort((a, b) => Number(b.severityScore || 0) - Number(a.severityScore || 0));
            const selectedOfficial = candidates[0];
            scores[hazard] = Math.max(scores[hazard], Number(selectedOfficial.severityScore || 0));
            officialByHazard[hazard] = selectedOfficial;
        }
    }
    return { scores, officialByHazard };
}

function buildDailyHazardForecast(weather, hydrology) {
    const daily = weather?.daily || {};
    const times = Array.isArray(daily.time) ? daily.time : [];
    const precip = Array.isArray(daily.precipitation_sum) ? daily.precipitation_sum.map(Number) : [];
    const rain = Array.isArray(daily.rain_sum) ? daily.rain_sum.map(Number) : [];
    const prob = Array.isArray(daily.precipitation_probability_max) ? daily.precipitation_probability_max.map(Number) : [];
    const temps = Array.isArray(daily.temperature_2m_max) ? daily.temperature_2m_max.map(Number) : [];
    const gusts = Array.isArray(daily.wind_gusts_10m_max) ? daily.wind_gusts_10m_max.map(Number) : [];
    const codes = Array.isArray(daily.weather_code) ? daily.weather_code.map(Number) : [];
    const river = Array.isArray(hydrology?.forecastDailyDischarge) ? hydrology.forecastDailyDischarge : [];
    const baseline = Number(hydrology?.recentBaselineM3s || 0);
    const todayKey = new Date().toISOString().slice(0, 10);
    let startIndex = times.findIndex(value => String(value).slice(0, 10) >= todayKey);
    if (startIndex < 0) startIndex = Math.max(0, times.length - 7);
    return times.slice(startIndex, startIndex + 7).map((date, offset) => {
        const i = startIndex + offset;
        const p = Number.isFinite(precip[i]) ? precip[i] : 0;
        const r = Number.isFinite(rain[i]) ? rain[i] : p;
        const rp = Number.isFinite(prob[i]) ? prob[i] : 0;
        const t = Number.isFinite(temps[i]) ? temps[i] : null;
        const g = Number.isFinite(gusts[i]) ? gusts[i] : 0;
        const code = Number.isFinite(codes[i]) ? codes[i] : 0;
        const riverQ = Number(river[i] || 0);
        const riverRatio = baseline > 0 ? riverQ / baseline : null;
        const flood = hydrology?.available
            ? clampScore((p / 80) * 35 + (rp / 100) * 10 + (riverRatio && riverRatio > 1 ? Math.min(55, (riverRatio - 1) * 55) : 0))
            : clampScore((p / 80) * 65 + (rp / 100) * 25 + (r / 80) * 10);
        const heavyRain = clampScore((p / 80) * 65 + (rp / 100) * 25 + (r / 80) * 10);
        const storm = clampScore(([95, 96, 99].includes(code) ? 70 : 0) + (g / 90) * 25 + (rp / 100) * 15);
        const heat = t === null ? 0 : clampScore(((t - 32) / 10) * 100);
        return {
            date,
            flood: Math.round(flood),
            heavyRain: Math.round(heavyRain),
            thunderstorm: Math.round(storm),
            extremeHeat: Math.round(heat),
            precipitationMm: Number(p.toFixed(1)),
            precipitationProbability: Math.round(rp),
            maxTemperatureC: t === null ? null : Number(t.toFixed(1)),
            maxWindGustKmh: Number(g.toFixed(1)),
            weatherCode: code,
            riverDischargeM3s: hydrology?.available && river[i] != null ? Number(riverQ.toFixed(2)) : null,
            methodology: hydrology?.available ? 'Weather forecast + GloFAS river guidance' : 'Weather forecast only'
        };
    });
}

async function buildRegionAnalysis(location, requestedHazard = 'auto') {
    const weather = await fetchOpenMeteoIntelligence(location.latitude, location.longitude);
    const aiFeatures = aggregateWeatherFeatures(weather.hourly);
    aiFeatures.latitude = location.latitude;
    aiFeatures.longitude = location.longitude;
    aiFeatures.month = new Date().getMonth() + 1;

    const model = LANDSLIDE_FEATURES.every(f => Number.isFinite(Number(aiFeatures[f])))
        ? await runLandslideModelForFeatures(aiFeatures)
        : { available: false, reason: 'Incomplete historical weather feature window.' };
    const terrain = await fetchTerrainContext(location.latitude, location.longitude);
    const historical = getHistoricalContext(location.latitude, location.longitude);

    let flood = { available: false, source: 'GloFAS v4 via Open-Meteo Flood API', error: null };
    try { flood = buildHydrologySummary(await fetchOpenMeteoFlood(location.latitude, location.longitude)); }
    catch (error) { flood.error = error.message; }

    const geo = await reverseGeocodeLocation(location.latitude, location.longitude);
    const [official, sachet, cycloneTrack] = await Promise.all([
        buildOfficialWeatherWarning(location.latitude, location.longitude),
        fetchSachetAlerts(),
        fetchImdCycloneTrack()
    ]);
    const sachetSummary = buildSachetHazardSummary(sachet, geo, requestedHazard, location.latitude, location.longitude);
    const cyclone = computeCycloneIndex(cycloneTrack, location.latitude, location.longitude);
    const weatherHazards = summarizeWeatherHazards(weather, flood);
    const landslideContext = computeCompositeRisk({ model, currentWeather: weather.current, historical, terrain, telemetry: null });
    const hazardSignals = {
        ...weatherHazards,
        landslide: model?.percentage ?? landslideContext?.score ?? 0,
        cyclone: cyclone.index
    };
    const riskBundle = buildMultiHazardRisk({
        hazardSignals,
        officialWarning: official,
        nationalAlerts: sachetSummary
    });
    const scores = riskBundle.scores;

    // Auto mode selects the strongest *available signal*. Official alerts win over derived indices.
    const officialCandidates = Object.entries(riskBundle.officialByHazard || {})
        .sort((a, b) => Number(b[1]?.severityScore || 0) - Number(a[1]?.severityScore || 0));
    const selectedHazard = requestedHazard === 'auto'
        ? (officialCandidates[0]?.[0] || Object.entries(scores).sort((a, b) => Number(b[1]) - Number(a[1]))[0]?.[0] || 'severe')
        : requestedHazard;
    const selected = Number(scores[selectedHazard] ?? scores.severe ?? 0);
    const selectedOfficial = riskBundle.officialByHazard?.[selectedHazard] || null;
    const selectedSource = selectedOfficial?.source || (selectedHazard === 'landslide' ? 'RakshaNet landslide ML + contextual signals' : 'Open-Meteo/GloFAS derived live-data index');

    let band = 'LOW';
    if (selected >= 80) band = 'EXTREME';
    else if (selected >= 60) band = 'HIGH';
    else if (selected >= 35) band = 'WATCH';

    const daily = weather.daily || {};
    const forecast = buildDailyHazardForecast(weather, flood);
    return {
        ok: true,
        checkedAt: new Date().toISOString(),
        location: {
            ...location,
            displayName: geo.displayName || location.region || 'Selected location',
            district: geo.district,
            state: geo.state,
            city: geo.city
        },
        selectedHazard,
        requestedHazard,
        selectedScore: Math.round(selected),
        selectedBand: band,
        selectedBasis: selectedOfficial?.source?.includes('SACHET') ? 'OFFICIAL_NDMA_SACHET' : selectedOfficial ? 'OFFICIAL_IMD' : 'DERIVED_LIVE_DATA',
        selectedSource,
        officialHazards: riskBundle.officialByHazard || {},
        selectedOfficialAlert: selectedOfficial ? {
            source: selectedOfficial.source || 'Official warning source',
            title: selectedOfficial.title || 'Official disaster warning',
            severity: selectedOfficial.severity || null,
            severityScore: Number(selectedOfficial.severityScore || 0),
            locationMatch: selectedOfficial.locationMatch || null,
            area: selectedOfficial.area || selectedOfficial.areaDesc || null,
            message: selectedOfficial.message || null,
            publishedAt: selectedOfficial.publishedAt || selectedOfficial.pubDate || selectedOfficial.date || null,
            expiresAt: selectedOfficial.expiresAt || selectedOfficial.expires || null,
            link: selectedOfficial.link || (String(selectedOfficial.source || '').includes('SACHET') ? 'https://sachet.ndma.gov.in/' : 'https://mausam.imd.gov.in/')
        } : null,
        weather: {
            current: weather.current || {},
            daily: daily,
            modelFeatures: aiFeatures
        },
        hazards: {
            scores,
            labels: {
                flood: 'Flood / hydrological stress',
                heavy_rain: 'Heavy rainfall',
                landslide: 'Landslide',
                thunderstorm: 'Thunderstorm',
                extreme_heat: 'Extreme heat',
                cyclone: 'Cyclone',
                severe: 'Severe weather'
            },
            rawWeatherSignals: weatherHazards.raw,
            availability: {
                flood: Boolean(flood.available),
                heavy_rain: true,
                landslide: Boolean(model?.available),
                thunderstorm: true,
                extreme_heat: true,
                cyclone: Boolean(cyclone.available),
                severe: true
            }
        },
        landslide: {
            mlProbabilityPercent: model?.percentage ?? null,
            mlRisk: model?.risk ?? 'UNAVAILABLE',
            compositeContextIndex: landslideContext?.score ?? null,
            basis: 'Current deployed ML model uses historical rainfall/weather and location features. Terrain and historical records remain explicit contextual signals; they are not silently injected into the trained model.'
        },
        hydrology: flood,
        forecast,
        liveEvidence: {
            observed24hRainMm: weatherHazards.raw.observed24hRain,
            observed72hRainMm: weatherHazards.raw.observed72hRain,
            observed7dRainMm: weatherHazards.raw.observed7dRain,
            forecast24hPrecipMm: weatherHazards.raw.forecast24hPrecip,
            forecast72hPrecipMm: weatherHazards.raw.forecast72hPrecip,
            precipitationProbabilityMax: weatherHazards.raw.maxPrecipProbability,
            currentWeatherCode: weatherHazards.raw.weatherCode,
            hydrologyAvailable: Boolean(flood.available),
            officialSources: {
                sachet: Boolean(sachetSummary.available),
                imd: Boolean(official.available)
            },
            officialAlertCount: Object.keys(riskBundle.officialByHazard || {}).length
        },
        officialWarnings: {
            ...official,
            sachet: sachetSummary
        },
        cyclone,
        terrain,
        historical,
        sources: [
            'Open-Meteo current/hourly/daily weather (cached/availability-aware)',
            'GloFAS v4 river discharge via Open-Meteo Flood API',
            'NDMA SACHET India CAP/RSS official disaster alerts',
            'India Meteorological Department district warnings when reachable',
            'India Meteorological Department cyclone track when reachable',
            'RakshaNet historical landslide inventory',
            'OpenTopoData terrain proxy'
        ],
        disclaimer: 'Hazard indices are decision-support signals, not guaranteed probabilities. The landslide percentage is the deployed model output and is preliminary. Official NDMA/SACHET and IMD/district warnings are shown separately and take precedence for emergency action. A source is never marked as present merely because a panel exists; the response carries the actual availability/error state.'
    };
}

async function buildUnifiedIntelligence(location, options = {}) {
    const includeTelemetry = options.includeTelemetry !== false;
    const includeTerrain = options.includeTerrain !== false;

    const weatherPromise = fetchOpenMeteoIntelligence(
        location.latitude,
        location.longitude
    );

    let telemetry = null;
    let telemetryError = null;

    if (
        includeTelemetry &&
        isAssamFocus(
            location.region,
            location.latitude,
            location.longitude
        )
    ) {
        try {
            telemetry = await getAssamSensorData();
        } catch (error) {
            telemetryError = error.message;
        }
    }

    const weather = await weatherPromise;
    const aiFeatures = aggregateWeatherFeatures(weather.hourly);

    aiFeatures.latitude = location.latitude;
    aiFeatures.longitude = location.longitude;
    aiFeatures.month = new Date().getMonth() + 1;

    // The live feature meanings now match training semantics:
    // temp/wind are computed over the previous 24 hours.
    const aiFeatureCount = LANDSLIDE_FEATURES.filter(
        feature => Number.isFinite(Number(aiFeatures[feature]))
    ).length;

    const aiFeaturesReady = aiFeatureCount === LANDSLIDE_FEATURES.length;
    const model = aiFeaturesReady
        ? await runLandslideModelForFeatures(aiFeatures)
        : {
            available: false,
            reason: 'Waiting for complete weather history.'
        };

    const terrain = includeTerrain
        ? await fetchTerrainContext(location.latitude, location.longitude)
        : { available: false, elevationMeters: null, slopePercent: null, source: 'Grid mode: terrain skipped' };

    const historical = getHistoricalContext(location.latitude, location.longitude);
    const explanations = buildSignalExplanations(aiFeatures, historical, terrain);
    const current = weather.current || {};

    let hydrology = { available: false, source: 'GloFAS v4 via Open-Meteo Flood API', error: null };
    try {
        hydrology = buildHydrologySummary(await fetchOpenMeteoFlood(location.latitude, location.longitude));
    } catch (error) {
        hydrology.error = error.message;
    }

    let officialWarnings = { available: false, matched: false, warning: { official: false, matched: false }, error: null };
    try {
        officialWarnings = await buildOfficialWeatherWarning(location.latitude, location.longitude);
    } catch (error) {
        officialWarnings.error = error.message;
    }
    const geo = await reverseGeocodeLocation(location.latitude, location.longitude);
    const [sachetFeed, cycloneTrack] = await Promise.all([fetchSachetAlerts(), fetchImdCycloneTrack()]);
    const sachetSummary = buildSachetHazardSummary(sachetFeed, geo, 'auto', location.latitude, location.longitude);
    const cyclone = computeCycloneIndex(cycloneTrack, location.latitude, location.longitude);

    const weatherHazards = summarizeWeatherHazards(weather, hydrology);
    const forecast = buildDailyHazardForecast(weather, hydrology);
    const landslideContext = computeCompositeRisk({ model, currentWeather: current, historical, terrain, telemetry });
    const liveRiskBundle = buildMultiHazardRisk({
        hazardSignals: { ...weatherHazards, landslide: model?.percentage ?? landslideContext?.score ?? 0, cyclone: cyclone.index },
        officialWarning: officialWarnings,
        nationalAlerts: sachetSummary
    });
    const liveHazardScores = liveRiskBundle.scores;

    const telemetryFeeds = telemetry?.feeds || {};
    const telemetryItems = Object.entries(telemetryFeeds).map(
        ([key, feed]) => ({
            key,
            connected: Boolean(feed?.connected),
            latest: feed?.latest || null,
            stationCount: Number(feed?.stationCount || 0),
            freshness: observationFreshness(feed?.latest?.timestamp)
        })
    );

    const confidence = dataConfidenceScore({
        aiFeatureCount,
        aiFeatureTotal: LANDSLIDE_FEATURES.length,
        telemetry,
        terrain,
        historical
    });

    const compositeRisk = computeCompositeRisk({
        model,
        currentWeather: current,
        historical,
        terrain,
        telemetry
    });

    return {
        ok: true,
        checkedAt: new Date().toISOString(),
        location: { ...location, displayName: geo.displayName || null, district: geo.district, state: geo.state, city: geo.city },
        sources: {
            weather: {
                connected: weather?.available !== false,
                provider: 'Open-Meteo',
                attribution: 'Open-Meteo / weather model data',
                cached: Boolean(weather?.cached),
                stale: Boolean(weather?.stale),
                error: weather?.error || weather?.staleReason || null
            },
            governmentTelemetry: {
                connected: Boolean(telemetry),
                scope: telemetry ? 'Assam NWDP telemetry' : 'No Assam telemetry selected for this location',
                source: telemetry?.source || 'National Water Data Portal',
                error: telemetryError
            },
            satellite: {
                connected: true,
                provider: 'NASA GIBS',
                scope: 'Global Earth-observation imagery; numerical satellite ML features are not claimed unless a processing feed is configured'
            },
            terrain: {
                connected: Boolean(terrain?.available),
                provider: terrain?.source || 'Terrain service'
            },
            historicalInventory: {
                connected: Boolean(historical?.available),
                provider: 'RakshaNet NER historical inventory'
            },
            hydrology: {
                connected: Boolean(hydrology?.available),
                provider: hydrology?.source || 'GloFAS v4 via Open-Meteo Flood API',
                error: hydrology?.error || null
            },
            officialWarnings: {
                connected: Boolean(officialWarnings?.available || sachetSummary?.available),
                provider: officialWarnings?.source || 'IMD district warnings',
                matched: Boolean(officialWarnings?.matched || sachetSummary?.matched),
                imd: officialWarnings,
                sachet: sachetSummary,
                error: officialWarnings?.error || sachetSummary?.error || null
            }
        },
        currentWeather: {
            temperature: Number.isFinite(Number(current.temperature_2m)) ? Number(current.temperature_2m) : null,
            humidity: Number.isFinite(Number(current.relative_humidity_2m)) ? Number(current.relative_humidity_2m) : null,
            precipitation: Number.isFinite(Number(current.precipitation)) ? Number(current.precipitation) : null,
            rain: Number.isFinite(Number(current.rain)) ? Number(current.rain) : null,
            cloudCover: Number.isFinite(Number(current.cloud_cover)) ? Number(current.cloud_cover) : null,
            windSpeed: Number.isFinite(Number(current.wind_speed_10m)) ? Number(current.wind_speed_10m) : null,
            windDirection: Number.isFinite(Number(current.wind_direction_10m)) ? Number(current.wind_direction_10m) : null,
            pressure: Number.isFinite(Number(current.surface_pressure)) ? Number(current.surface_pressure) : null,
            soilMoisture: Number.isFinite(Number(current.soil_moisture_0_to_1cm)) ? Number(current.soil_moisture_0_to_1cm) : null,
            apparentTemperature: Number.isFinite(Number(current.apparent_temperature)) ? Number(current.apparent_temperature) : null,
            weatherCode: Number.isFinite(Number(current.weather_code)) ? Number(current.weather_code) : null,
            windGusts: Number.isFinite(Number(current.wind_gusts_10m)) ? Number(current.wind_gusts_10m) : null,
            cloudCover: Number.isFinite(Number(current.cloud_cover)) ? Number(current.cloud_cover) : null,
            observedAt: weather.current?.time || null,
            unitSystem: {
                temperature: '°C',
                humidity: '%',
                precipitation: 'mm',
                windSpeed: 'km/h',
                windDirection: '°',
                pressure: 'hPa'
            }
        },
        governmentTelemetry: telemetry
            ? {
                source: telemetry.source,
                updatedAt: telemetry.updatedAt,
                feedCount: Number(telemetry.feedCount || 0),
                feeds: telemetryItems
            }
            : { source: null, updatedAt: null, feedCount: 0, feeds: [] },
        context: {
            terrain,
            historical,
            hydrology,
            officialWarnings,
            sachetAlerts: sachetSummary,
            cyclone,
            liveHazardScores,
            forecast,
            liveEvidence: {
                observed24hRainMm: weatherHazards.raw.observed24hRain,
                observed72hRainMm: weatherHazards.raw.observed72hRain,
                observed7dRainMm: weatherHazards.raw.observed7dRain,
                forecast24hPrecipMm: weatherHazards.raw.forecast24hPrecip,
                forecast72hPrecipMm: weatherHazards.raw.forecast72hPrecip,
                precipitationProbabilityMax: weatherHazards.raw.maxPrecipProbability,
                currentWeatherCode: weatherHazards.raw.weatherCode,
                weatherAvailable: weather?.available !== false,
                weatherCached: Boolean(weather?.cached),
                weatherStale: Boolean(weather?.stale),
                weatherError: weather?.error || weather?.staleReason || null,
                hydrologyAvailable: Boolean(hydrology?.available),
                officialSources: {
                    sachet: Boolean(sachetSummary?.available),
                    imd: Boolean(officialWarnings?.available)
                },
                officialAlertCount: Object.keys(liveRiskBundle.officialByHazard || {}).length
            },
            dataConfidenceScore: confidence,
            explanations,
            compositeRisk
        },
        fusion: {
            stage: model?.available ? 'ML_INFERENCE' : 'NORMALIZED',
            inputs: {
                satellite: true,
                currentWeather: true,
                governmentTelemetry: Boolean(telemetry),
                terrain: Boolean(terrain?.available),
                historicalInventory: Boolean(historical?.available),
                soilMoisture: Number.isFinite(Number(current.soil_moisture_0_to_1cm)),
                hydrology: Boolean(hydrology?.available),
                officialWarnings: Boolean(officialWarnings?.available && officialWarnings?.matched),
                nationalDisasterAlerts: Boolean(sachetSummary?.matched)
            },
            aiFeatureReady: aiFeaturesReady,
            aiFeaturesReady,
            aiFeatureCount,
            aiFeatureTotal: LANDSLIDE_FEATURES.length,
            aiFeatures,
            model,
            compositeRisk,
            liveHazardScores,
            hydrology,
            forecast,
            officialWarnings,
            cyclone,
            dataConfidenceScore: confidence,
            modelVersion: ML_MODEL_VERSION
        },
        hydrology,
        terrain,
        historical,
        cyclone,
        officialWarnings,
        liveEvidence: {
            observed24hRainMm: weatherHazards.raw.observed24hRain,
            observed72hRainMm: weatherHazards.raw.observed72hRain,
            observed7dRainMm: weatherHazards.raw.observed7dRain,
            forecast24hPrecipMm: weatherHazards.raw.forecast24hPrecip,
            forecast72hPrecipMm: weatherHazards.raw.forecast72hPrecip,
            precipitationProbabilityMax: weatherHazards.raw.maxPrecipProbability,
            hydrologyAvailable: Boolean(hydrology?.available),
            officialAlertCount: Object.keys(liveRiskBundle.officialByHazard || {}).length
        },
        message: 'RakshaNet combines live weather, hydrology, official warnings when reachable, available telemetry, terrain and historical context. Risk indices are preliminary decision-support signals; official warnings take precedence.'
    };
}


const NER_HAZARD_POINTS = [
    { key: 'assam', name: 'Assam', latitude: 26.2006, longitude: 92.9376 },
    { key: 'arunachal', name: 'Arunachal Pradesh', latitude: 28.2180, longitude: 94.7278 },
    { key: 'meghalaya', name: 'Meghalaya', latitude: 25.4670, longitude: 91.3662 },
    { key: 'manipur', name: 'Manipur', latitude: 24.8170, longitude: 93.9368 },
    { key: 'mizoram', name: 'Mizoram', latitude: 23.7271, longitude: 92.7176 },
    { key: 'nagaland', name: 'Nagaland', latitude: 25.6751, longitude: 94.1086 },
    { key: 'sikkim', name: 'Sikkim', latitude: 27.3389, longitude: 88.6065 },
    { key: 'tripura', name: 'Tripura', latitude: 23.8315, longitude: 91.2868 }
];
const NER_HAZARD_CACHE = { timestamp: 0, data: null };
const NER_HAZARD_CACHE_MS = 10 * 60 * 1000;

async function buildQuickNerPoint(point, sachetFeed, earthquakeFeed = null, cycloneTrack = null) {
    const [weather, flood] = await Promise.all([
        fetchOpenMeteoIntelligence(point.latitude, point.longitude),
        fetchOpenMeteoFlood(point.latitude, point.longitude).catch(() => null)
    ]);
    const hydrology = buildHydrologySummary(flood);
    const aiFeatures = aggregateWeatherFeatures(weather.hourly);
    Object.assign(aiFeatures, { latitude: point.latitude, longitude: point.longitude, month: new Date().getMonth() + 1 });
    const model = LANDSLIDE_FEATURES.every(f => Number.isFinite(Number(aiFeatures[f])))
        ? await runLandslideModelForFeatures(aiFeatures)
        : { available: false, percentage: null };
    const hazards = summarizeWeatherHazards(weather, hydrology);
    const geo = { district: point.name, state: point.name, city: point.name };
    const sachet = buildSachetHazardSummary(sachetFeed, geo, 'auto', point.latitude, point.longitude);
    const cyclone = computeCycloneIndex(cycloneTrack, point.latitude, point.longitude);
    const riskBundle = buildMultiHazardRisk({
        hazardSignals: { ...hazards, landslide: model?.percentage ?? 0, cyclone: cyclone.index },
        officialWarning: null,
        nationalAlerts: sachet
    });
    const scores = riskBundle.scores;
    const events = Array.isArray(earthquakeFeed?.events) ? earthquakeFeed.events
        .map(event => ({ ...event, distanceKm: safeDistanceKm(point.latitude, point.longitude, event.latitude, event.longitude) }))
        .filter(event => Number(event.distanceKm) <= 250)
        .sort((a,b) => Number(a.distanceKm)-Number(b.distanceKm)) : [];
    return {
        ...point,
        checkedAt: new Date().toISOString(),
        scores,
        model: { percentage: model?.percentage ?? null, risk: model?.risk ?? 'UNAVAILABLE' },
        hydrology: { available: hydrology.available, riverStressIndex: hydrology.riverStressIndex ?? null, source: hydrology.source || null },
        cyclone: { available: cyclone.available, active: cyclone.active, index: cyclone.index, detail: cyclone.detail || null, name: cyclone.name || null, distanceKm: cyclone.distanceKm || null },
        earthquakes: events.slice(0, 5),
        officialAlerts: Object.values(sachet.byHazard || {}).map(item => ({ hazard: item.hazard, title: item.title, severity: item.severity || null, severityScore: item.severityScore, area: item.areaDesc || null, link: item.link || 'https://sachet.ndma.gov.in/' }))
    };
}

app.get('/api/ner-hazard-overview', async (_req, res) => {
    if (NER_HAZARD_CACHE.data && Date.now() - NER_HAZARD_CACHE.timestamp < NER_HAZARD_CACHE_MS) return res.json(NER_HAZARD_CACHE.data);
    try {
        const [sachetFeed, earthquakeFeed, cycloneTrack] = await Promise.all([
            fetchSachetAlerts(),
            getLiveEarthquakes().catch(() => ({ events: [] })),
            fetchImdCycloneTrack().catch(() => ({ available: false, data: null }))
        ]);
        const points = await Promise.all(NER_HAZARD_POINTS.map(point => buildQuickNerPoint(point, sachetFeed, earthquakeFeed, cycloneTrack)));
        const result = { ok: true, checkedAt: new Date().toISOString(), source: 'RakshaNet NER live hazard overview', points, disclaimer: 'NER circles are live decision-support indices from current/forecast weather, GloFAS guidance, matched official alerts and nearby earthquake/cyclone feeds; they are not calibrated probabilities. Earthquake markers represent recent USGS activity and are not landslide predictions.' };
        NER_HAZARD_CACHE.timestamp = Date.now();
        NER_HAZARD_CACHE.data = result;
        return res.json(result);
    } catch (error) {
        return res.status(502).json({ ok: false, error: 'Unable to build live NER hazard overview.', details: error.message });
    }
});

app.get('/api/region-analysis', async (req, res) => {
    try {
        const location = resolveIntelligenceLocation(req);
        const hazard = String(req.query?.hazard || 'auto').toLowerCase();
        const validHazards = new Set(['auto', 'flood', 'heavy_rain', 'landslide', 'thunderstorm', 'extreme_heat', 'cyclone', 'severe']);
        const selectedHazard = validHazards.has(hazard) ? hazard : 'auto';
        return res.json(await buildRegionAnalysis(location, selectedHazard));
    } catch (error) {
        console.error('Region analysis API error:', error);
        return res.status(502).json({ ok: false, error: 'Unable to build live region analysis.', details: error.message });
    }
});

app.get('/api/intelligence', async (req, res) => {
    try {
        const location = resolveIntelligenceLocation(req);
        return res.json(await buildUnifiedIntelligence(location, {
            includeTelemetry: true,
            includeTerrain: true
        }));
    } catch (error) {
        console.error('Unified intelligence API error:', error);
        return res.status(502).json({
            ok: false,
            error: 'Unable to build unified intelligence.',
            details: error.message
        });
    }
});

app.post('/api/intelligence/grid', async (req, res) => {
    try {
        const points = Array.isArray(req.body?.points) ? req.body.points : [];
        if (!points.length || points.length > 25) {
            return res.status(400).json({
                ok: false,
                error: 'Provide 1 to 25 coordinate points.'
            });
        }

        const normalized = points.map((point, index) => {
            const latitude = Number(point?.lat);
            const longitude = Number(point?.lon);
            if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
                throw new Error(`Invalid coordinate at point ${index + 1}.`);
            }
            return {
                id: point?.id ?? index,
                latitude,
                longitude,
                region: point?.region || 'Grid point'
            };
        });

        const results = [];
        for (let start = 0; start < normalized.length; start += 5) {
            const chunk = normalized.slice(start, start + 5);
            const chunkResults = await Promise.all(
                chunk.map(point => buildUnifiedIntelligence(point, {
                    includeTelemetry: false,
                    includeTerrain: false
                }))
            );
            results.push(...chunkResults.map((item, index) => ({
                id: chunk[index].id,
                location: item.location,
                weather: item.currentWeather,
                model: item.fusion.model,
                compositeRisk: item.fusion.compositeRisk,
                historical: item.context.historical,
                dataConfidenceScore: item.context.dataConfidenceScore
            })));
        }

        return res.json({
            ok: true,
            count: results.length,
            results,
            message: 'Grid risk values use the existing landslide ML model plus historical context; satellite/terrain numeric features are not silently substituted into the model.'
        });
    } catch (error) {
        console.error('Intelligence grid error:', error);
        return res.status(502).json({
            ok: false,
            error: 'Unable to build intelligence grid.',
            details: error.message
        });
    }
});

app.post('/api/routes/risk', async (req, res) => {
    try {
        const coordinates = Array.isArray(req.body?.coordinates)
            ? req.body.coordinates
            : [];
        const sample = coordinates.filter(Boolean).slice(0, 25).map((point, index) => ({
            id: index,
            lat: Number(point?.lat),
            lon: Number(point?.lon)
        })).filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon));

        if (!sample.length) {
            return res.status(400).json({ ok: false, error: 'No valid route coordinates supplied.' });
        }

        const grid = await buildUnifiedRouteRisk(sample);
        return res.json(grid);
    } catch (error) {
        console.error('Route risk error:', error);
        return res.status(502).json({ ok: false, error: 'Unable to score route risk.', details: error.message });
    }
});

async function buildUnifiedRouteRisk(points) {
    const results = await Promise.all(
        points.map(point => buildUnifiedIntelligence(
            { region: 'Route sample', latitude: point.lat, longitude: point.lon },
            { includeTelemetry: false, includeTerrain: false }
        ))
    );

    const scores = results
        .map(item => Number(item?.fusion?.compositeRisk?.score ?? item?.fusion?.model?.percentage))
        .filter(Number.isFinite);

    return {
        ok: true,
        samples: scores.length,
        maxRisk: scores.length ? Math.max(...scores) : null,
        meanRisk: scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : null,
        sampleResults: results.map(item => ({
            location: item.location,
            risk: item.fusion.compositeRisk?.score ?? item.fusion.model?.percentage ?? null,
            riskBand: item.fusion.compositeRisk?.band ?? item.fusion.model?.risk ?? 'UNAVAILABLE'
        }))
    };
}


// =====================================================
// EARLY WARNING DELIVERY ENGINE
// Web Push + SMS + optional webhook, with dedupe/audit.
// =====================================================

const ALERT_SUBSCRIPTION_FILE = path.join(ROOT, 'data', 'push_subscriptions.json');
const ALERT_DELIVERY_FILE = path.join(ROOT, 'data', 'alert_deliveries.json');
const ALERT_AUTOMATION_ENABLED = String(process.env.ALERT_AUTOMATION_ENABLED || 'false').toLowerCase() === 'true';
const ALERT_AUTOMATION_INTERVAL_MS = Math.max(60 * 1000, Number(process.env.ALERT_AUTOMATION_INTERVAL_MS || 5 * 60 * 1000));
const ALERT_MIN_SCORE = Math.max(0, Math.min(100, Number(process.env.ALERT_MIN_SCORE || 60)));
const ALERT_DEDUP_MINUTES = Math.max(5, Number(process.env.ALERT_DEDUP_MINUTES || 60));

const ALERT_HAZARD_THRESHOLDS = {
    heavy_rain: Math.max(0, Math.min(100, Number(process.env.ALERT_HEAVY_RAIN_SCORE || 60))),
    flood: Math.max(0, Math.min(100, Number(process.env.ALERT_FLOOD_SCORE || 60))),
    landslide: Math.max(0, Math.min(100, Number(process.env.ALERT_LANDSLIDE_SCORE || 60))),
    thunderstorm: Math.max(0, Math.min(100, Number(process.env.ALERT_THUNDERSTORM_SCORE || 60))),
    extreme_heat: Math.max(0, Math.min(100, Number(process.env.ALERT_EXTREME_HEAT_SCORE || 65))),
    cyclone: Math.max(0, Math.min(100, Number(process.env.ALERT_CYCLONE_SCORE || 50)))
};

function ensureJsonArrayFile(filename) {
    try {
        fs.mkdirSync(path.dirname(filename), { recursive: true });
        if (!fs.existsSync(filename)) fs.writeFileSync(filename, '[]', 'utf8');
    } catch (error) {
        console.warn('Alert JSON storage initialization failed:', error.message);
    }
}

ensureJsonArrayFile(ALERT_SUBSCRIPTION_FILE);
ensureJsonArrayFile(ALERT_DELIVERY_FILE);

function readJsonArrayFile(filename) {
    try {
        const value = JSON.parse(fs.readFileSync(filename, 'utf8'));
        return Array.isArray(value) ? value : [];
    } catch (_) {
        return [];
    }
}

function writeJsonArrayFile(filename, rows) {
    const temp = `${filename}.tmp`;
    fs.writeFileSync(temp, JSON.stringify(rows, null, 2), 'utf8');
    fs.renameSync(temp, filename);
}

function normalizePhoneNumber(phone) {
    const raw = String(phone || '').trim();
    if (!raw) return null;
    if (/^\+\d{8,15}$/.test(raw)) return raw;
    // Friendly India-only normalization for the project's primary demo region.
    if (/^91\d{10}$/.test(raw)) return `+${raw}`;
    if (/^\d{10}$/.test(raw)) return `+91${raw}`;
    return null;
}

function buildAlertKey({ userId = 'global', location = 'selected-area', severity = 'HIGH', score = 0, hazard = 'landslide' }) {
    const normalizedLocation = String(location).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'selected-area';
    return `${hazard}:${userId}:${normalizedLocation}:${severity}:${Math.round(Number(score) / 5) * 5}`;
}

function preferredHazardForBand() {
    return 'landslide';
}

const ALERT_HAZARD_KEYS = ['heavy_rain', 'flood', 'landslide', 'thunderstorm', 'extreme_heat', 'cyclone'];

function getUserAlertPreferences(user) {
    const preferences = Array.isArray(user?.alert_preferences)
        ? user.alert_preferences
        : Array.isArray(user?.alertPreferences)
            ? user.alertPreferences
            : ['landslide'];
    return preferences.map(value => String(value).toLowerCase()).filter(value => ALERT_HAZARD_KEYS.includes(value));
}

function userWantsHazard(user) {
    return getUserAlertPreferences(user).length > 0;
}

function preferredLanguageForUser(user) {
    const value = String(user?.preferred_language || user?.preferredLanguage || 'en').toLowerCase();
    return ALERT_TRANSLATIONS[value] ? value : 'en';
}


async function ensureAlertDeliveryTables() {
    if (!DATABASE_CONFIGURED) return false;
    try {
        await pool.query(`
            CREATE TABLE IF NOT EXISTS public.push_subscriptions (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT,
                endpoint TEXT UNIQUE NOT NULL,
                subscription JSONB NOT NULL,
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        `);
        await pool.query(`
            CREATE TABLE IF NOT EXISTS public.alert_deliveries (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT,
                alert_key TEXT NOT NULL,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                provider TEXT,
                destination TEXT,
                error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        `);
        await pool.query(`
            CREATE INDEX IF NOT EXISTS idx_alert_deliveries_key_time
            ON public.alert_deliveries(alert_key, created_at DESC)
        `);
        await pool.query(`ALTER TABLE public.users ADD COLUMN IF NOT EXISTS preferred_language TEXT NOT NULL DEFAULT 'en'`);
        return true;
    } catch (error) {
        console.warn('Alert delivery tables unavailable; local JSON audit fallback remains active:', error.message);
        return false;
    }
}

async function wasRecentlyDelivered(alertKey) {
    const since = new Date(Date.now() - ALERT_DEDUP_MINUTES * 60 * 1000).toISOString();
    try {
        const result = await pool.query(
            'SELECT 1 FROM public.alert_deliveries WHERE alert_key = $1 AND status = $2 AND created_at >= $3 LIMIT 1',
            [alertKey, 'delivered', since]
        );
        if (result.rows.length) return true;
    } catch (_) {}

    return readJsonArrayFile(ALERT_DELIVERY_FILE).some(row =>
        row.alertKey === alertKey &&
        row.status === 'delivered' &&
        Date.parse(row.createdAt || '') >= Date.now() - ALERT_DEDUP_MINUTES * 60 * 1000
    );
}

async function recordAlertDelivery({ userId, alertKey, channel, status, provider, destination, error }) {
    const row = {
        userId: userId || null,
        alertKey,
        channel,
        status,
        provider: provider || null,
        destination: destination || null,
        error: error || null,
        createdAt: new Date().toISOString()
    };
    try {
        await pool.query(
            `INSERT INTO public.alert_deliveries(user_id, alert_key, channel, status, provider, destination, error)\n             VALUES ($1,$2,$3,$4,$5,$6,$7)`,
            [row.userId, row.alertKey, row.channel, row.status, row.provider, row.destination, row.error]
        );
    } catch (_) {
        const rows = readJsonArrayFile(ALERT_DELIVERY_FILE);
        rows.push(row);
        writeJsonArrayFile(ALERT_DELIVERY_FILE, rows.slice(-5000));
    }
}

async function savePushSubscription({ userId, subscription }) {
    if (!subscription?.endpoint) throw new Error('A valid PushSubscription endpoint is required.');
    try {
        await pool.query(
            `INSERT INTO public.push_subscriptions(user_id, endpoint, subscription, enabled, updated_at)\n             VALUES ($1,$2,$3::jsonb,TRUE,NOW())\n             ON CONFLICT(endpoint) DO UPDATE SET user_id=EXCLUDED.user_id, subscription=EXCLUDED.subscription, enabled=TRUE, updated_at=NOW()`,
            [userId || null, subscription.endpoint, JSON.stringify(subscription)]
        );
        return { stored: true, storage: 'postgres' };
    } catch (_) {
        const rows = readJsonArrayFile(ALERT_SUBSCRIPTION_FILE);
        const existing = rows.find(row => row.endpoint === subscription.endpoint);
        const normalized = {
            userId: userId || existing?.userId || null,
            endpoint: subscription.endpoint,
            subscription,
            enabled: true,
            updatedAt: new Date().toISOString()
        };
        if (existing) Object.assign(existing, normalized);
        else rows.push(normalized);
        writeJsonArrayFile(ALERT_SUBSCRIPTION_FILE, rows.slice(-2000));
        return { stored: true, storage: 'json' };
    }
}

async function removePushSubscription(endpoint) {
    if (!endpoint) return;
    try {
        await pool.query('UPDATE public.push_subscriptions SET enabled = FALSE, updated_at = NOW() WHERE endpoint = $1', [endpoint]);
    } catch (_) {}
    const rows = readJsonArrayFile(ALERT_SUBSCRIPTION_FILE).filter(row => row.endpoint !== endpoint);
    writeJsonArrayFile(ALERT_SUBSCRIPTION_FILE, rows);
}

async function listPushSubscriptions(userId = null) {
    try {
        const query = userId
            ? ['SELECT id, user_id, endpoint, subscription FROM public.push_subscriptions WHERE enabled = TRUE AND user_id = $1', [userId]]
            : ['SELECT id, user_id, endpoint, subscription FROM public.push_subscriptions WHERE enabled = TRUE', []];
        const result = await pool.query(query[0], query[1]);
        return result.rows.map(row => ({ ...row, userId: row.user_id, subscription: row.subscription }));
    } catch (_) {
        return readJsonArrayFile(ALERT_SUBSCRIPTION_FILE)
            .filter(row => row.enabled !== false && (!userId || row.userId === userId));
    }
}

function webPushConfigured() {
    return Boolean(
        webPush &&
        process.env.VAPID_PUBLIC_KEY &&
        process.env.VAPID_PRIVATE_KEY &&
        process.env.VAPID_SUBJECT
    );
}

function configureWebPush() {
    if (!webPushConfigured()) return false;
    webPush.setVapidDetails(
        process.env.VAPID_SUBJECT,
        process.env.VAPID_PUBLIC_KEY,
        process.env.VAPID_PRIVATE_KEY
    );
    return true;
}

async function sendWebPush(subscription, payload) {
    if (!webPushConfigured()) {
        throw new Error('Web Push is not configured. Run npm run generate:vapid and set the VAPID environment variables.');
    }
    configureWebPush();
    await webPush.sendNotification(subscription, JSON.stringify(payload), {
        TTL: 300,
        urgency: payload.severity === 'EXTREME' ? 'high' : 'normal'
    });
    return true;
}

async function sendTwilioSms(to, body) {
    const phone = normalizePhoneNumber(to);
    if (!phone) throw new Error('Recipient phone number must be in E.164 format.');
    if (!process.env.TWILIO_ACCOUNT_SID || !process.env.TWILIO_AUTH_TOKEN) {
        throw new Error('Twilio credentials are not configured.');
    }
    const messagingServiceSid = String(process.env.TWILIO_MESSAGING_SERVICE_SID || '').trim();
    const from = String(process.env.TWILIO_FROM_NUMBER || '').trim();
    if (!messagingServiceSid && !from) {
        throw new Error('Set TWILIO_MESSAGING_SERVICE_SID or TWILIO_FROM_NUMBER.');
    }

    const params = new URLSearchParams({
        To: phone,
        Body: String(body || '').slice(0, 1500)
    });
    if (messagingServiceSid) params.set('MessagingServiceSid', messagingServiceSid);
    else params.set('From', from);

    const auth = Buffer.from(`${process.env.TWILIO_ACCOUNT_SID}:${process.env.TWILIO_AUTH_TOKEN}`).toString('base64');
    const response = await fetch(
        `https://api.twilio.com/2010-04-01/Accounts/${encodeURIComponent(process.env.TWILIO_ACCOUNT_SID)}/Messages.json`,
        {
            method: 'POST',
            headers: {
                Authorization: `Basic ${auth}`,
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: params,
            signal: AbortSignal.timeout(15000)
        }
    );
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(payload.message || `Twilio HTTP ${response.status}`);
    }
    return payload;
}

async function sendSms(to, body) {
    if (process.env.TWILIO_ACCOUNT_SID && process.env.TWILIO_AUTH_TOKEN) {
        return { provider: 'Twilio', result: await sendTwilioSms(to, body) };
    }

    const webhook = process.env.SMS_WEBHOOK_URL;
    if (webhook) {
        const phone = normalizePhoneNumber(to);
        if (!phone) throw new Error('Recipient phone number must be in E.164 format.');
        const response = await fetch(webhook, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-RakshaNet-Source': 'RakshaNet' },
            body: JSON.stringify({ to: phone, message: String(body || '').slice(0, 1500), channel: 'sms' }),
            signal: AbortSignal.timeout(15000)
        });
        if (!response.ok) throw new Error(`SMS webhook HTTP ${response.status}`);
        return { provider: 'SMS webhook', result: await response.text().catch(() => '') };
    }

    throw new Error('No SMS provider is configured.');
}

function smsConfigured() {
    return Boolean(
        (process.env.TWILIO_ACCOUNT_SID && process.env.TWILIO_AUTH_TOKEN && (process.env.TWILIO_FROM_NUMBER || process.env.TWILIO_MESSAGING_SERVICE_SID)) ||
        process.env.SMS_WEBHOOK_URL
    );
}

async function getAlertRecipients() {
    let users = [];
    try {
        const result = await pool.query(`
            SELECT id, name, mobile, email, permanent_location, alert_preferences, preferred_language
            FROM public.users
            WHERE COALESCE(status, 'ACTIVE') = 'ACTIVE'
        `);
        users = result.rows.filter(user => userWantsHazard(user));
    } catch (_) {}

    if (users.length) return users;

    return String(process.env.ALERT_RECIPIENT_PHONES || '')
        .split(',')
        .map(normalizePhoneNumber)
        .filter(Boolean)
        .map((mobile, index) => ({
            id: `env-recipient-${index + 1}`,
            name: 'Configured demo recipient',
            mobile,
            permanent_location: {
                name: process.env.ALERT_RECIPIENT_REGION || 'Configured demo area',
                latitude: Number(process.env.ALERT_RECIPIENT_LATITUDE),
                longitude: Number(process.env.ALERT_RECIPIENT_LONGITUDE)
            },
            alert_preferences: ['landslide']
        }));
}


const ALERT_HAZARD_LABELS = {
    heavy_rain: 'Heavy Rainfall',
    flood: 'Flood',
    landslide: 'Landslide',
    thunderstorm: 'Thunderstorm',
    extreme_heat: 'Extreme Heat',
    cyclone: 'Cyclone'
};

function hazardAlertTranslation(hazard, language, band) {
    const labels = ALERT_HAZARD_LABELS[hazard] || 'Disaster';
    const dictionary = {
        en: {
            low: `${labels}: monitor local conditions and official updates.`,
            watch: `${labels} watch: stay alert and monitor local conditions and official warnings.`,
            high: `${labels} warning signal: prepare for possible disruption and follow district instructions.`,
            extreme: `${labels} high-priority warning signal: follow official emergency instructions immediately.`
        },
        hi: {
            low: `${labels}: स्थानीय परिस्थितियों और आधिकारिक अपडेट पर नजर रखें।`,
            watch: `${labels} वॉच: सतर्क रहें और स्थानीय परिस्थितियों तथा आधिकारिक चेतावनियों की निगरानी करें।`,
            high: `${labels} चेतावनी संकेत: संभावित व्यवधान के लिए तैयार रहें और जिला प्रशासन के निर्देशों का पालन करें।`,
            extreme: `${labels} उच्च प्राथमिकता चेतावनी: तुरंत आधिकारिक आपात निर्देशों का पालन करें।`
        },
        as: {
            low: `${labels}: স্থানীয় পৰিস্থিতি আৰু চৰকাৰী আপডেটসমূহ লক্ষ্য ৰাখক।`,
            watch: `${labels} সতৰ্কতা: স্থানীয় পৰিস্থিতি আৰু চৰকাৰী সতৰ্কবাণী নিৰীক্ষণ কৰক।`,
            high: `${labels} সতৰ্ক সংকেত: সম্ভাব্য ব্যাঘাতৰ বাবে সাজু থাকক আৰু জিলা প্ৰশাসনৰ নিৰ্দেশনা অনুসৰণ কৰক।`,
            extreme: `${labels} উচ্চ অগ্ৰাধিকাৰৰ সতৰ্কতা: তৎক্ষণাৎ চৰকাৰী জৰুৰী নিৰ্দেশনা অনুসৰণ কৰক।`
        },
        bn: {
            low: `${labels}: স্থানীয় পরিস্থিতি ও সরকারি আপডেট পর্যবেক্ষণ করুন।`,
            watch: `${labels} সতর্কতা: স্থানীয় পরিস্থিতি ও সরকারি সতর্কবার্তা পর্যবেক্ষণ করুন।`,
            high: `${labels} সতর্ক সংকেত: সম্ভাব্য বিঘ্নের জন্য প্রস্তুত থাকুন এবং জেলা প্রশাসনের নির্দেশ অনুসরণ করুন।`,
            extreme: `${labels} উচ্চ অগ্রাধিকার সতর্কতা: অবিলম্বে সরকারি জরুরি নির্দেশ অনুসরণ করুন।`
        }
    };
    return dictionary[language]?.[band] || dictionary.en[band];
}

function pickAlertScore(hazard, intelligence) {
    const scores = intelligence?.hazards?.scores || intelligence?.fusion?.liveHazardScores || {};
    return Number(scores[hazard] ?? 0);
}

function buildSmsBody({ location, score, band, message }) {
    return [
        `RakshaNet Early Warning | ${band} | Risk index ${Math.round(Number(score) || 0)}/100`,
        String(location || 'Selected area'),
        String(message || 'Monitor local conditions and official warnings.'),
        'Risk index is decision-support. Follow official emergency instructions.'
    ].join(' | ').slice(0, 1500);
}

async function dispatchAlert({
    userId = null,
    mobile = null,
    location = 'Selected area',
    score = 0,
    risk = 'LOW',
    language = 'en',
    officialWarning = false,
    hazard = preferredHazardForBand(),
    force = false,
    officialAlert = null
}) {
    const percentage = Math.max(0, Math.min(100, Number(score || 0)));
    const band = normalizeAlertBand(risk, percentage);
    const hazardThreshold = ALERT_HAZARD_THRESHOLDS[hazard] ?? ALERT_MIN_SCORE;
    if (!force && percentage < hazardThreshold && !officialWarning) {
        return { ok: true, skipped: true, reason: `Score below ${hazard.toUpperCase()} threshold (${hazardThreshold}).`, deliveries: [] };
    }

    const safeLanguage = Object.prototype.hasOwnProperty.call(ALERT_TRANSLATIONS, String(language || 'en').toLowerCase())
        ? String(language || 'en').toLowerCase()
        : 'en';
    const message = officialAlert
        ? `${location}: Official ${ALERT_HAZARD_LABELS[hazard] || 'disaster'} alert — ${officialAlert.title || 'follow the latest official alert'}`
        : `${location}: ${hazardAlertTranslation(hazard, safeLanguage, band)}`;
    const alertKey = buildAlertKey({ userId: userId || 'global', location, severity: band.toUpperCase(), score: percentage, hazard });

    if (!force && await wasRecentlyDelivered(alertKey)) {
        return { ok: true, skipped: true, deduped: true, alertKey, deliveries: [] };
    }

    const deliveries = [];
    const payload = {
        type: 'rakshanet-early-warning',
        title: `RakshaNet ${ALERT_HAZARD_LABELS[hazard] || 'Disaster'} · ${band.toUpperCase()} Warning`,
        body: message,
        severity: band.toUpperCase(),
        score: percentage,
        location,
        officialWarning,
        sourceType: officialAlert ? 'OFFICIAL_SOURCE' : 'RAKSHA_NET_DERIVED_INDEX',
        threshold: hazardThreshold,
        officialAlert: officialAlert ? { title: officialAlert.title, area: officialAlert.areaDesc || null, severity: officialAlert.severity || null, link: officialAlert.link || 'https://sachet.ndma.gov.in/' } : null,
        url: '/pages/intelligence.html#alerts',
        timestamp: new Date().toISOString()
    };

    // App/browser push
    const subscriptions = await listPushSubscriptions(userId);
    for (const item of subscriptions) {
        try {
            await sendWebPush(item.subscription, payload);
            deliveries.push({ channel: 'push', destination: item.endpoint, status: 'delivered', provider: 'Web Push' });
            await recordAlertDelivery({ userId: userId || item.userId, alertKey, channel: 'push', status: 'delivered', provider: 'Web Push', destination: item.endpoint });
        } catch (error) {
            deliveries.push({ channel: 'push', destination: item.endpoint, status: 'failed', provider: 'Web Push', error: error.message });
            await recordAlertDelivery({ userId: userId || item.userId, alertKey, channel: 'push', status: 'failed', provider: 'Web Push', destination: item.endpoint, error: error.message });
            if (/410|404/i.test(error.message)) await removePushSubscription(item.endpoint);
        }
    }

    // SMS
    if (mobile && smsConfigured()) {
        try {
            const smsBody = buildSmsBody({ location, score: percentage, band: band.toUpperCase(), message });
            const sent = await sendSms(mobile, smsBody);
            deliveries.push({ channel: 'sms', destination: normalizePhoneNumber(mobile), status: 'delivered', provider: sent.provider });
            await recordAlertDelivery({ userId, alertKey, channel: 'sms', status: 'delivered', provider: sent.provider, destination: normalizePhoneNumber(mobile) });
        } catch (error) {
            deliveries.push({ channel: 'sms', destination: normalizePhoneNumber(mobile), status: 'failed', provider: 'SMS', error: error.message });
            await recordAlertDelivery({ userId, alertKey, channel: 'sms', status: 'failed', provider: 'SMS', destination: normalizePhoneNumber(mobile), error: error.message });
        }
    }

    // Server-side audit/forwarding webhook is optional and never exposes secrets.
    const webhook = process.env.ALERT_WEBHOOK_URL;
    if (webhook) {
        try {
            const response = await fetch(webhook, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-RakshaNet-Source': 'RakshaNet' },
                body: JSON.stringify({ ...payload, alertKey, channels: deliveries.map(item => item.channel) }),
                signal: AbortSignal.timeout(10000)
            });
            if (!response.ok) throw new Error(`Alert webhook HTTP ${response.status}`);
            deliveries.push({ channel: 'webhook', destination: 'configured webhook', status: 'delivered', provider: 'Alert webhook' });
            await recordAlertDelivery({ userId, alertKey, channel: 'webhook', status: 'delivered', provider: 'Alert webhook', destination: 'configured webhook' });
        } catch (error) {
            deliveries.push({ channel: 'webhook', destination: 'configured webhook', status: 'failed', provider: 'Alert webhook', error: error.message });
            await recordAlertDelivery({ userId, alertKey, channel: 'webhook', status: 'failed', provider: 'Alert webhook', destination: 'configured webhook', error: error.message });
        }
    }

    return {
        ok: deliveries.some(item => item.status === 'delivered'),
        alertKey,
        skipped: false,
        band: band.toUpperCase(),
        score: percentage,
        message,
        deliveries
    };
}

async function runAutomatedAlertScan() {
    if (!ALERT_AUTOMATION_ENABLED) return { ok: true, enabled: false, scanned: 0, delivered: 0 };

    const recipients = await getAlertRecipients();
    let scanned = 0;
    let delivered = 0;
    const results = [];

    for (const user of recipients.slice(0, 100)) {
        const location = user?.permanent_location || {};
        const latitude = Number(location.latitude);
        const longitude = Number(location.longitude);
        const preferences = getUserAlertPreferences(user);
        if (!Number.isFinite(latitude) || !Number.isFinite(longitude) || !preferences.length) continue;

        try {
            scanned++;
            const intelligence = await buildRegionAnalysis({
                region: location.name || 'Monitored area',
                latitude,
                longitude
            }, 'auto');
            const scores = intelligence?.hazards?.scores || {};
            const language = preferredLanguageForUser(user);
            const userResults = [];

            for (const hazard of preferences) {
                const score = Number(scores[hazard] ?? 0);
                const officialAlert = intelligence?.officialHazards?.[hazard] || null;
                const official = Boolean(officialAlert);
                const risk = normalizeAlertBand(null, score).toUpperCase();
                const result = await dispatchAlert({
                    userId: user.id,
                    mobile: user.mobile,
                    location: intelligence.location?.region || location.name || 'Monitored area',
                    score,
                    risk,
                    language,
                    officialWarning: official,
                    hazard,
                    officialAlert
                });
                if (result.deliveries?.some(item => item.status === 'delivered')) delivered++;
                userResults.push({ hazard, score, risk, result });
            }
            results.push({ userId: user.id, preferences, scores, results: userResults });
        } catch (error) {
            results.push({ userId: user.id, error: error.message });
        }
    }

    return { ok: true, enabled: true, scanned, delivered, results };
}


app.get('/api/user/alert-status/:id', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT id, name, mobile, email, permanent_location, alert_preferences, preferred_language, status
            FROM public.users WHERE id = $1 LIMIT 1
        `, [req.params.id]);
        if (!result.rows.length) return res.status(404).json({ ok: false, error: 'User not found.' });
        const row = result.rows[0];
        const subscriptions = await listPushSubscriptions(row.id);
        return res.json({
            ok: true,
            user: {
                id: row.id,
                name: row.name,
                mobile: row.mobile,
                email: row.email,
                permanentLocation: row.permanent_location,
                alertPreferences: row.alert_preferences || [],
                preferredLanguage: row.preferred_language || 'en',
                status: row.status
            },
            delivery: {
                browserPushSubscribed: subscriptions.length > 0,
                smsConfigured: smsConfigured(),
                automationEnabled: ALERT_AUTOMATION_ENABLED,
                automationIntervalMs: ALERT_AUTOMATION_INTERVAL_MS,
                minimumScore: ALERT_MIN_SCORE,
                thresholds: ALERT_HAZARD_THRESHOLDS
            }
        });
    } catch (error) {
        return res.status(500).json({ ok: false, error: 'Unable to read alert status.', details: error.message });
    }
});

app.get('/api/user/alert-preview/:id', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT id, name, mobile, email, permanent_location, alert_preferences, status
            FROM public.users WHERE id = $1 LIMIT 1
        `, [req.params.id]);
        if (!result.rows.length) return res.status(404).json({ ok: false, error: 'User not found.' });
        const user = result.rows[0];
        const location = user.permanent_location || {};
        const latitude = Number(location.latitude);
        const longitude = Number(location.longitude);
        if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
            return res.status(400).json({ ok: false, error: 'A valid permanent location is required before monitoring can run.' });
        }
        const intelligence = await buildRegionAnalysis({
            region: location.name || 'Registered location',
            latitude,
            longitude
        }, 'auto');
        const preferences = getUserAlertPreferences(user);
        const alerts = preferences.map(hazard => {
            const score = Number(intelligence?.hazards?.scores?.[hazard] ?? 0);
            const official = intelligence?.officialHazards?.[hazard] || null;
            return {
                hazard,
                label: ALERT_HAZARD_LABELS[hazard] || hazard,
                score: Math.round(score),
                band: normalizeAlertBand(null, score).toUpperCase(),
                threshold: ALERT_HAZARD_THRESHOLDS[hazard] ?? ALERT_MIN_SCORE,
                wouldAlert: Boolean(official || score >= (ALERT_HAZARD_THRESHOLDS[hazard] ?? ALERT_MIN_SCORE)),
                official: Boolean(official),
                source: official?.source || 'Derived live-data index'
            };
        });
        return res.json({
            ok: true,
            checkedAt: intelligence.checkedAt,
            user: { id: user.id, name: user.name, permanentLocation: location, alertPreferences: preferences },
            alerts,
            sourceStatus: {
                weather: intelligence.sources?.weather || null,
                hydrology: intelligence.sources?.hydrology || null,
                satellite: intelligence.sources?.satellite || null,
                terrain: intelligence.sources?.terrain || null,
                historicalInventory: intelligence.sources?.historicalInventory || null,
                officialWarnings: intelligence.sources?.officialWarnings || null
            },
            note: 'Preview only. No message is sent by this endpoint.'
        });
    } catch (error) {
        return res.status(502).json({ ok: false, error: 'Unable to calculate alert preview.', details: error.message });
    }
});

app.get('/api/notifications/config', (_req, res) => {
    res.json({
        ok: true,
        push: {
            available: Boolean(webPush && process.env.VAPID_PUBLIC_KEY && process.env.VAPID_PRIVATE_KEY && process.env.VAPID_SUBJECT),
            publicKey: process.env.VAPID_PUBLIC_KEY || null
        },
        sms: {
            available: smsConfigured(),
            provider: process.env.TWILIO_ACCOUNT_SID ? 'Twilio' : (process.env.SMS_WEBHOOK_URL ? 'SMS webhook' : null)
        },
        automation: {
            enabled: ALERT_AUTOMATION_ENABLED,
            intervalMs: ALERT_AUTOMATION_INTERVAL_MS,
            minimumScore: ALERT_MIN_SCORE,
            thresholds: ALERT_HAZARD_THRESHOLDS,
            dedupMinutes: ALERT_DEDUP_MINUTES
        }
    });
});

app.post('/api/notifications/subscribe', async (req, res) => {
    try {
        const result = await savePushSubscription({ userId: req.body?.userId || null, subscription: req.body?.subscription });
        return res.json({ ok: true, push: true, ...result });
    } catch (error) {
        return res.status(400).json({ ok: false, error: error.message });
    }
});

app.delete('/api/notifications/subscribe', async (req, res) => {
    try {
        await removePushSubscription(req.body?.endpoint);
        return res.json({ ok: true, removed: true });
    } catch (error) {
        return res.status(400).json({ ok: false, error: error.message });
    }
});

app.post('/api/notifications/test', async (req, res) => {
    try {
        const userId = req.body?.userId || null;
        const subscriptions = await listPushSubscriptions(userId);
        if (!subscriptions.length) {
            return res.status(404).json({ ok: false, error: 'No app/browser push subscription is registered for this account/browser.' });
        }
        const payload = {
            type: 'rakshanet-test-alert',
            title: 'RakshaNet Test Early Warning',
            body: 'This is a test notification. No emergency has been declared.',
            severity: 'TEST',
            score: 0,
            location: 'Local test',
            timestamp: new Date().toISOString()
        };
        const results = [];
        for (const item of subscriptions) {
            try {
                await sendWebPush(item.subscription, payload);
                results.push({ endpoint: item.endpoint, delivered: true });
            } catch (error) {
                results.push({ endpoint: item.endpoint, delivered: false, error: error.message });
            }
        }
        return res.json({ ok: results.some(item => item.delivered), results });
    } catch (error) {
        return res.status(502).json({ ok: false, error: error.message });
    }
});

app.post('/api/alerts/test', async (req, res) => {
    try {
        const configuredPhone = normalizePhoneNumber(process.env.ALERT_TEST_PHONE);
        if (!configuredPhone) {
            return res.status(400).json({ ok: false, error: 'Set ALERT_TEST_PHONE in .env before using the SMS test endpoint.' });
        }
        if (!smsConfigured()) {
            return res.status(501).json({ ok: false, error: 'No SMS provider is configured.' });
        }
        const result = await dispatchAlert({
            userId: 'test-recipient',
            mobile: configuredPhone,
            location: 'RakshaNet local test',
            score: Number(req.body?.score || 82),
            risk: 'EXTREME',
            language: String(req.body?.language || 'en'),
            force: true
        });
        return res.json({ ok: result.ok, test: true, ...result });
    } catch (error) {
        return res.status(502).json({ ok: false, test: true, error: error.message });
    }
});

const ALERT_TRANSLATIONS = {
    en: {
        low: 'Monitor local conditions and official updates.',
        watch: 'Stay alert and monitor local conditions and official warnings.',
        high: 'Prepare for possible disruption and follow district instructions.',
        extreme: 'High-priority warning signal. Follow official emergency instructions immediately.'
    },
    hi: {
        low: 'स्थानीय परिस्थितियों और आधिकारिक अपडेट पर नजर रखें।',
        watch: 'सतर्क रहें और स्थानीय परिस्थितियों तथा आधिकारिक चेतावनियों की निगरानी करें।',
        high: 'संभावित व्यवधान के लिए तैयार रहें और जिला प्रशासन के निर्देशों का पालन करें।',
        extreme: 'उच्च प्राथमिकता का चेतावनी संकेत। तुरंत आधिकारिक आपात निर्देशों का पालन करें।'
    },
    as: {
        low: 'স্থানীয় পৰিস্থিতি আৰু চৰকাৰী আপডেটসমূহ লক্ষ্য ৰাখক।',
        watch: 'সতৰ্ক থাকক আৰু স্থানীয় পৰিস্থিতি আৰু চৰকাৰী সতৰ্কবাণী নিৰীক্ষণ কৰক।',
        high: 'সম্ভাব্য ব্যাঘাতৰ বাবে সাজু থাকক আৰু জিলা প্ৰশাসনৰ নিৰ্দেশনা অনুসৰণ কৰক।',
        extreme: 'উচ্চ অগ্ৰাধিকাৰৰ সতৰ্ক সংকেত। তৎক্ষণাৎ চৰকাৰী জৰুৰী নিৰ্দেশনা অনুসৰণ কৰক।'
    },
    bn: {
        low: 'স্থানীয় পরিস্থিতি ও সরকারি আপডেট পর্যবেক্ষণ করুন।',
        watch: 'সতর্ক থাকুন এবং স্থানীয় পরিস্থিতি ও সরকারি সতর্কবার্তা পর্যবেক্ষণ করুন।',
        high: 'সম্ভাব্য বিঘ্নের জন্য প্রস্তুত থাকুন এবং জেলা প্রশাসনের নির্দেশ অনুসরণ করুন।',
        extreme: 'উচ্চ অগ্রাধিকারের সতর্ক সংকেত। অবিলম্বে সরকারি জরুরি নির্দেশ অনুসরণ করুন।'
    }
};

function normalizeAlertBand(risk, percentage) {
    const p = Number(percentage ?? 0);
    if (String(risk).toUpperCase() === 'EXTREME' || p >= 80) return 'extreme';
    if (String(risk).toUpperCase() === 'HIGH' || p >= 60) return 'high';
    if (String(risk).toUpperCase() === 'WATCH' || p >= 35) return 'watch';
    return 'low';
}

app.post('/api/alerts/evaluate', async (req, res) => {
    try {
        const body = req.body || {};
        const percentage = Math.max(0, Math.min(100, Number(body.percentage || 0)));
        const band = normalizeAlertBand(body.risk, percentage);
        const language = Object.prototype.hasOwnProperty.call(ALERT_TRANSLATIONS, String(body.language || 'en').toLowerCase())
            ? String(body.language || 'en').toLowerCase()
            : 'en';
        const officialWarning = Boolean(body.officialWarning);
        const text = ALERT_TRANSLATIONS[language][band];

        return res.json({
            ok: true,
            alert: {
                severity: band.toUpperCase(),
                percentage,
                location: body.location || 'Selected area',
                message: `${body.location || 'Selected area'}: ${text}`,
                officialWarning,
                hazard: body.hazard || 'landslide',
                sourceType: officialWarning ? 'OFFICIAL_SOURCE' : 'RAKSHA_NET_DERIVED_SIGNAL',
                advisory: 'RakshaNet AI output is preliminary. Official IMD/district warnings take precedence.'
            },
            delivery: {
                inApp: true,
                browser: true,
                push: Boolean(webPushConfigured()),
                sms: smsConfigured(),
                automation: ALERT_AUTOMATION_ENABLED,
                webhookConfigured: Boolean(process.env.ALERT_WEBHOOK_URL || process.env.SMS_WEBHOOK_URL)
            }
        });
    } catch (error) {
        return res.status(400).json({ ok: false, error: 'Unable to evaluate alert.', details: error.message });
    }
});

app.get('/api/alerts/evaluate-hazards', async (req, res) => {
    try {
        const location = resolveIntelligenceLocation(req);
        const hazard = String(req.query?.hazard || 'auto').toLowerCase();
        const intelligence = await buildRegionAnalysis(location, hazard);
        return res.json({ ok: true, location, hazard, scores: intelligence.hazards.scores, selectedScore: intelligence.selectedScore, selectedBand: intelligence.selectedBand, officialWarnings: intelligence.officialWarnings, hydrology: intelligence.hydrology });
    } catch (error) {
        return res.status(502).json({ ok: false, error: 'Unable to evaluate live hazard data.', details: error.message });
    }
});

app.get('/api/alerts/evaluate-from-intelligence', async (req, res) => {
    try {
        const location = resolveIntelligenceLocation(req);
        const intelligence = await buildUnifiedIntelligence(location, {
            includeTelemetry: true,
            includeTerrain: true
        });
        const composite = intelligence?.fusion?.compositeRisk || {};
        const model = intelligence?.fusion?.model || {};
        const requestedLanguage = String(req.query?.language || 'en').toLowerCase();
        const language = Object.prototype.hasOwnProperty.call(ALERT_TRANSLATIONS, requestedLanguage)
            ? requestedLanguage
            : 'en';
        const score = Number(composite.score ?? model.percentage ?? 0);
        const band = normalizeAlertBand(composite.band ?? model.risk, score);
        const officialWarning = Boolean(req.query?.officialWarning === 'true');
        const text = ALERT_TRANSLATIONS[language][band];

        return res.json({
            ok: true,
            location,
            alert: {
                severity: band.toUpperCase(),
                score,
                mlProbabilityPercent: Number(model.percentage ?? 0),
                message: `${location.region}: ${text}`,
                officialWarning,
                compositeBasis: composite.basis || null,
                advisory: 'RakshaNet output is preliminary. Official IMD, district administration and disaster-management instructions take precedence.'
            },
            intelligence
        });
    } catch (error) {
        return res.status(502).json({
            ok: false,
            error: 'Unable to evaluate intelligence alert.',
            details: error.message
        });
    }
});

app.post('/api/alerts/dispatch', async (req, res) => {
    try {
        const body = req.body || {};
        const score = Number(body.score ?? body.percentage ?? 0);
        const result = await dispatchAlert({
            userId: body.userId || null,
            mobile: body.mobile || null,
            location: body.location || 'Selected area',
            score,
            risk: body.risk || normalizeAlertBand(body.risk, score),
            language: body.language || 'en',
            officialWarning: Boolean(body.officialWarning),
            hazard: body.hazard || 'landslide',
            force: Boolean(body.force)
        });

        if (!result.ok && !result.skipped) {
            return res.status(502).json(result);
        }
        return res.json(result);
    } catch (error) {
        return res.status(502).json({ ok: false, delivered: false, error: 'Alert delivery failed.', details: error.message });
    }
});



// =====================================================
// RAKSHANET PULSE 2.0 — SIH26069 NATIONAL WEATHER BIG DATA
//
// Real-source ingestion layer:
// 1. NDMA SACHET official CAP/RSS alerts
// 2. IMD district warnings (requires IMD API key when the API is protected)
// 3. Bluesky public social search for #IMD and weather hashtags
// 4. GDELT public web/news index for India weather signals
// 5. Existing NWIC/NWDP Assam telemetry connection
// 6. Existing RakshaNet citizen emergency reports
//
// No placeholder event records are inserted. A source that cannot be
// reached is reported as unavailable and its data count remains zero.
// =====================================================

const PULSE_STORE_FILE = path.join(ROOT, 'data', 'pulse_events.json');
const PULSE_CACHE = { timestamp: 0, overview: null };
const PULSE_CACHE_MS = 90 * 1000;
const PULSE_AUTOMATION_ENABLED = String(process.env.PULSE_AUTOMATION_ENABLED ?? 'true').toLowerCase() === 'true';
const PULSE_AUTOMATION_INTERVAL_MS = Math.max(2 * 60 * 1000, Number(process.env.PULSE_AUTOMATION_INTERVAL_MS || 5 * 60 * 1000));
const PULSE_MAX_EVENTS = 500;
const PULSE_SUPABASE_TABLE = String(process.env.PULSE_SUPABASE_TABLE || 'pulse_events').trim() || 'pulse_events';
const PULSE_WEATHER_CACHE_MS = 10 * 60 * 1000;
const PULSE_SOCIAL_CACHE_MS = 15 * 60 * 1000;
let pulseDbReadyPromise = null;
let pulseSupabaseReadyPromise = null;
const PULSE_WEATHER_CACHE = { timestamp: 0, data: null };
const PULSE_SOCIAL_CACHE = { timestamp: 0, data: null };
const PULSE_IMD_CAP_CACHE = { timestamp: 0, data: null };
const PULSE_IMD_CAP_CACHE_MS = 5 * 60 * 1000;
const PULSE_WEATHER_LOCATIONS = [
    { city: 'Srinagar', state: 'Jammu and Kashmir' },
    { city: 'Delhi', state: 'Delhi' },
    { city: 'Jaipur', state: 'Rajasthan' },
    { city: 'Mumbai', state: 'Maharashtra' },
    { city: 'Pune', state: 'Maharashtra' },
    { city: 'Bengaluru', state: 'Karnataka' },
    { city: 'Chennai', state: 'Tamil Nadu' },
    { city: 'Kolkata', state: 'West Bengal' },
    { city: 'Guwahati', state: 'Assam' },
    { city: 'Bhubaneswar', state: 'Odisha' },
    { city: 'Hyderabad', state: 'Telangana' },
    { city: 'Lucknow', state: 'Uttar Pradesh' }
];
const PULSE_NWDP_FEED_META = {
    riverLevel: { label: 'River water level', unit: 'm', category: 'other', url: NWDP_SENSOR_SOURCES.riverLevel },
    rainfall: { label: 'Rainfall', unit: 'mm', category: 'rainfall', url: NWDP_SENSOR_SOURCES.rainfall },
    temperature: { label: 'Temperature', unit: '°C', category: 'other', url: NWDP_SENSOR_SOURCES.temperature },
    humidity: { label: 'Relative humidity', unit: '%', category: 'other', url: NWDP_SENSOR_SOURCES.humidity },
    windSpeed: { label: 'Wind speed', unit: 'km/h', category: 'other', url: NWDP_SENSOR_SOURCES.windSpeed },
    windDirection: { label: 'Wind direction', unit: '°', category: 'other', url: NWDP_SENSOR_SOURCES.windDirection },
    pressure: { label: 'Atmospheric pressure', unit: 'mb', category: 'other', url: NWDP_SENSOR_SOURCES.pressure }
};

function pulseNowIso() {
    return new Date().toISOString();
}

function pulseReadLocal() {
    try {
        if (!fs.existsSync(PULSE_STORE_FILE)) return [];
        const raw = fs.readFileSync(PULSE_STORE_FILE, 'utf8');
        const rows = JSON.parse(raw);
        return Array.isArray(rows) ? rows : [];
    } catch (error) {
        console.warn('Pulse local store read failed:', error.message);
        return [];
    }
}

function pulseWriteLocal(rows) {
    try {
        fs.mkdirSync(path.dirname(PULSE_STORE_FILE), { recursive: true });
        fs.writeFileSync(PULSE_STORE_FILE, JSON.stringify(rows.slice(0, PULSE_MAX_EVENTS), null, 2));
    } catch (error) {
        console.warn('Pulse local store write failed:', error.message);
    }
}

async function ensurePulseStorage() {
    if (!DATABASE_CONFIGURED) return false;
    if (pulseDbReadyPromise) return pulseDbReadyPromise;
    pulseDbReadyPromise = (async () => {
        try {
            await pool.query(`
                CREATE TABLE IF NOT EXISTS public.pulse_events (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT UNIQUE NOT NULL,
                    source_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_url TEXT,
                    source_id TEXT,
                    title TEXT,
                    content TEXT,
                    author TEXT,
                    published_at TIMESTAMPTZ,
                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    city TEXT,
                    district TEXT,
                    state TEXT,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    event_category TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    trust_score INTEGER NOT NULL DEFAULT 0,
                    duplicate_of TEXT,
                    media JSONB,
                    metadata JSONB
                );
                CREATE INDEX IF NOT EXISTS pulse_events_published_idx
                    ON public.pulse_events (published_at DESC);
                CREATE INDEX IF NOT EXISTS pulse_events_category_idx
                    ON public.pulse_events (event_category);
                ALTER TABLE public.pulse_events ADD COLUMN IF NOT EXISTS district TEXT;
                CREATE INDEX IF NOT EXISTS pulse_events_state_idx
                    ON public.pulse_events (state);
                CREATE INDEX IF NOT EXISTS pulse_events_verification_idx
                    ON public.pulse_events (verification_status);
            `);
            return true;
        } catch (error) {
            console.warn('Pulse Postgres storage unavailable; local JSON fallback will be used:', error.message);
            return false;
        }
    })();
    return pulseDbReadyPromise;
}

async function ensurePulseSupabaseStorage() {
    if (!supabaseAdmin || DATABASE_CONFIGURED) return false;
    if (pulseSupabaseReadyPromise) return pulseSupabaseReadyPromise;
    pulseSupabaseReadyPromise = (async () => {
        try {
            const { error } = await supabaseAdmin
                .from(PULSE_SUPABASE_TABLE)
                .select('id')
                .limit(1);
            if (error) throw error;
            return true;
        } catch (_) {
            return false;
        }
    })();
    return pulseSupabaseReadyPromise;
}

function pulseSupabaseRowToEvent(row) {
    if (!row || typeof row !== 'object') return null;
    return {
        id: row.id,
        fingerprint: row.fingerprint,
        sourceType: row.source_type,
        sourceName: row.source_name,
        sourceUrl: row.source_url,
        sourceId: row.source_id,
        title: row.title,
        content: row.content,
        author: row.author,
        publishedAt: row.published_at,
        ingestedAt: row.ingested_at,
        city: row.city,
        district: row.district,
        state: row.state,
        latitude: row.latitude,
        longitude: row.longitude,
        eventCategory: row.event_category,
        verificationStatus: row.verification_status,
        trustScore: row.trust_score,
        duplicateOf: row.duplicate_of,
        media: row.media || [],
        metadata: row.metadata || {}
    };
}

function pulseEventToSupabaseRow(event) {
    return {
        id: event.id,
        fingerprint: event.fingerprint,
        source_type: event.sourceType,
        source_name: event.sourceName,
        source_url: event.sourceUrl || null,
        source_id: event.sourceId || null,
        title: event.title || '',
        content: event.content || '',
        author: event.author || null,
        published_at: event.publishedAt || null,
        ingested_at: event.ingestedAt || pulseNowIso(),
        city: event.city || null,
        district: event.district || null,
        state: event.state || null,
        latitude: pulseValidCoordinate(event.latitude, event.longitude) ? Number(event.latitude) : null,
        longitude: pulseValidCoordinate(event.latitude, event.longitude) ? Number(event.longitude) : null,
        event_category: event.eventCategory || 'other',
        verification_status: event.verificationStatus || 'NEEDS_REVIEW',
        trust_score: Math.round(Number(event.trustScore || 0)),
        duplicate_of: event.duplicateOf || null,
        media: Array.isArray(event.media) ? event.media : [],
        metadata: event.metadata || {}
    };
}

function pulseValidCoordinate(latitude, longitude) {
    const lat=Number(latitude), lon=Number(longitude);
    return Number.isFinite(lat) && Number.isFinite(lon) && lat>=-90 && lat<=90 && lon>=-180 && lon<=180 && !(lat===0 && lon===0);
}

async function pulseRepairStoredNwdpTimestamps() {
    if (!DATABASE_CONFIGURED) return 0;
    const dbReady = await ensurePulseStorage();
    if (!dbReady) return 0;

    try {
        const result = await pool.query(`
            SELECT
                id,
                source_type AS "sourceType",
                source_url AS "sourceUrl",
                source_id AS "sourceId",
                title,
                content,
                author,
                published_at AS "publishedAt",
                ingested_at AS "ingestedAt",
                city,
                district,
                state,
                latitude,
                longitude,
                event_category AS "eventCategory",
                verification_status AS "verificationStatus",
                trust_score AS "trustScore",
                duplicate_of AS "duplicateOf",
                media,
                metadata
            FROM public.pulse_events
            WHERE source_type = 'public_dataset'
              AND source_id LIKE 'nwdp:%'
            ORDER BY ingested_at DESC
            LIMIT 500
        `);

        let repaired = 0;
        for (const row of result.rows) {
            const rawTimestamp = row?.metadata?.rawObservationTimestamp || null;
            const parsed = parseObservationDate(rawTimestamp);
            if (!parsed || !Number.isFinite(parsed.getTime())) continue;

            const nextIso = parsed.toISOString();
            const current = row.publishedAt ? new Date(row.publishedAt).toISOString() : null;
            if (current === nextIso) continue;

            const screening = pulseRunIntegrityScreen({ ...row, publishedAt: nextIso });
            const metadata = {
                ...(row.metadata && typeof row.metadata === 'object' ? row.metadata : {}),
                integrityScreen: screening
            };

            await pool.query(
                'UPDATE public.pulse_events SET published_at = $1, metadata = $2::jsonb WHERE id = $3',
                [nextIso, JSON.stringify(metadata), row.id]
            );
            repaired++;
        }

        return repaired;
    } catch (error) {
        if (process.env.RAKSHANET_DEBUG === 'true') {
            console.warn('NWDP timestamp repair skipped:', error.message || error);
        }
        return 0;
    }
}

function pulseNormalizeStoredEvent(row) {
    const content = `${row?.title || ''} ${row?.content || ''}`.trim();
    const detected = pulseDetectHazards(content);
    const current = String(row?.eventCategory || 'other').toLowerCase();
    const safeCurrent = ['rainfall','thunderstorm','flooding','heatwave','fog','dust_storm','strong_winds','other'].includes(current) ? current : 'other';
    const eventCategory = pulseEventCategory(content, safeCurrent);
    const metadata = (row?.metadata && typeof row.metadata === 'object') ? row.metadata : {};
    const loc = pulseFindLocation(content, {
        city: row?.city, district: row?.district, state: row?.state,
        locationBasis: metadata.locationBasis, mentionedPlaces: metadata.mentionedPlaces
    });
    const sourceGeometry = metadata.locationBasis === 'SOURCE_GEOMETRY' || metadata.sourceCoordinates === true;
    const multiArea = loc.locationScope === 'MULTI_AREA' || metadata.locationScope === 'MULTI_AREA';
    const keepCoordinates = pulseValidCoordinate(row?.latitude, row?.longitude) && (!multiArea || sourceGeometry);
    const nextMetadata = {
        ...metadata,
        hazards: detected,
        mentionedPlaces: loc.mentionedPlaces,
        locationScope: loc.locationScope,
        locationBasis: sourceGeometry ? 'SOURCE_GEOMETRY' : (keepCoordinates ? (metadata.locationBasis || loc.locationBasis || null) : (loc.locationBasis || 'TEXT_MENTION')),
        sourceCoordinates: sourceGeometry
    };
    if (multiArea && !sourceGeometry) {
        nextMetadata.locationResolution = 'MULTI_AREA_TEXT_ONLY';
    }
    return {
        ...row,
        city: loc.city || row?.city || null,
        district: loc.district || row?.district || null,
        state: loc.state || row?.state || null,
        latitude: keepCoordinates ? Number(row.latitude) : null,
        longitude: keepCoordinates ? Number(row.longitude) : null,
        eventCategory,
        metadata: nextMetadata
    };
}

async function pulseLoadEvents() {
    const dbReady = await ensurePulseStorage();
    if (dbReady) {
        try {
            const result = await pool.query(`
                SELECT
                    id,
                    fingerprint,
                    source_type AS "sourceType",
                    source_name AS "sourceName",
                    source_url AS "sourceUrl",
                    source_id AS "sourceId",
                    title,
                    content,
                    author,
                    published_at AS "publishedAt",
                    ingested_at AS "ingestedAt",
                    city,
                    district,
                    state,
                    latitude,
                    longitude,
                    event_category AS "eventCategory",
                    verification_status AS "verificationStatus",
                    trust_score AS "trustScore",
                    duplicate_of AS "duplicateOf",
                    media,
                    metadata
                FROM public.pulse_events
                ORDER BY COALESCE(published_at, ingested_at) DESC
                LIMIT $1
            `, [PULSE_MAX_EVENTS]);
            return result.rows.map(pulseNormalizeStoredEvent);
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pulse Postgres read failed; falling back:', error.message);
        }
    }

    const sbReady = await ensurePulseSupabaseStorage();
    if (sbReady) {
        try {
            const { data, error } = await supabaseAdmin
                .from(PULSE_SUPABASE_TABLE)
                .select('*')
                .order('published_at', { ascending: false, nullsFirst: false })
                .limit(PULSE_MAX_EVENTS);
            if (error) throw error;
            return (Array.isArray(data) ? data : []).map(pulseSupabaseRowToEvent).map(pulseNormalizeStoredEvent);
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pulse Supabase read failed; local fallback:', error.message);
        }
    }

    return pulseReadLocal().map(pulseNormalizeStoredEvent);
}

async function pulseUpsertEvent(event) {
    const dbReady = await ensurePulseStorage();
    if (dbReady) {
        try {
            await pool.query(`
                INSERT INTO public.pulse_events (
                    id, fingerprint, source_type, source_name, source_url, source_id,
                    title, content, author, published_at, ingested_at, city, district, state,
                    latitude, longitude, event_category, verification_status, trust_score,
                    duplicate_of, media, metadata
                )
                VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21::jsonb,$22::jsonb
                )
                ON CONFLICT (fingerprint)
                DO UPDATE SET
                    source_url = EXCLUDED.source_url,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    author = EXCLUDED.author,
                    published_at = COALESCE(EXCLUDED.published_at, public.pulse_events.published_at),
                    city = COALESCE(EXCLUDED.city, public.pulse_events.city),
                    district = COALESCE(EXCLUDED.district, public.pulse_events.district),
                    state = COALESCE(EXCLUDED.state, public.pulse_events.state),
                    latitude = COALESCE(EXCLUDED.latitude, public.pulse_events.latitude),
                    longitude = COALESCE(EXCLUDED.longitude, public.pulse_events.longitude),
                    verification_status = EXCLUDED.verification_status,
                    trust_score = EXCLUDED.trust_score,
                    media = EXCLUDED.media,
                    metadata = EXCLUDED.metadata
            `, [
                event.id,
                event.fingerprint,
                event.sourceType,
                event.sourceName,
                event.sourceUrl || null,
                event.sourceId || null,
                event.title || '',
                event.content || '',
                event.author || null,
                event.publishedAt || null,
                event.ingestedAt || pulseNowIso(),
                event.city || null,
                event.district || null,
                event.state || null,
                pulseValidCoordinate(event.latitude, event.longitude) ? Number(event.latitude) : null,
                pulseValidCoordinate(event.latitude, event.longitude) ? Number(event.longitude) : null,
                event.eventCategory || 'other',
                event.verificationStatus || 'NEEDS_REVIEW',
                Math.round(Number(event.trustScore || 0)),
                event.duplicateOf || null,
                JSON.stringify(event.media || []),
                JSON.stringify(event.metadata || {})
            ]);
            return true;
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pulse Postgres upsert failed:', error.message);
        }
    }

    const sbReady = await ensurePulseSupabaseStorage();
    if (sbReady) {
        try {
            const { error } = await supabaseAdmin
                .from(PULSE_SUPABASE_TABLE)
                .upsert(pulseEventToSupabaseRow(event), { onConflict: 'fingerprint' });
            if (!error) return true;
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pulse Supabase upsert failed:', error.message);
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pulse Supabase upsert failed:', error.message);
        }
    }

    const rows = pulseReadLocal();
    const index = rows.findIndex(row => row.fingerprint === event.fingerprint);
    if (index >= 0) rows[index] = { ...rows[index], ...event };
    else rows.unshift(event);
    pulseWriteLocal(rows);
    return true;
}

async function pulseUpdateEventStatus(id, verificationStatus) {
    const allowed = new Set([
        'VERIFIED_OFFICIAL',
        'SOURCE_VERIFIED',
        'CORROBORATED',
        'NEEDS_REVIEW',
        'UNVERIFIED_CITIZEN',
        'LOW_CONFIDENCE'
    ]);
    if (!allowed.has(verificationStatus)) throw new Error('Invalid verification status.');

    const dbReady = await ensurePulseStorage();
    if (dbReady) {
        try {
            const result = await pool.query(`
                UPDATE public.pulse_events
                SET verification_status = $1,
                    trust_score = CASE
                        WHEN $1 = 'VERIFIED_OFFICIAL' THEN GREATEST(trust_score, 98)
                        WHEN $1 = 'SOURCE_VERIFIED' THEN GREATEST(trust_score, 75)
                        WHEN $1 = 'CORROBORATED' THEN GREATEST(trust_score, 75)
                        WHEN $1 = 'LOW_CONFIDENCE' THEN LEAST(trust_score, 39)
                        ELSE trust_score
                    END
                WHERE id = $2
                RETURNING id, verification_status AS "verificationStatus", trust_score AS "trustScore"
            `, [verificationStatus, id]);
            if (result.rows[0]) return result.rows[0];
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pulse Postgres status update failed:', error.message);
        }
    }

    const sbReady = await ensurePulseSupabaseStorage();
    if (sbReady) {
        try {
            const current = await supabaseAdmin.from(PULSE_SUPABASE_TABLE).select('trust_score').eq('id', id).maybeSingle();
            if (current.error) throw current.error;
            if (current.data) {
                let trust = Number(current.data.trust_score || 0);
                if (verificationStatus === 'VERIFIED_OFFICIAL') trust = Math.max(trust, 98);
                if (verificationStatus === 'SOURCE_VERIFIED' || verificationStatus === 'CORROBORATED') trust = Math.max(trust, 75);
                if (verificationStatus === 'LOW_CONFIDENCE') trust = Math.min(trust, 39);
                const updated = await supabaseAdmin.from(PULSE_SUPABASE_TABLE).update({ verification_status: verificationStatus, trust_score: trust }).eq('id', id).select('id,verification_status,trust_score').maybeSingle();
                if (updated.error) throw updated.error;
                if (updated.data) return { id: updated.data.id, verificationStatus: updated.data.verification_status, trustScore: updated.data.trust_score };
            }
        } catch (error) {
            if (process.env.RAKSHANET_DEBUG === 'true') console.warn('Pulse Supabase status update failed:', error.message);
        }
    }

    const rows = pulseReadLocal();
    const index = rows.findIndex(row => row.id === id);
    if (index < 0) return null;
    rows[index].verificationStatus = verificationStatus;
    if (verificationStatus === 'VERIFIED_OFFICIAL') rows[index].trustScore = Math.max(Number(rows[index].trustScore || 0), 98);
    if (verificationStatus === 'SOURCE_VERIFIED' || verificationStatus === 'CORROBORATED') rows[index].trustScore = Math.max(Number(rows[index].trustScore || 0), 75);
    if (verificationStatus === 'LOW_CONFIDENCE') rows[index].trustScore = Math.min(Number(rows[index].trustScore || 0), 39);
    pulseWriteLocal(rows);
    return { id, verificationStatus, trustScore: rows[index].trustScore };
}

function pulseNormalizeText(value) {
    return String(value || '')
        .toLowerCase()
        .normalize('NFKC')
        .replace(/https?:\/\/\S+/gu, ' ')
        // Preserve Unicode letters/numbers so Hindi, Marathi, Odia, Telugu, Bengali, etc.
        // still participate in search and similarity calculations.
        .replace(/[^\p{L}\p{N}#\s]/gu, ' ')
        .replace(/\s+/gu, ' ')
        .trim();
}

function pulseTokenSet(value) {
    return new Set(pulseNormalizeText(value).split(/\s+/).filter(token => token.length > 2));
}

function pulseTextSimilarity(a, b) {
    const aa = pulseTokenSet(a);
    const bb = pulseTokenSet(b);
    if (!aa.size || !bb.size) return 0;
    let intersection = 0;
    for (const token of aa) if (bb.has(token)) intersection++;
    return intersection / (aa.size + bb.size - intersection);
}

const PULSE_STATE_ALIASES = {
    'उत्तर प्रदेश':'Uttar Pradesh','मध्य प्रदेश':'Madhya Pradesh','हिमाचल प्रदेश':'Himachal Pradesh','उत्तराखंड':'Uttarakhand',
    'छत्तीसगढ़':'Chhattisgarh','महाराष्ट्र':'Maharashtra','राजस्थान':'Rajasthan','बिहार':'Bihar','पश्चिम बंगाल':'West Bengal',
    'ओडिशा':'Odisha','झारखंड':'Jharkhand','असम':'Assam','तेलंगाना':'Telangana','आंध्र प्रदेश':'Andhra Pradesh',
    'तमिलनाडु':'Tamil Nadu','कर्नाटक':'Karnataka','केरल':'Kerala','गुजरात':'Gujarat','पंजाब':'Punjab','हरियाणा':'Haryana',
    'सिक्किम':'Sikkim','गोवा':'Goa','मणिपुर':'Manipur','मेघालय':'Meghalaya','मिजोरम':'Mizoram','नागालैंड':'Nagaland',
    'त्रिपुरा':'Tripura','अरुणाचल प्रदेश':'Arunachal Pradesh','दिल्ली':'Delhi','पुडुचेरी':'Puducherry',
    'ଓଡ଼ିଶା':'Odisha','ମହାରାଷ୍ଟ୍ର':'Maharashtra','ପଶ୍ଚିମବଙ୍ଗ':'West Bengal','ଆସାମ':'Assam',
    'తెలంగాణ':'Telangana','ఆంధ్ర ప్రదేశ్':'Andhra Pradesh','ఆంధ్రప్రదేశ్':'Andhra Pradesh',
    'মহারাষ্ট্র':'Maharashtra','পশ্চিমবঙ্গ':'West Bengal'
};

const PULSE_PLACE_ALIASES = {
    'फर्रुखाबाद':'Farrukhabad','फतेहगढ़':'Fatehgarh','फतेहगढ़':'Fatehgarh','मंडला':'Mandla','तुर्तीपार':'Turtipar','खड्डा':'Khadda','अयोध्या':'Ayodhya',
    'बलरामपुर':'Balrampur','कानपुर':'Kanpur','शाहजहांपुर':'Shahjahanpur','अंकिनघाट':'Ankinghat','उत्तarkाशी':'Uttarkashi','उत्तरकाशी':'Uttarkashi',
    'अल्मोड़ा':'Almora','बागेश्वर':'Bageshwar','चंपावत':'Champawat','नैनीताल':'Nainital','पिथौरागढ़':'Pithoragarh','पिथौरगढ़':'Pithoragarh',
    'किन्नौर':'Kinnaur','कुल्लू':'Kullu','लाहौल स्पीति':'Lahaul Spiti','लाहौल-स्पीति':'Lahaul Spiti','शिमला':'Shimla','सिरमौर':'Sirmaur',
    'सातारा':'Satara','कोचबिहार':'Cooch Behar','मेखलीगंज':'Mekhliganj','फरक्का':'Farakka','मुरशिदाबाद':'Murshidabad',
    'जाजपुर':'Jajpur','बालेश्वर':'Balasore','बलेश्वर':'Balasore','ବାଲେଶ୍ୱର':'Balasore','ମଥାନୀ ରୋଡ୍ ବ୍ରିଜ୍':'Mathani Road Bridge',
    'ମଥାନୀ ରୋଡ୍ ବ୍ରିଜ':'Mathani Road Bridge','ଆଖୁଆପଦା':'Akhuapada','ରାଜଘାଟ':'Rajghat','ଜଳକା':'Jalka','ବୈତରଣୀ':'Baitarani',
    'కూనవరం':'Kunavaram','కూనవరం':'Kunavaram','धवळेश्वर':'Dowleswaram','ధవళేశ్వరం':'Dowleswaram','భద్రాచలం':'Bhadrachalam',
    'ಕೂನವರಂ':'Kunavaram','ಕೊಳ':'Kol','কোচবিহার':'Cooch Behar','फारक्का':'Farakka','मुर्शिदाबाद':'Murshidabad'
};

const PULSE_STATE_NAMES = [
    'Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh','Goa','Gujarat','Haryana',
    'Himachal Pradesh','Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra','Manipur',
    'Meghalaya','Mizoram','Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu','Telangana',
    'Tripura','Uttar Pradesh','Uttarakhand','West Bengal','Delhi','Jammu and Kashmir','Ladakh','Puducherry'
];

const PULSE_CITY_NAMES = [
    'Ahmedabad','Ajmer','Alwar','Amritsar','Bengaluru','Bikaner','Bhopal','Bhubaneswar','Chandigarh','Chennai',
    'Chittorgarh','Coimbatore','Dehradun','Delhi','Dholpur','Dausa','Fatehgarh','Gangtok','Gandhinagar','Guwahati',
    'Hyderabad','Indore','Jaipur','Jaisalmer','Jammu','Jodhpur','Kochi','Kohima','Kota','Kolkata','Lucknow',
    'Madhubani','Mandala','Mumbai','Muzaffarpur','Nagpur','Nagaur','Patna','Pithoragarh','Pune','Ranchi','Rewa',
    'Sawai Madhopur','Sikar','Shillong','Shimla','Siliguri','Srinagar','Surat','Thiruvananthapuram','Tonk',
    'Udaipur','Vadodara','Varanasi','Visakhapatnam','Warangal','Jagdalpur','Bhadrachalam','Dibrugarh','Gopalganj',
    'Katihar','Khagaria','Purnea','Araria','Bhagalpur','Sitamarhi','Siwan','Kahalgaon','Supaul','Chamoli',
    'Haridwar','Uttarkashi','Bageshwar','Nainital','Rudraprayag','Tehri Garhwal','Champa(wat)'
];

const PULSE_STATE_HINTS = [
    ['Farrukhabad','Uttar Pradesh'], ['Fatehgarh','Uttar Pradesh'], ['Mandla','Madhya Pradesh'], ['Turtipar','Uttar Pradesh'],
    ['Khadda','Uttar Pradesh'], ['Ayodhya','Uttar Pradesh'], ['Balrampur','Uttar Pradesh'], ['Kanpur','Uttar Pradesh'], ['Shahjahanpur','Uttar Pradesh'],
    ['Neamatighat','Assam'], ['Jorhat','Assam'], ['Dibrugarh','Assam'], ['Charaideo','Assam'], ['Dhemaji','Assam'], ['Lakhimpur','Assam'], ['Sivasagar','Assam'], ['Tinsukia','Assam'],
    ['Jagdalpur','Chhattisgarh'], ['Bastar','Chhattisgarh'], ['Bhadrachalam','Telangana'], ['Bhadradri','Telangana'],
    ['Bhubaneswar','Odisha'], ['Balasore','Odisha'], ['Bhadrak','Odisha'], ['Akhuapada','Odisha'], ['Mathani Road Bridge','Odisha'], ['Farakka','West Bengal'], ['Murshidabad','West Bengal'], ['Mekhliganj','West Bengal'], ['Cooch Behar','West Bengal'],
    ['Satara','Maharashtra'], ['Pune','Maharashtra'], ['Mumbai','Maharashtra'], ['Sikar','Rajasthan'], ['Ajmer','Rajasthan'], ['Alwar','Rajasthan'], ['Jaipur','Rajasthan'], ['Dausa','Rajasthan'],
    ['Champawat','Uttarakhand'], ['Nainital','Uttarakhand'], ['Pithoragarh','Uttarakhand'], ['Bageshwar','Uttarakhand'], ['Chamoli','Uttarakhand'], ['Dehradun','Uttarakhand'], ['Haridwar','Uttarakhand'], ['Uttarkashi','Uttarakhand'], ['Rudraprayag','Uttarakhand'], ['Tehri Garhwal','Uttarakhand'], ['Kinnaur','Himachal Pradesh'], ['Kullu','Himachal Pradesh'], ['Lahaul Spiti','Himachal Pradesh'], ['Shimla','Himachal Pradesh'], ['Sirmaur','Himachal Pradesh'],
    ['Muzaffarpur','Bihar'], ['Sitamarhi','Bihar'], ['Gopalganj','Bihar'], ['Madhubani','Bihar'], ['Katihar','Bihar'], ['Khagaria','Bihar'], ['Purnea','Bihar'], ['Araria','Bihar'], ['Bhagalpur','Bihar'], ['Siwan','Bihar'], ['Supaul','Bihar'], ['Patna','Bihar'], ['Jhanjharpur','Bihar'], ['Jhawa','Bihar'], ['Basua','Bihar'], ['Baltara','Bihar'],
    ['Sahibganj','Jharkhand'], ['Kahalgaon','Bihar']
]

function pulseInferStateFromText(value) {
    const raw = String(value || '');
    const normalized = pulseNormalizeText(raw);
    const explicit = [
        ['उत्तर प्रदेश','Uttar Pradesh'], ['मध्य प्रदेश','Madhya Pradesh'], ['हिमाचल प्रदेश','Himachal Pradesh'], ['उत्तराखंड','Uttarakhand'],
        ['छत्तीसगढ़','Chhattisgarh'], ['राजस्थान','Rajasthan'], ['बिहार','Bihar'], ['पश्चिम बंगाल','West Bengal'], ['ओडिशा','Odisha'], ['झारखंड','Jharkhand'],
        ['असम','Assam'], ['तेलंगाना','Telangana'], ['आंध्र प्रदेश','Andhra Pradesh'], ['महाराष्ट्र','Maharashtra'], ['तमिलनाडु','Tamil Nadu'],
        ['કચ્છ','Gujarat'], ['પશ્ચિમ બંગાળ','West Bengal'], ['ଜମ୍ମୁ','Jammu and Kashmir'], ['ओडिशा','Odisha'], ['पंजाब','Punjab'], ['हरियाणा','Haryana'],
        ['Maharashtra','Maharashtra'], ['Odisha','Odisha'], ['Assam','Assam'], ['Bihar','Bihar'], ['Rajasthan','Rajasthan'], ['Telangana','Telangana'], ['West Bengal','West Bengal'], ['Uttar Pradesh','Uttar Pradesh'], ['Madhya Pradesh','Madhya Pradesh'], ['Uttarakhand','Uttarakhand'], ['Himachal Pradesh','Himachal Pradesh'], ['Chhattisgarh','Chhattisgarh'], ['Jharkhand','Jharkhand'], ['Andhra Pradesh','Andhra Pradesh'], ['Tamil Nadu','Tamil Nadu'], ['Karnataka','Karnataka'], ['Delhi','Delhi']
    ];
    for (const [needle, state] of explicit) if (raw.includes(needle) || normalized.includes(pulseNormalizeText(needle))) return state;
    for (const [place, state] of PULSE_STATE_HINTS) if (normalized.includes(pulseNormalizeText(place))) return state;
    if (/ASDMA|Assam State Disaster|Guwahati|Brahmaputra/.test(raw)) return 'Assam';
    if (/SRC,?\s*Odisha|Odisha.*river|ଜଳକା|ବୈତରଣୀ|ସୁବର୍ଣ୍ଣରେଖା/.test(raw)) return 'Odisha';
    if (/Telangana.*Command|TGiCCC|Bhadrachalam|Godavari/.test(raw)) return 'Telangana';
    return null;
}

function pulseCanonicalState(value) {
    const raw = String(value || '').trim();
    if (!raw) return null;
    if (PULSE_STATE_ALIASES[raw]) return PULSE_STATE_ALIASES[raw];
    const compact = normalizeLocationText(raw);
    const aliases = {
        andhrapradesh:'Andhra Pradesh', arunachalpradesh:'Arunachal Pradesh', assam:'Assam', bihar:'Bihar',
        chhattisgarh:'Chhattisgarh', chattisgarh:'Chhattisgarh', goa:'Goa', gujarat:'Gujarat', haryana:'Haryana',
        himachalpradesh:'Himachal Pradesh', himachal:'Himachal Pradesh', jharkhand:'Jharkhand', karnataka:'Karnataka',
        kerala:'Kerala', madhyapradesh:'Madhya Pradesh', maharashtra:'Maharashtra', manipur:'Manipur', meghalaya:'Meghalaya',
        mizoram:'Mizoram', nagaland:'Nagaland', odisha:'Odisha', orissa:'Odisha', punjab:'Punjab', rajasthan:'Rajasthan',
        sikkim:'Sikkim', tamilnadu:'Tamil Nadu', telangana:'Telangana', tripura:'Tripura', uttarakhand:'Uttarakhand',
        uttrakhand:'Uttarakhand', uttarpradesh:'Uttar Pradesh', westbengal:'West Bengal', delhi:'Delhi',
        jammuandkashmir:'Jammu and Kashmir', ladakh:'Ladakh', puducherry:'Puducherry'
    };
    return aliases[compact] || null;
}

function pulseCanonicalPlace(value) {
    const raw = String(value || '').trim();
    if (!raw) return null;
    return PULSE_PLACE_ALIASES[raw] || raw;
}

function pulseExtractMentionedPlaces(text) {
    const value = String(text || '');
    const places = [];
    const add = value => {
        const raw = String(value || '').replace(/[.()]/g,' ').replace(/\s+/g,' ').trim();
        if (!raw) return;
        const canonical = pulseCanonicalPlace(raw);
        if (canonical && !places.includes(canonical)) places.push(canonical);
    };

    const over = value.match(/\bover\s+(.{2,360}?)(?:\s+in\s+(?:the\s+)?next\b|\s+for\s+the\s+next\b|\s+is\s+likely\b|\s+are\s+likely\b)/i);
    if (over) String(over[1]).split(/,|;|\band\b/i).forEach(add);

    // English river/alert text: "at Jagdalpur in Bastar district of Chhattisgarh"
    const river = value.match(/\bat\s+([^|,]+?)\s+in\s+([^|,]+?)\s+district\s+of\s+([^|,.]+)/i);
    if (river) add(river[1]);

    for (const name of PULSE_CITY_NAMES) {
        const re = new RegExp(`(?:^|[^\\p{L}\\p{N}])${name.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\$&')}(?=$|[^\\p{L}\\p{N}])`, 'iu');
        if (re.test(value)) add(name);
    }

    // Common Indian-script place names that appear in current SACHET alerts.
    for (const name of Object.keys(PULSE_PLACE_ALIASES)) {
        if (value.includes(name)) add(name);
    }
    return [...new Set(places)].slice(0, 20);
}

function pulseFindLocation(text, explicit = {}) {
    const value = String(text || '');
    const lower = value.toLowerCase();
    let state = pulseCanonicalState(explicit.state) || null;
    let city = pulseCanonicalPlace(explicit.city) || null;
    let district = pulseCanonicalPlace(explicit.district) || null;
    let locationBasis = explicit.locationBasis || null;
    const mentionedPlaces = Array.isArray(explicit.mentionedPlaces)
        ? explicit.mentionedPlaces.filter(Boolean).map(pulseCanonicalPlace)
        : pulseExtractMentionedPlaces(value);

    if (!state) {
        for (const name of PULSE_STATE_NAMES) {
            if (lower.includes(name.toLowerCase())) { state = name; break; }
        }
    }
    if (!state) {
        for (const [native, canonical] of Object.entries(PULSE_STATE_ALIASES)) {
            if (value.includes(native)) { state = canonical; break; }
        }
    }
    if (!state) state = pulseInferStateFromText(value);

    // English CAP pattern: "at Jagdalpur in Bastar district of Chhattisgarh"
    const englishRiver = value.match(/\bat\s+([^|,]+?)\s+in\s+([^|,]+?)\s+district\s+of\s+([^|,.]+)/i);
    if (englishRiver) {
        if (!city) city = pulseCanonicalPlace(englishRiver[1]);
        if (!district) district = pulseCanonicalPlace(englishRiver[2]);
        state = pulseCanonicalState(englishRiver[3]) || state;
        locationBasis = 'TEXT_MENTION';
    }

    const englishAt = value.match(/\bat\s+([^|,]+?)\s+in\s+([^|,]+?)\s+district\b/i);
    if (englishAt) {
        if (!city) city = pulseCanonicalPlace(englishAt[1]);
        if (!district) district = pulseCanonicalPlace(englishAt[2]);
        locationBasis = locationBasis || 'TEXT_MENTION';
    }

    // Forecast lists such as "over Ajmer, Beawar, ...".
    const overMatch = value.match(/\bover\s+([^|.;]+?)(?:\s+in\s+next\b|\s+in\s+the\s+next\b)/i);
    if (!city && overMatch) {
        const first = String(overMatch[1]).split(/,|\band\b/i).map(v => v.trim()).filter(Boolean)[0];
        if (first) city = pulseCanonicalPlace(first);
        locationBasis = locationBasis || 'TEXT_MENTION';
    }

    // Indian-language district patterns. These capture the district/state without inventing a coordinate.
    const hiDistrict = value.match(/([^,|]{2,50}?)\s+(?:के\s+)?(?:ज़िले|जिले|जनपद|जिला)\b/iu);
    if (!district && hiDistrict) district = pulseCanonicalPlace(hiDistrict[1].trim());
    const mrDistrict = value.match(/([^,|]{2,50}?)\s+जिल्ह(?:ा|्यात|्यांमध्ये)\b/iu);
    if (!district && mrDistrict) district = pulseCanonicalPlace(mrDistrict[1].trim());
    const bnDistrict = value.match(/([^,|]{2,50}?)\s+জেল[াার]\b/iu);
    if (!district && bnDistrict) district = pulseCanonicalPlace(bnDistrict[1].trim());

    if (!city) {
        const hit = PULSE_CITY_NAMES.find(name => lower.includes(name.toLowerCase()));
        if (hit) { city = hit; locationBasis = locationBasis || 'TEXT_MENTION'; }
    }
    if (!city) {
        for (const [native, canonical] of Object.entries(PULSE_PLACE_ALIASES)) {
            if (value.includes(native)) { city = canonical; locationBasis = locationBasis || 'TEXT_MENTION'; break; }
        }
    }

    const uniquePlaces = [...new Set(mentionedPlaces.filter(p => p && pulseNormalizeText(p).length > 1))];
    return {
        city,
        district,
        state,
        mentionedPlaces: uniquePlaces,
        locationScope: uniquePlaces.length > 1 ? 'MULTI_AREA' : 'SINGLE_AREA',
        locationBasis: locationBasis || (city || district || state ? 'TEXT_MENTION' : null)
    };
}

function pulseDetectHazards(text) {
    const value = String(text || '').toLowerCase();
    const hits = [];
    const hasThunder = /thunderstorm|thunder shower|thundering|lightning|garjan|गर्जन|बिजली|विजे|વજ્ર|మెరుపు|বিদ্যুৎ|ବଜ୍ର/.test(value);
    const hasFlood = /flood|inundation|waterlogging|flash flood|river overflow|flood situation|flood risk|water level|water level is rising|increase of water level|increasing water level|river .*?(above normal|severe flood|extreme flood|continues to flow|flood)|बाढ़|बाढ़|बाढ|पूर|जलमग्न|नदी.*(?:बढ़|बाढ|बाढ़|जलस्तर|सामान्य से ऊपर)|বন্যা|জলমগ্ন|জলস্তর|নদী.*(?:বন্যা|জলস্তর)|వరద|నది.*(?:వరద|నీటిమట్టం)|ବନ୍ୟା|ନଦୀ.*(?:ବନ୍ୟା|ଜଳସ୍ତର)/.test(value);
    const coldSnowContext = /snow|snowfall|blizzard|cold wave|cold day|ground frost|frost|बर्फ|हिमपात|बर्फबारी|स्नोफॉल|शीत लहर|थंडी|थंड|বরফ|শৈত্যপ্রবাহ|ਬਰਫ|ਬਰਫ਼ਬਾਰੀ|ଵରଫ|ବରଫ|మంచు|చలిగాలి/.test(value);
    const hasHeatRaw = /heat ?wave|heatwave|hot day|severe heat|heat alert|(?:^|[^\p{L}\p{M}])लू(?:$|[^\p{L}\p{M}])|हीट\s*वेव|उष्णतेची लाट|তাপপ্রবাহ|వడగాల్పు/u.test(value);
    const hasHeat = hasHeatRaw && !coldSnowContext;
    const hasFog = /fog|dense fog|poor visibility|mist|कोहरा|धुंध|धुके|কুয়াশা|କୁହୁଡି/.test(value);
    const hasDust = /dust storm|duststorm|blowing dust|धूल भरी आंधी|धुळीचे वादळ|ধুলিঝড়|ଧୂଳି ଝଡ/.test(value);
    const hasWind = /strong wind|strong surface wind|gusty wind|gusting|high wind|gale|squally|squall|wind speed\s*\d{2}|\d{2}\s*-\s*\d{2}\s*kmph|तेज़ हव|तेज हव|आंधी|वादळ|जोरदार वारे|वारे वाह|জোরালো হাওয়া|పెద్ద గాలులు|తీవ్ర గాలులు/.test(value);
    const hasRain = /very heavy rain|extremely heavy rain|heavy rainfall|heavy rain|intense rainfall|rainfall warning|snowfall|rain|बारिश|वर्षा|पाऊस|बৃষ্টি|বর্ষণ|ବର୍ଷା|వర్ష/.test(value);
    if (hasFlood) hits.push('flooding');
    if (hasThunder) hits.push('thunderstorm');
    if (hasHeat) hits.push('heatwave');
    if (hasFog) hits.push('fog');
    if (hasDust) hits.push('dust_storm');
    if (hasRain) hits.push('rainfall');
    if (hasWind) hits.push('strong_winds');
    return [...new Set(hits)];
}

function pulseHazardPriority(hazards) {
    const order = ['flooding','thunderstorm','heatwave','fog','dust_storm','rainfall','strong_winds'];
    return order.find(item => hazards.includes(item)) || null;
}

function pulseEventCategory(text, fallback = 'other', preferred = null) {
    const hazards = pulseDetectHazards(text);
    const preferredValue = String(preferred || '').toLowerCase();
    if (preferredValue && preferredValue !== 'other') {
        const preferredMap = {
            flood:'flooding', flooding:'flooding', thunderstorm:'thunderstorm', heavy_rain:'rainfall', rainfall:'rainfall',
            heatwave:'heatwave', extreme_heat:'heatwave', fog:'fog', dust_storm:'dust_storm', dust:'dust_storm',
            strong_winds:'strong_winds', wind:'strong_winds'
        };
        const mapped = preferredMap[preferredValue];
        if (mapped && hazards.includes(mapped)) return mapped;
    }
    const inferred = pulseHazardPriority(hazards);
    if (inferred) return inferred;
    const safeFallback = String(fallback || 'other').toLowerCase();
    return hazards.includes(safeFallback) ? safeFallback : 'other';
}



const PULSE_GEOCODE_FILE = path.join(ROOT, 'data', 'pulse_geocodes.json');
const PULSE_GEOCODE_TTL_MS = 30 * 24 * 60 * 60 * 1000;
let pulseGeocodeCache = null;

function pulseReadGeoCache() {
    if (pulseGeocodeCache) return pulseGeocodeCache;
    try {
        pulseGeocodeCache = fs.existsSync(PULSE_GEOCODE_FILE)
            ? JSON.parse(fs.readFileSync(PULSE_GEOCODE_FILE, 'utf8'))
            : {};
        if (!pulseGeocodeCache || typeof pulseGeocodeCache !== 'object') pulseGeocodeCache = {};
    } catch (_) { pulseGeocodeCache = {}; }
    return pulseGeocodeCache;
}

function pulseWriteGeoCache() {
    try {
        fs.mkdirSync(path.dirname(PULSE_GEOCODE_FILE), { recursive: true });
        fs.writeFileSync(PULSE_GEOCODE_FILE, JSON.stringify(pulseReadGeoCache(), null, 2));
    } catch (_) {}
}

async function pulseGeocode(query, expectedState = null, expectedDistrict = null) {
    const clean = String(query || '').trim();
    if (!clean) return null;
    const key = pulseNormalizeText(clean);
    const cache = pulseReadGeoCache();
    const cached = cache[key];
    if (cached && Date.now() - Number(cached.timestamp || 0) < PULSE_GEOCODE_TTL_MS) return cached.result || null;

    try {
        const url = new URL('https://geocoding-api.open-meteo.com/v1/search');
        url.search = new URLSearchParams({ name: clean, count: '10', language: 'en', format: 'json', countryCode: 'IN' });
        const response = await fetch(url, {
            headers: { 'User-Agent': 'RakshaNet-Pulse/2.0', 'Accept': 'application/json' },
            signal: AbortSignal.timeout(8000)
        });
        if (!response.ok) throw new Error(`Geocoder HTTP ${response.status}`);
        const data = await response.json();
        const results = Array.isArray(data?.results) ? data.results : [];
        const stateNeedle = pulseNormalizeText(expectedState || '');
        const districtNeedle = pulseNormalizeText(expectedDistrict || '');
        const candidates = results.filter(item => String(item?.country_code || '').toUpperCase() === 'IN' && Number.isFinite(Number(item?.latitude)) && Number.isFinite(Number(item?.longitude)));
        const scored = candidates.map(item => {
            let score = 0;
            if (stateNeedle && pulseNormalizeText(item?.admin1 || '') === stateNeedle) score += 4;
            if (districtNeedle && pulseNormalizeText(item?.admin2 || '') === districtNeedle) score += 3;
            if (pulseNormalizeText(item?.name || '') === pulseNormalizeText(clean)) score += 2;
            if (item?.feature_code === 'PPLA' || item?.feature_code === 'PPLA2') score += 1;
            return { item, score };
        }).sort((a,b) => b.score - a.score);
        const chosen = scored[0]?.item || null;
        cache[key] = { timestamp: Date.now(), result: chosen ? {
            name: chosen.name || clean,
            latitude: Number(chosen.latitude),
            longitude: Number(chosen.longitude),
            admin1: chosen.admin1 || null,
            admin2: chosen.admin2 || null,
            country: chosen.country || 'India',
            countryCode: chosen.country_code || 'IN'
        } : null };
        pulseWriteGeoCache();
        return cache[key].result;
    } catch (_) {
        cache[key] = { timestamp: Date.now(), result: null };
        pulseWriteGeoCache();
        return null;
    }
}

async function pulseEnrichLocations(events) {
    const tasks = new Map();
    for (const event of events) {
        const valid = Number.isFinite(Number(event.latitude)) && Number.isFinite(Number(event.longitude)) &&
            !(Number(event.latitude) === 0 && Number(event.longitude) === 0);
        if (valid) continue;
        const multiArea = event.metadata?.locationScope === 'MULTI_AREA' || (Array.isArray(event.metadata?.mentionedPlaces) && event.metadata.mentionedPlaces.length > 1);
        // Never geocode one arbitrary place when the source clearly covers many areas.
        // Multi-area geometry will still be plotted when the source supplied it.
        if (multiArea && event.metadata?.locationBasis !== 'SOURCE_GEOMETRY') continue;
        const primaryPlace = event.city || null;
        const fallbackDistrict = event.district || null;
        if (!primaryPlace && !fallbackDistrict) continue;
        const primaryQuery = primaryPlace ? [primaryPlace, event.district, event.state, 'India'].filter(Boolean).join(', ') : null;
        const fallbackQuery = !primaryPlace && fallbackDistrict ? [fallbackDistrict, event.state, 'India'].filter(Boolean).join(', ') : null;
        const queries = [primaryQuery, fallbackQuery].filter(Boolean);
        for (const query of queries) {
            if (!tasks.has(query)) tasks.set(query, []);
        }
        const targetQuery = primaryQuery || fallbackQuery;
        if (targetQuery) tasks.get(targetQuery).push(event);
    }
    const entries = Array.from(tasks.entries()).slice(0, 8);
    const results = await Promise.all(entries.map(([query, items]) => {
        const sample = items[0] || {};
        return pulseGeocode(query, sample.state || null, sample.district || null);
    }));
    entries.forEach(([query, items], index) => {
        const result = results[index];
        if (!result) return;
        for (const event of items) {
            // Revalidate administrative match before applying derived coordinates.
            if (event.state && result.admin1 && pulseNormalizeText(event.state) !== pulseNormalizeText(result.admin1)) continue;
            event.latitude = result.latitude;
            event.longitude = result.longitude;
            if (!event.city && result.name) event.city = result.name;
            if (!event.state && result.admin1) event.state = result.admin1;
            event.metadata = {
                ...(event.metadata || {}),
                locationBasis: event.city ? 'DERIVED_GEOCODE' : 'DERIVED_DISTRICT_GEOCODE',
                locationSource: 'Open-Meteo Geocoding API',
                locationQuery: query,
                sourceCoordinates: false,
                locationResolution: event.city ? 'PLACE' : 'DISTRICT'
            };
        }
    });
    return events;
}

function pulseSourceBaseTrust(sourceType) {
    return {
        official: 96,
        public_dataset: 91,
        weather_api: 90,
        citizen: 55,
        social: 42,
        web: 48
    }[sourceType] ?? 40;
}

function pulseBuildTrust({ sourceType, title, content, city, state, latitude, longitude, mediaCount = 0, corroborated = false }) {
    let score = pulseSourceBaseTrust(sourceType);
    if (city || state) score += 4;
    if (Number.isFinite(Number(latitude)) && Number.isFinite(Number(longitude))) score += 5;
    if (mediaCount > 0) score += 3;
    if (corroborated) score += 12;
    if (/100%|guaranteed|share immediately|forward this|panic|confirmed everywhere|breaking!!!/.test(String(`${title} ${content}`).toLowerCase())) score -= 15;
    if (String(content || '').trim().length < 25) score -= 5;
    return Math.round(Math.max(0, Math.min(100, score)));
}

function pulseRunIntegrityScreen(event) {
    const content = String(event?.content || '').trim();
    const sourceType = String(event?.sourceType || '').toLowerCase();
    const flags = [];
    let score = 100;
    if (content.length < 25) { score -= 12; flags.push('very_short_content'); }
    if (/(.)\1{7,}/u.test(content)) { score -= 10; flags.push('repeated_characters'); }
    if (/[!?]{4,}/.test(content)) { score -= 8; flags.push('excessive_punctuation'); }
    if ((content.match(/[A-Z]/g) || []).length > 25 && (content.match(/[a-z]/g) || []).length < 8) { score -= 6; flags.push('unusual_capitalization'); }
    if (/guaranteed|100% true|share immediately|forward this|panic|confirmed everywhere|breaking!!!/i.test(content)) { score -= 18; flags.push('urgency_manipulation'); }
    if (sourceType === 'social' && !event?.author) { score -= 7; flags.push('missing_author'); }
    if (['social','web','citizen'].includes(sourceType) && !event?.sourceUrl) { score -= 8; flags.push('missing_source_url'); }
    if (sourceType === 'citizen' && !event?.latitude && !event?.longitude) { score -= 5; flags.push('no_coordinates'); }
    if (event?.metadata?.locationScope === 'MULTI_AREA' && event?.latitude && event?.longitude && event?.metadata?.locationBasis !== 'SOURCE_GEOMETRY') { score -= 20; flags.push('multi_area_single_point'); }
    const published = new Date(event?.publishedAt || '').getTime();
    if (Number.isFinite(published) && published > Date.now() + 10 * 60 * 1000) { score -= 15; flags.push('future_timestamp'); }
    score = Math.max(0, Math.min(100, Math.round(score)));
    return { score, flags, decision: score < 40 ? 'LOW_CONFIDENCE' : (flags.length ? 'NEEDS_REVIEW' : 'NORMAL_SCREENING') };
}

function pulseApplyIntegrityScreen(event) {
    const screening = pulseRunIntegrityScreen(event);
    event.metadata = { ...(event.metadata || {}), integrityScreen: screening };
    if (!['official','public_dataset','weather_api'].includes(event.sourceType)) {
        if (screening.score < 40) event.verificationStatus = 'LOW_CONFIDENCE';
        else if (event.verificationStatus === 'NEEDS_REVIEW' && screening.score < 60) event.verificationStatus = 'LOW_CONFIDENCE';
        event.trustScore = Math.min(Number(event.trustScore || 0), screening.score);
    }
    return event;
}

function pulseInitialVerification(sourceType, trustScore) {
    if (sourceType === 'official') return 'VERIFIED_OFFICIAL';
    if (sourceType === 'public_dataset' || sourceType === 'weather_api') return 'SOURCE_VERIFIED';
    if (sourceType === 'citizen') return 'UNVERIFIED_CITIZEN';
    return trustScore < 40 ? 'LOW_CONFIDENCE' : 'NEEDS_REVIEW';
}

function pulseFingerprint(event) {
    const key = [
        event.sourceType,
        event.sourceId || '',
        event.eventCategory || 'other',
        event.publishedAt || '',
        event.city || '',
        event.state || '',
        pulseNormalizeText(`${event.title} ${event.content}`).slice(0, 220)
    ].join('|');
    return crypto.createHash('sha256').update(key).digest('hex');
}

function pulseMakeId(prefix, seed) {
    const safe = String(seed || '').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 34);
    return `PULSE-${prefix}-${safe || crypto.randomBytes(6).toString('hex')}`;
}

function pulseExtractPostRkey(uri) {
    const parts = String(uri || '').split('/');
    return parts[parts.length - 1] || null;
}


function pulseCapField(xml, name) {
    const escaped = String(name || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp('<(?:(?:cap):)?' + escaped + '\\b[^>]*>([\\s\\S]*?)<\\/(?:(?:cap):)?' + escaped + '>', 'i');
    const match = String(xml || '').match(re);
    return match ? decodePulseXml(match[1]).replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim() : '';
}

function pulseCapAreaRecords(xml) {
    const areas = [];
    const blocks = String(xml || '').match(/<(?:(?:cap):)?area\b[^>]*>[\s\S]*?<\/(?:(?:cap):)?area>/gi) || [];
    for (const block of blocks) {
        const areaDesc = pulseCapField(block, 'areaDesc');
        const polygonRaw = pulseCapField(block, 'polygon');
        const circleRaw = pulseCapField(block, 'circle');
        const polygon = polygonRaw.split(/\s+/).map(pair => pair.split(',').map(Number)).filter(pair => pair.length === 2 && pair.every(Number.isFinite));
        const circleParts = circleRaw.split(/\s*,\s*/).map(Number);
        const circle = circleParts.length >= 2 && circleParts.slice(0,2).every(Number.isFinite) ? { latitude:circleParts[0], longitude:circleParts[1] } : null;
        if (areaDesc || polygon.length || circle) areas.push({ areaDesc, polygon, circle });
    }
    return areas;
}

function pulseParseImdCapXml(xml, fallback = {}) {
    const areas = pulseCapAreaRecords(xml);
    return {
        guid: pulseCapField(xml, 'identifier') || fallback.guid || null,
        sent: pulseCapField(xml, 'sent') || fallback.pubDate || null,
        event: pulseCapField(xml, 'event') || fallback.title || '',
        headline: pulseCapField(xml, 'headline') || fallback.title || '',
        description: pulseCapField(xml, 'description') || fallback.description || '',
        instruction: pulseCapField(xml, 'instruction') || '',
        urgency: pulseCapField(xml, 'urgency') || null,
        severity: pulseCapField(xml, 'severity') || null,
        certainty: pulseCapField(xml, 'certainty') || null,
        expires: pulseCapField(xml, 'expires') || null,
        areas
    };
}

function parseImdCapRss(xml) {
    const items = [];
    const blocks = String(xml || '').match(/<item\b>[\s\S]*?<\/item>|<entry\b>[\s\S]*?<\/entry>/gi) || [];
    for (const block of blocks) {
        const tag = name => {
            const direct = block.match(new RegExp('<' + name + '\\b[^>]*>([\\s\\S]*?)<\\/' + name + '>', 'i'));
            if (direct) return decodePulseXml(direct[1]).trim();
            if (name === 'link') {
                const attr = block.match(/<link\b[^>]*href=["']([^"']+)["'][^>]*\/?\s*>/i);
                return attr ? decodePulseXml(attr[1]).trim() : '';
            }
            return '';
        };
        const title = tag('title');
        const link = tag('link');
        const pubDate = tag('pubDate') || tag('published') || tag('updated');
        const description = (tag('description') || tag('summary') || tag('content')).replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').trim();
        const source = tag('source') || 'Public feed';
        if (title || link) items.push({ title, link:link || null, pubDate:pubDate || null, description, source });
    }
    return items;
}

async function fetchPulseImdCapRss() {
    const url = 'https://cap-sources.s3.amazonaws.com/in-imd-en/rss.xml';
    if (PULSE_IMD_CAP_CACHE.data && Date.now() - PULSE_IMD_CAP_CACHE.timestamp < PULSE_IMD_CAP_CACHE_MS) return PULSE_IMD_CAP_CACHE.data;
    try {
        const response = await fetch(url, { headers:{ 'User-Agent':'RakshaNet-Pulse/2.0','Accept':'application/rss+xml,application/xml,text/xml' }, signal:AbortSignal.timeout(12000) });
        if (!response.ok) throw new Error(`IMD CAP RSS HTTP ${response.status}`);
        const rss = await response.text();
        const items = parseImdCapRss(rss).slice(0, 60);
        const alerts = [];
        for (const item of items) {
            let cap = null;
            if (item.link) {
                try {
                    const capResponse = await fetch(item.link, { headers:{ 'User-Agent':'RakshaNet-Pulse/2.0','Accept':'application/xml,text/xml' }, signal:AbortSignal.timeout(8000) });
                    if (capResponse.ok) cap = pulseParseImdCapXml(await capResponse.text(), item);
                } catch (_) {}
            }
            const value = cap || pulseParseImdCapXml('', item);
            const validPolygons = (value.areas || []).flatMap(a => a.polygon || []).filter(pt => Array.isArray(pt) && pt.length === 2 && pt.every(Number.isFinite));
            const firstCircle = (value.areas || []).map(a=>a.circle).find(Boolean) || null;
            alerts.push({
                guid:value.guid || item.link || item.title,
                link:item.link || url,
                publishedAt:value.sent || item.pubDate || null,
                title:value.headline || item.title || 'IMD CAP alert',
                description:[value.description,value.instruction].filter(Boolean).join(' | '),
                areaDesc:(value.areas || []).map(a=>a.areaDesc).filter(Boolean).join('; ') || null,
                polygon:validPolygons,
                circle:firstCircle,
                hazard:value.event || item.title || null,
                severity:value.severity || null,
                urgency:value.urgency || null,
                certainty:value.certainty || null,
                expiresAt:value.expires || null
            });
        }
        const result={available:alerts.length>0,rows:alerts,source:'India Meteorological Department CAP RSS alerts',sourceName:'IMD CAP / WMO Alert Hub RSS',sourceUrl:url,provider:'IMD CAP RSS',checkedAt:pulseNowIso(),error:alerts.length?null:'IMD CAP RSS returned no alert items.'};
        PULSE_IMD_CAP_CACHE.timestamp=Date.now(); PULSE_IMD_CAP_CACHE.data=result; return result;
    } catch (error) {
        const result={available:false,rows:[],source:'India Meteorological Department CAP RSS alerts',sourceName:'IMD CAP / WMO Alert Hub RSS',sourceUrl:url,provider:'IMD CAP RSS',checkedAt:pulseNowIso(),error:error.message};
        PULSE_IMD_CAP_CACHE.timestamp=Date.now(); PULSE_IMD_CAP_CACHE.data=result; return result;
    }
}

async function fetchPulseImdCombined() {
    const api = await fetchImdDistrictWarnings();
    if (api?.available && Array.isArray(api.rows) && api.rows.length) return api;
    const cap = await fetchPulseImdCapRss();
    if (cap?.available && Array.isArray(cap.rows) && cap.rows.length) return { ...cap, apiError: api?.error || null, fallback:'CAP_RSS' };
    return { ...api, sourceName:'IMD district warnings / CAP RSS', capError:cap?.error || null };
}

function pulseSocialRelevance(text, tags = []) {
    const raw = String(text || '');
    const norm = pulseNormalizeText(raw);
    if (!pulseDetectHazards(raw).length) return false;

    const indiaTerms = ['india', 'bharat', 'ndma', 'asdma', ...PULSE_STATE_NAMES, ...PULSE_CITY_NAMES];
    const indiaContext = indiaTerms.some(term => term && norm.includes(pulseNormalizeText(term)));
    const tagList = Array.isArray(tags) ? tags.map(tag => String(tag || '').toLowerCase()) : [];
    const strongIndiaHashtag = tagList.includes('#imd') || tagList.includes('#monsoonindia');
    const explicitForeign = /(\bunited states\b|\busa\b|\bu\.?s\.?\b|\bcanada\b|\bunited kingdom\b|\buk\b|\bpakistan\b|\bbangladesh\b|\bindonesia\b|\bphilippines\b|\baustralia\b|\beurope\b)/i.test(raw);
    if (explicitForeign && !indiaContext) return false;
    return indiaContext || strongIndiaHashtag;
}

async function fetchPulseMastodon() {
    if (PULSE_SOCIAL_CACHE.data && Date.now() - PULSE_SOCIAL_CACHE.timestamp < PULSE_SOCIAL_CACHE_MS) return PULSE_SOCIAL_CACHE.data;
    const configured = String(process.env.MASTODON_INSTANCE_URL || '').trim().replace(/\/$/, '');
    const instances = [...new Set([
        configured,
        'https://mastodon.social',
        'https://mastodon.online',
        'https://fosstodon.org',
        'https://mstdn.social'
    ].filter(Boolean))];
    const tags = ['IMD','WeatherAlert','MonsoonIndia','FloodAlert','Thunderstorm'];
    const errors = [];
    for (const instance of instances) {
        const posts = [];
        let successfulTag = 0;
        let filteredOut = 0;
        for (const tag of tags) {
            try {
                const url = `${instance}/api/v1/timelines/tag/${encodeURIComponent(tag)}?limit=30&remote=true`;
                const response = await fetch(url, {
                    headers: { 'User-Agent': 'RakshaNet-Pulse/2.0 (+https://rakshanet.local)', 'Accept': 'application/json' },
                    signal: AbortSignal.timeout(9000)
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                successfulTag++;
                const data = await response.json();
                for (const item of Array.isArray(data) ? data : []) {
                    const parser = String(item?.content || '').replace(/<[^>]+>/g, ' ').replace(/&nbsp;/gi, ' ').replace(/&amp;/gi, '&').replace(/\s+/g,' ').trim();
                    const author = item?.account?.acct || item?.account?.username || 'unknown';
                    const media = Array.isArray(item?.media_attachments) ? item.media_attachments : [];
                    const tagNames = Array.isArray(item?.tags) ? item.tags.map(t => `#${t?.name || ''}`) : [];
                    const mergedText = [parser, ...tagNames].join(' ').trim();
                    if (!pulseSocialRelevance(mergedText, tagNames)) { filteredOut++; continue; }
                    posts.push({
                        uri: item?.uri || item?.url || '', author, text: mergedText,
                        createdAt: item?.created_at || null, indexedAt: item?.created_at || null,
                        sourceUrl: item?.url || null,
                        mediaCount: media.length,
                        media: media.map(m => ({ url: m?.url || m?.preview_url || null, type: m?.type || 'image' })).filter(m => m.url),
                        tags: tagNames
                    });
                }
            } catch (error) { errors.push(`${instance} #${tag}: ${error.message}`); }
        }
        const unique = new Map();
        for (const post of posts) {
            const key = post.uri || `${post.author}|${post.createdAt}|${post.text}`;
            if (!unique.has(key)) unique.set(key, post);
        }
        if (successfulTag > 0) {
            const result = {
                available: true,
                source: 'Mastodon public hashtag stream',
                sourceName: `Mastodon public hashtags · ${instance.replace(/^https?:\/\//,'')}`,
                sourceUrl: `${instance}/api/v1/timelines/tag/IMD`,
                provider: 'Mastodon', queries: tags.map(t => `#${t}`), posts: Array.from(unique.values()).slice(0,100), filteredOut, checkedAt: pulseNowIso(),
                error: errors.length ? errors.slice(-8).join(' | ') : null
            };
            PULSE_SOCIAL_CACHE.timestamp = Date.now();
            PULSE_SOCIAL_CACHE.data = result;
            if (result.posts.length) return result;
            // A successful endpoint with zero relevant posts is real data state, but keep trying another instance.
        }
    }
    // Mastodon documents RSS syndication for hashtag pages; try it when API timelines are blocked.
    for (const instance of instances) {
        const posts = [];
        let successfulTag = 0;
        let filteredOut = 0;
        for (const tag of tags) {
            try {
                const rssUrl = `${instance}/tags/${encodeURIComponent(tag)}.rss`;
                const response = await fetch(rssUrl, { headers:{'User-Agent':'RakshaNet-Pulse/2.0','Accept':'application/rss+xml,application/xml,text/xml'}, signal:AbortSignal.timeout(8000) });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                successfulTag++;
                const xml = await response.text();
                const items = parseImdCapRss(xml);
                for (const item of items) {
                    const combined = `${item.title || ''} ${item.description || ''} #${tag}`;
                    if (!pulseSocialRelevance(combined, [`#${tag}`])) { filteredOut++; continue; }
                    posts.push({uri:item.link||'',author:item.source||'Mastodon public user',text:combined,createdAt:item.pubDate||null,indexedAt:item.pubDate||null,sourceUrl:item.link||null,mediaCount:0,media:[],tags:[`#${tag}`]});
                }
            } catch (error) { errors.push(`${instance} RSS #${tag}: ${error.message}`); }
        }
        const unique = new Map();
        for (const post of posts) { const key=post.uri || `${post.author}|${post.createdAt}|${post.text}`; if (!unique.has(key)) unique.set(key,post); }
        if (successfulTag > 0) {
            const result={available:true,source:'Mastodon public hashtag RSS',sourceName:`Mastodon public hashtag RSS · ${instance.replace(/^https?:\/\//,'')}`,sourceUrl:`${instance}/tags/IMD.rss`,provider:'Mastodon RSS',queries:tags.map(t=>`#${t}`),posts:Array.from(unique.values()).slice(0,100),filteredOut,checkedAt:pulseNowIso(),error:errors.length?errors.slice(-8).join(' | '):null};
            PULSE_SOCIAL_CACHE.timestamp=Date.now(); PULSE_SOCIAL_CACHE.data=result;
            if (result.posts.length) return result;
        }
    }
    const result = {
        available: false,
        source: 'Public social feeds',
        sourceName: 'Bluesky / Mastodon public hashtags',
        sourceUrl: 'https://docs.bsky.app/', provider: null, queries: tags.map(t => `#${t}`), posts: [], checkedAt: pulseNowIso(),
        error: errors.slice(-12).join(' | ') || 'No public India-relevant hashtag posts returned.'
    };
    PULSE_SOCIAL_CACHE.timestamp = Date.now();
    PULSE_SOCIAL_CACHE.data = result;
    return result;
}

async function fetchPulseBluesky() {
    const queries = ['#IMD','#WeatherAlert','#MonsoonIndia','#FloodAlert','#Thunderstorm'];
    const posts = [], errors = [];
    let filteredOut = 0;
    for (const query of queries) {
        try {
            const url = new URL('https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts');
            url.search = new URLSearchParams({ q: query, limit: '25', sort: 'latest' });
            const response = await fetch(url, {
                headers: { 'User-Agent': 'RakshaNet-Pulse/2.0', 'Accept': 'application/json' },
                signal: AbortSignal.timeout(12000)
            });
            if (!response.ok) throw new Error(`Bluesky HTTP ${response.status}`);
            const data = await response.json();
            for (const item of Array.isArray(data?.posts) ? data.posts : []) {
                const author = item?.author?.handle || item?.author?.displayName || 'unknown';
                const uri = item?.uri || '';
                const text = String(item?.record?.text || '');
                const rkey = pulseExtractPostRkey(uri);
                const media = Array.isArray(item?.embed?.images) ? item.embed.images.map(img => ({ url: img?.fullsize || img?.thumb || null, type: 'image' })).filter(m => m.url) : [];
                const queryTags = [query];
                if (!pulseSocialRelevance(text, queryTags)) { filteredOut++; continue; }
                posts.push({ uri, author, text, createdAt: item?.record?.createdAt || item?.indexedAt || null, indexedAt: item?.indexedAt || null,
                    sourceUrl: rkey ? `https://bsky.app/profile/${author}/post/${rkey}` : null, mediaCount: media.length, media });
            }
        } catch (error) { errors.push(`${query}: ${error.message}`); }
    }
    const unique = new Map();
    for (const post of posts) { const key = post.uri || `${post.author}|${post.createdAt}|${post.text}`; if (!unique.has(key)) unique.set(key, post); }
    if (unique.size) {
        return { available:true, source:'Bluesky public social search', sourceName:'Bluesky public posts', sourceUrl:'https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts', provider:'Bluesky', queries, posts:Array.from(unique.values()).slice(0,100), filteredOut, checkedAt:pulseNowIso(), error: errors.length ? errors.join(' | ') : null };
    }
    // Public Bluesky reads are documented as unauthenticated, but deployments can still block or rate-limit this endpoint. Fall back to another public social network rather than fabricating data.
    const mastodon = await fetchPulseMastodon();
    if (mastodon.available) return mastodon;
    return { available:false, source:'Public social feeds', sourceName:'Bluesky / Mastodon public hashtags', sourceUrl:'https://docs.bsky.app/', provider:null, queries, posts:[], checkedAt:pulseNowIso(), error:[...errors,...(mastodon.error?[mastodon.error]:[])].join(' | ') || 'No public social signal returned.' };
}

function decodePulseXml(value) {
    return String(value || '').replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
        .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
        .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n,16)))
        .replace(/&amp;/g,'&').replace(/&quot;/g,'"').replace(/&apos;/g,"'").replace(/&lt;/g,'<').replace(/&gt;/g,'>');
}
function parsePulseGoogleNewsRss(xml) {
    const items = [];
    const blocks = String(xml || '').match(/<item>[\s\S]*?<\/item>/gi) || [];
    for (const block of blocks) {
        const tag = name => {
            const match = block.match(new RegExp(`<${name}>([\\s\\S]*?)<\\/${name}>`, 'i'));
            return match ? decodePulseXml(match[1]).trim() : '';
        };
        const title = tag('title');
        const link = tag('link');
        const pubDate = tag('pubDate');
        const source = tag('source');
        const description = tag('description').replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').trim();
        if (title || link) items.push({ title, url: link || null, seendate: pubDate || null, domain: source || 'Google News', sourcecountry: 'IN', language: 'en', snippet: description });
    }
    return items;
}

async function fetchGoogleNewsPulseRss(query) {
    const url = new URL('https://news.google.com/rss/search');
    url.search = new URLSearchParams({ q: query, hl:'en-IN', gl:'IN', ceid:'IN:en' });
    const response = await fetch(url, { headers:{'User-Agent':'RakshaNet-Pulse/2.0','Accept':'application/rss+xml,application/xml,text/xml'}, signal:AbortSignal.timeout(12000) });
    if (!response.ok) throw new Error(`Google News RSS HTTP ${response.status}`);
    const xml = await response.text();
    return parsePulseGoogleNewsRss(xml);
}

async function fetchPulseGdelt() {
    const query = '(India weather OR India rainfall OR India thunderstorm OR India flood OR India heatwave OR India fog OR India "dust storm")';
    try {
        const url = new URL('https://api.gdeltproject.org/api/v2/doc/doc');
        url.search = new URLSearchParams({ query, mode:'artlist', maxrecords:'50', timespan:'24h', sort:'datedesc', format:'json' });
        const response = await fetch(url, { headers:{'User-Agent':'RakshaNet-Pulse/2.0','Accept':'application/json'}, signal:AbortSignal.timeout(15000) });
        if (!response.ok) throw new Error(`GDELT HTTP ${response.status}`);
        const data = await response.json();
        const articles = Array.isArray(data?.articles) ? data.articles : [];
        if (articles.length) return { available:true, source:'GDELT public web/news index', sourceName:'GDELT public web', sourceUrl:'https://api.gdeltproject.org/api/v2/doc/doc', provider:'GDELT', query, articles, checkedAt:pulseNowIso(), error:null };
        throw new Error('GDELT returned no matching articles.');
    } catch (gdeltError) {
        try {
            const rssQueries = ['India weather IMD','India rainfall flood','India thunderstorm heatwave fog'];
            const merged = [];
            for (const q of rssQueries) merged.push(...await fetchGoogleNewsPulseRss(q));
            const unique = new Map();
            for (const item of merged) if (item.url && !unique.has(item.url)) unique.set(item.url,item);
            const articles = Array.from(unique.values()).slice(0,60);
            if (articles.length) return { available:true, source:'Google News public RSS', sourceName:'Google News public RSS', sourceUrl:'https://news.google.com/rss', provider:'Google News RSS', query:rssQueries.join(' | '), articles, checkedAt:pulseNowIso(), error:`GDELT unavailable: ${gdeltError.message}` };
            throw new Error('Google News RSS returned no matching articles.');
        } catch (fallbackError) {
            return { available:false, source:'Public web/news feeds', sourceName:'GDELT / Google News RSS', sourceUrl:'https://news.google.com/rss', provider:null, query, articles:[], checkedAt:pulseNowIso(), error:`GDELT: ${gdeltError.message} | RSS: ${fallbackError.message}` };
        }
    }
}

function pulseWeatherCodeLabel(code) {
    const c = Number(code);
    const labels = {
        0:'Clear sky',1:'Mainly clear',2:'Partly cloudy',3:'Overcast',45:'Fog',48:'Depositing rime fog',
        51:'Light drizzle',53:'Moderate drizzle',55:'Dense drizzle',56:'Freezing drizzle',57:'Freezing drizzle',
        61:'Slight rain',63:'Moderate rain',65:'Heavy rain',66:'Freezing rain',67:'Freezing rain',71:'Slight snow',73:'Moderate snow',75:'Heavy snow',
        77:'Snow grains',80:'Slight rain showers',81:'Moderate rain showers',82:'Violent rain showers',85:'Slight snow showers',86:'Heavy snow showers',
        95:'Thunderstorm',96:'Thunderstorm with slight hail',99:'Thunderstorm with heavy hail'
    };
    return labels[c] || `Weather code ${c}`;
}

async function fetchPulseOpenMeteo() {
    if (PULSE_WEATHER_CACHE.data && Date.now() - PULSE_WEATHER_CACHE.timestamp < PULSE_WEATHER_CACHE_MS) return PULSE_WEATHER_CACHE.data;
    try {
        const geocoded = [];
        for (const place of PULSE_WEATHER_LOCATIONS) {
            const result = await pulseGeocode(`${place.city}, ${place.state}, India`, place.state, null);
            if (result && pulseValidCoordinate(result.latitude, result.longitude)) {
                geocoded.push({ ...place, latitude: result.latitude, longitude: result.longitude, geocoderName: result.name });
            }
        }
        if (!geocoded.length) throw new Error('No national reference locations could be geocoded.');
        const url = new URL('https://api.open-meteo.com/v1/forecast');
        url.search = new URLSearchParams({
            latitude: geocoded.map(item => item.latitude).join(','),
            longitude: geocoded.map(item => item.longitude).join(','),
            current: 'temperature_2m,relative_humidity_2m,precipitation,rain,showers,snowfall,weather_code,wind_speed_10m,wind_direction_10m,visibility',
            timezone: 'auto'
        });
        const response = await fetch(url, { headers:{ 'User-Agent':'RakshaNet-Pulse/2.0 (+https://rakshanet.local)', 'Accept':'application/json' }, signal:AbortSignal.timeout(15000) });
        if (!response.ok) throw new Error(`Open-Meteo HTTP ${response.status}`);
        const payload = await response.json();
        const rows = Array.isArray(payload) ? payload : [payload];
        const observations = rows.map((row,index) => {
            const place = geocoded[index] || geocoded[0];
            const current = row?.current || {};
            return {
                ...place,
                time: current.time || row?.current?.time || null,
                temperature: Number.isFinite(Number(current.temperature_2m)) ? Number(current.temperature_2m) : null,
                humidity: Number.isFinite(Number(current.relative_humidity_2m)) ? Number(current.relative_humidity_2m) : null,
                precipitation: Number.isFinite(Number(current.precipitation)) ? Number(current.precipitation) : null,
                rain: Number.isFinite(Number(current.rain)) ? Number(current.rain) : null,
                showers: Number.isFinite(Number(current.showers)) ? Number(current.showers) : null,
                snowfall: Number.isFinite(Number(current.snowfall)) ? Number(current.snowfall) : null,
                weatherCode: Number.isFinite(Number(current.weather_code)) ? Number(current.weather_code) : null,
                windSpeed: Number.isFinite(Number(current.wind_speed_10m)) ? Number(current.wind_speed_10m) : null,
                windDirection: Number.isFinite(Number(current.wind_direction_10m)) ? Number(current.wind_direction_10m) : null,
                visibility: Number.isFinite(Number(current.visibility)) ? Number(current.visibility) : null
            };
        }).filter(item => item.city && item.time);
        const result = {
            available: observations.length > 0,
            source: 'Open-Meteo public weather API',
            sourceName: 'Open-Meteo public API observations',
            sourceUrl: 'https://open-meteo.com/',
            provider: 'Open-Meteo',
            observations,
            checkedAt: pulseNowIso(),
            error: null
        };
        PULSE_WEATHER_CACHE.timestamp = Date.now();
        PULSE_WEATHER_CACHE.data = result;
        return result;
    } catch (error) {
        const result = { available:false, source:'Open-Meteo public weather API', sourceName:'Open-Meteo public API observations', sourceUrl:'https://open-meteo.com/', provider:'Open-Meteo', observations:[], checkedAt:pulseNowIso(), error:error.message };
        PULSE_WEATHER_CACHE.timestamp = Date.now();
        PULSE_WEATHER_CACHE.data = result;
        return result;
    }
}

function pulseWeatherObservationCategory(observation) {
    const c = Number(observation?.weatherCode);
    if ([95,96,99].includes(c)) return 'thunderstorm';
    if ([45,48].includes(c)) return 'fog';
    if ([51,53,55,56,57,61,63,65,66,67,71,73,75,77,80,81,82,85,86].includes(c)) return observation?.snowfall > 0 ? 'other' : 'rainfall';
    if (Number(observation?.windSpeed) >= 40) return 'strong_winds';
    return 'other';
}

function pulseToWeatherApiEvents(feed) {
    const events = [];
    for (const observation of Array.isArray(feed?.observations) ? feed.observations : []) {
        const category = pulseWeatherObservationCategory(observation);
        const title = `Open-Meteo observation — ${observation.city}, ${observation.state}`;
        const readings = [
            `Temperature ${observation.temperature ?? 'not available'} °C`,
            `Humidity ${observation.humidity ?? 'not available'} %`,
            `Precipitation ${observation.precipitation ?? 'not available'} mm`,
            `Wind ${observation.windSpeed ?? 'not available'} km/h`,
            `Weather ${pulseWeatherCodeLabel(observation.weatherCode)}`
        ];
        const content = readings.join(' | ');
        const sourceId = `openmeteo:${pulseNormalizeText(observation.city)}:${observation.time}`;
        const event = {
            id:pulseMakeId('OPENMETEO',sourceId), fingerprint:null,
            sourceType:'weather_api', sourceName:'Open-Meteo public API', sourceUrl:'https://open-meteo.com/', sourceId,
            title, content, author:'Open-Meteo', publishedAt:observation.time, ingestedAt:pulseNowIso(),
            city:observation.city, district:null, state:observation.state, latitude:Number(observation.latitude), longitude:Number(observation.longitude),
            eventCategory:category, verificationStatus:'SOURCE_VERIFIED', trustScore:90, duplicateOf:null, media:[],
            metadata:{ observationType:'current_weather', provider:'Open-Meteo', locationBasis:'DERIVED_GEOCODE', locationSource:'Open-Meteo Geocoding API',
                temperature:observation.temperature, humidity:observation.humidity, precipitation:observation.precipitation, rain:observation.rain, showers:observation.showers,
                snowfall:observation.snowfall, weatherCode:observation.weatherCode, weatherDescription:pulseWeatherCodeLabel(observation.weatherCode),
                windSpeed:observation.windSpeed, windDirection:observation.windDirection, visibility:observation.visibility,
                hazards:pulseDetectHazards(content), mentionedPlaces:[observation.city], locationScope:'SINGLE_AREA'
            }
        };
        pulseApplyIntegrityScreen(event);
        events.push(event);
    }
    return events;
}

function pulseToNwdpEvents(sensorData) {
    const events = [];
    const feeds = sensorData?.feeds || {};
    for (const [key, meta] of Object.entries(PULSE_NWDP_FEED_META)) {
        const feed = feeds[key];
        const latest = feed?.latest;
        if (!latest || latest.station === 'Unknown station') continue;
        let value = null;
        if (key === 'riverLevel') value = latest.waterLevel;
        if (key === 'rainfall') value = latest.rainfall;
        if (key === 'temperature') value = latest.temperature;
        if (key === 'humidity') value = latest.humidity;
        if (key === 'windSpeed') value = latest.windSpeed;
        if (key === 'windDirection') value = latest.windDirection;
        if (key === 'pressure') value = latest.pressure;
        if (!Number.isFinite(Number(value))) continue;
        const category = key === 'rainfall' ? 'rainfall' : (key === 'windSpeed' && Number(value) >= 40 ? 'strong_winds' : 'other');
        const rawTimestamp = latest.timestamp || sensorData.updatedAt || null;
        const parsedTimestamp = parseObservationDate(rawTimestamp);
        const publishedAt = parsedTimestamp ? parsedTimestamp.toISOString() : (rawTimestamp ? String(rawTimestamp) : null);
        const sourceId = `nwdp:${key}:${latest.station}:${rawTimestamp || sensorData.updatedAt}`;
        const event = {
            id:pulseMakeId('NWDP',sourceId), fingerprint:null, sourceType:'public_dataset', sourceName:'National Water Data Portal / NWIC telemetry', sourceUrl:meta.url, sourceId,
            title:`NWDP telemetry — ${meta.label} at ${latest.station}`, content:`${meta.label}: ${value} ${meta.unit}${latest.river ? ` | River: ${latest.river}` : ''}${latest.district ? ` | District: ${latest.district}` : ''}`,
            author:latest.agency || 'NWIC/NWDP', publishedAt, ingestedAt:pulseNowIso(),
            city:latest.station || null, district:latest.district || null, state:latest.state || 'Assam', latitude:pulseValidCoordinate(latest.latitude, latest.longitude)?Number(latest.latitude):null, longitude:pulseValidCoordinate(latest.latitude, latest.longitude)?Number(latest.longitude):null,
            eventCategory:category, verificationStatus:'SOURCE_VERIFIED', trustScore:91, duplicateOf:null, media:[], metadata:{ observationType:key, unit:meta.unit, value:Number(value), station:latest.station, river:latest.river || null, rawObservationTimestamp:rawTimestamp, locationBasis:pulseValidCoordinate(latest.latitude,latest.longitude)?'SOURCE_GEOMETRY':'TEXT_MENTION', sourceCoordinates:pulseValidCoordinate(latest.latitude,latest.longitude), hazards:pulseDetectHazards(`${meta.label} ${value} ${latest.river || ''}`) }
        };
        pulseApplyIntegrityScreen(event);
        events.push(event);
    }
    return events;
}

function pulseReadLocalCitizenReports() {
    const file = path.join(ROOT, 'data', 'reports.json');
    try {
        const value = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file,'utf8')) : [];
        return Array.isArray(value) ? value : [];
    } catch (_) { return []; }
}

async function pulseFetchCitizenReports() {
    if (DATABASE_CONFIGURED) {
        try {
            const result = await pool.query(`
                SELECT id, reporter_name AS "reporterName", contact, incident_type AS "incidentType", latitude, longitude,
                       people_affected AS "peopleAffected", injured, trapped, description, priority, timestamp, status, synced, evidence
                FROM public.emergency_reports ORDER BY timestamp DESC LIMIT 100
            `);
            return { available:true, reports:result.rows, source:'RakshaNet citizen emergency reports', storage:'PostgreSQL', checkedAt:pulseNowIso(), error:null };
        } catch (error) {
            const local = pulseReadLocalCitizenReports();
            return { available:local.length>0, reports:local, source:'RakshaNet citizen emergency reports', storage:'LOCAL_FALLBACK', checkedAt:pulseNowIso(), error:'PostgreSQL unavailable; local JSON reports are active.' };
        }
    }
    const local = pulseReadLocalCitizenReports();
    return { available:local.length>0, reports:local, source:'RakshaNet citizen emergency reports', storage:'LOCAL_FALLBACK', checkedAt:pulseNowIso(), error:null };
}

function pulseToSachetEvents(feed) {
    const events = [];
    for (const alert of Array.isArray(feed?.alerts) ? feed.alerts.slice(0, 100) : []) {
        const loc = pulseFindLocation(alert.areaDesc, {});
        const circle = alert.circle || null;
        const centroid = Array.isArray(alert.polygon) && alert.polygon.length
            ? alert.polygon.reduce((acc, point) => ({ lat: acc.lat + Number(point[0]), lon: acc.lon + Number(point[1]) }), { lat: 0, lon: 0 })
            : null;
        const lat = circle?.latitude ?? (centroid ? centroid.lat / alert.polygon.length : null);
        const lon = circle?.longitude ?? (centroid ? centroid.lon / alert.polygon.length : null);
        const title = alert.title || 'NDMA/SACHET alert';
        const content = [alert.description, alert.areaDesc, alert.category, alert.severity].filter(Boolean).join(' | ');
        const trust = pulseBuildTrust({
            sourceType: 'official', title, content,
            city: loc.city, state: loc.state,
            latitude: lat, longitude: lon,
            mediaCount: 0
        });
        events.push({
            id: pulseMakeId('SACHET', alert.guid || alert.link || title),
            sourceType: 'official',
            sourceName: 'NDMA SACHET India CAP/RSS',
            sourceUrl: alert.link || feed.url,
            sourceId: alert.guid || alert.link || title,
            title,
            content,
            author: 'NDMA SACHET',
            publishedAt: alert.publishedAt || alert.pubDate || null,
            ingestedAt: pulseNowIso(),
            city: loc.city,
            district: loc.district || null,
            state: loc.state || null,
            latitude: pulseValidCoordinate(lat, lon) ? Number(lat) : null,
            longitude: pulseValidCoordinate(lat, lon) ? Number(lon) : null,
            eventCategory: pulseEventCategory(`${title} ${content}`, 'other', alert.hazard),
            verificationStatus: 'VERIFIED_OFFICIAL',
            trustScore: Math.max(96, trust),
            duplicateOf: null,
            media: [],
            metadata: {
                severity: alert.severity || null,
                locationBasis: pulseValidCoordinate(lat, lon) ? 'SOURCE_GEOMETRY' : (loc.locationBasis || null),
                sourceCoordinates: pulseValidCoordinate(lat, lon),
                district: loc.district || null,
                mentionedPlaces: loc.mentionedPlaces || [],
                locationScope: loc.locationScope || 'SINGLE_AREA',
                urgency: alert.urgency || null,
                certainty: alert.certainty || null,
                expiresAt: alert.expiresAt || alert.expires || null,
                polygon: alert.polygon || [],
                circle: circle || null,
                hazards: pulseDetectHazards(`${title} ${content}`)

            }
        });
    }
    return events;
}


function pulseToImdCapEvents(feed) {
    const events = [];
    for (const alert of Array.isArray(feed?.rows) ? feed.rows : []) {
        const content = [alert.description, alert.areaDesc, alert.hazard, alert.severity, alert.urgency, alert.certainty].filter(Boolean).join(' | ');
        const loc = pulseFindLocation(alert.areaDesc || content, {});
        const validPolygon = Array.isArray(alert.polygon) && alert.polygon.length >= 3 && alert.polygon.every(pt => Array.isArray(pt) && pt.length === 2 && pt.every(Number.isFinite));
        const centroid = validPolygon ? alert.polygon.reduce((acc, point) => ({ lat:acc.lat + point[0], lon:acc.lon + point[1] }), {lat:0,lon:0}) : null;
        const circle = pulseValidCoordinate(alert.circle?.latitude, alert.circle?.longitude) ? alert.circle : null;
        const lat = circle ? Number(circle.latitude) : (centroid ? centroid.lat / alert.polygon.length : null);
        const lon = circle ? Number(circle.longitude) : (centroid ? centroid.lon / alert.polygon.length : null);
        const trust = pulseBuildTrust({sourceType:'official',title:alert.title,content,city:loc.city,state:loc.state,latitude:lat,longitude:lon});
        events.push({
            id:pulseMakeId('IMDCAP',alert.guid || alert.link || alert.title), sourceType:'official', sourceName:'IMD CAP / WMO Alert Hub RSS', sourceUrl:alert.link || feed.sourceUrl, sourceId:alert.guid || alert.link || alert.title,
            title:alert.title || 'IMD CAP alert', content, author:'India Meteorological Department', publishedAt:alert.publishedAt || null, ingestedAt:pulseNowIso(), city:loc.city || null, district:loc.district || null, state:loc.state || null,
            latitude:pulseValidCoordinate(lat,lon)?lat:null, longitude:pulseValidCoordinate(lat,lon)?lon:null,
            eventCategory:pulseEventCategory(`${alert.title || ''} ${content}`, 'other', alert.hazard), verificationStatus:'VERIFIED_OFFICIAL', trustScore:Math.max(96,trust), duplicateOf:null, media:[],
            metadata:{severity:alert.severity||null,urgency:alert.urgency||null,certainty:alert.certainty||null,expiresAt:alert.expiresAt||null,locationBasis:pulseValidCoordinate(lat,lon)?'SOURCE_GEOMETRY':(loc.locationBasis||null),sourceCoordinates:pulseValidCoordinate(lat,lon),polygon:validPolygon?alert.polygon:[],circle:circle||null,mentionedPlaces:loc.mentionedPlaces||[],locationScope:loc.locationScope||'SINGLE_AREA',hazards:pulseDetectHazards(`${alert.title || ''} ${content}`),provider:'IMD CAP RSS'}
        });
    }
    return events.slice(0,120);
}

function pulseToImdEvents(feed) {
    const events = [];
    for (const row of Array.isArray(feed?.rows) ? feed.rows : []) {
        const district = row?.District || row?.district || null;
        const state = row?.State || row?.state || null;
        const day1 = row?.Day_1 ?? row?.day1_warning ?? '';
        const day2 = row?.Day_2 ?? row?.day2_warning ?? '';
        const day3 = row?.Day_3 ?? row?.day3_warning ?? '';
        const day4 = row?.Day_4 ?? row?.day4_warning ?? '';
        const day5 = row?.Day_5 ?? row?.day5_warning ?? '';
        const content = `Day 1: ${day1} | Day 2: ${day2} | Day 3: ${day3} | Day 4: ${day4} | Day 5: ${day5}`;
        const warningValues = [day1, day2, day3, day4, day5].map(value => String(value ?? '').trim()).filter(Boolean);
        if (!warningValues.length) continue;
        const analyzed = analyzeImdWarning(row);
        const title = `IMD district warning — ${district || 'district'}`;
        const trust = pulseBuildTrust({ sourceType: 'official', title, content, city: district, state, latitude: null, longitude: null });
        const rawDate = row?.Date || row?.date_obs || null;
        const dateObj = rawDate ? new Date(rawDate) : null;
        const publishedAt = dateObj && Number.isFinite(dateObj.getTime()) ? dateObj.toISOString() : null;
        events.push({
            id: pulseMakeId('IMD', row?.Obj_id || `${district}|${row?.Date || ''}`),
            sourceType: 'official',
            sourceName: 'India Meteorological Department district warnings',
            sourceUrl: 'https://api.imd.gov.in/api/v1/districtwarning',
            sourceId: String(row?.Obj_id || `${district}|${row?.Date || ''}`),
            title,
            content: `${content}${analyzed?.hazards?.length ? ` | Hazards: ${analyzed.hazards.join(', ')}` : ''}`,
            author: 'IMD',
            publishedAt,
            ingestedAt: pulseNowIso(),
            city: null,
            district: district,
            state,
            latitude: null,
            longitude: null,
            eventCategory: pulseEventCategory(content, analyzed?.hazards?.[0] || 'other'),
            verificationStatus: 'VERIFIED_OFFICIAL',
            trustScore: Math.max(96, trust),
            duplicateOf: null,
            media: [],
            metadata: { warningCodes: analyzed?.codes || [], hazards: pulseDetectHazards(content), mentionedPlaces: pulseFindLocation(content, { district, state }).mentionedPlaces, locationScope: pulseFindLocation(content, { district, state }).locationScope, colors: [row?.Day1_Color, row?.Day2_Color, row?.Day3_Color, row?.Day4_Color, row?.Day5_Color].filter(Boolean) }
        });
    }
    return events.slice(0, 120);
}

function pulseToSocialEvents(feed) {
    const events = [];
    const sourceName = feed?.sourceName || (feed?.provider === 'Mastodon' ? 'Mastodon public hashtags' : 'Bluesky public posts');
    for (const post of Array.isArray(feed?.posts) ? feed.posts : []) {
        const loc = pulseFindLocation(post.text, {});
        const title = `Social signal by @${post.author}`;
        const content = post.text;
        const trust = pulseBuildTrust({ sourceType:'social', title, content, city:loc.city, state:loc.state, mediaCount:Number(post.mediaCount || 0) });
        events.push({
            id:pulseMakeId('SOCIAL',post.uri || `${post.author}|${post.createdAt}|${post.text}`), sourceType:'social', sourceName,
            sourceUrl:post.sourceUrl || feed.sourceUrl || null, sourceId:post.uri || null, title, content, author:post.author,
            publishedAt:post.createdAt || post.indexedAt || null, ingestedAt:pulseNowIso(), city:loc.city, district:loc.district || null, state:loc.state,
            latitude:null, longitude:null, eventCategory:pulseEventCategory(content), verificationStatus:pulseInitialVerification('social',trust), trustScore:trust,
            duplicateOf:null, media:Array.isArray(post.media) ? post.media : (Number(post.mediaCount||0) ? [{count:Number(post.mediaCount||0),type:'image'}] : []),
            metadata:{ hazards:pulseDetectHazards(content), mentionedPlaces:loc.mentionedPlaces || [], locationScope:loc.locationScope || 'SINGLE_AREA',
                hashtags:['#IMD','#WeatherAlert','#MonsoonIndia','#FloodAlert','#Thunderstorm'].filter(tag=>content.toLowerCase().includes(tag.toLowerCase())), provider:feed?.provider || null }
        });
    }
    return events;
}

function pulseToWebEvents(feed) {
    const events = [];
    for (const article of Array.isArray(feed?.articles) ? feed.articles : []) {
        const title = String(article?.title || '').trim();
        if (!title) continue;
        const content = [title, article?.snippet].filter(Boolean).join(' — ');
        const loc = pulseFindLocation(content, {});
        const domain = article?.domain || article?.sourcecountry || 'public web';
        const trust = pulseBuildTrust({ sourceType:'web', title, content, city:loc.city, state:loc.state, mediaCount:article?.socialimage ? 1 : 0 });
        events.push({
            id:pulseMakeId('WEB',article?.url || `${domain}|${title}`), sourceType:'web', sourceName:`${feed?.sourceName || 'Public web/news'} · ${domain}`,
            sourceUrl:article?.url || feed?.sourceUrl || null, sourceId:article?.url || null, title, content, author:domain,
            publishedAt:pulseParseGdeltDate(article?.seendate || article?.pubDate || article?.publishedAt), ingestedAt:pulseNowIso(),
            city:loc.city, district:loc.district || null, state:loc.state, latitude:null, longitude:null, eventCategory:pulseEventCategory(content),
            verificationStatus:pulseInitialVerification('web',trust), trustScore:trust, duplicateOf:null,
            media:article?.socialimage ? [{url:article.socialimage,type:'image'}] : [],
            metadata:{hazards:pulseDetectHazards(content),mentionedPlaces:loc.mentionedPlaces || [],locationScope:loc.locationScope || 'SINGLE_AREA',domain,language:article?.language || null,sourceCountry:article?.sourcecountry || null,provider:feed?.provider || null}
        });
    }
    return events;
}

function pulseParseGdeltDate(value) {
    const raw = String(value || '');
    if (/^\d{14}$/.test(raw)) {
        const year = Number(raw.slice(0, 4));
        const month = Number(raw.slice(4, 6)) - 1;
        const day = Number(raw.slice(6, 8));
        const hour = Number(raw.slice(8, 10));
        const minute = Number(raw.slice(10, 12));
        const second = Number(raw.slice(12, 14));
        return new Date(Date.UTC(year, month, day, hour, minute, second)).toISOString();
    }
    const date = new Date(raw);
    return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

function pulseToCitizenEvents(bundle) {
    const events = [];
    for (const report of Array.isArray(bundle?.reports) ? bundle.reports : []) {
        const content = String(report?.description || report?.incidentType || 'Citizen weather/disaster report');
        const title = `Citizen report — ${report?.incidentType || 'Weather event'}`;
        const cityState = pulseFindLocation(content, {});
        const latRaw = Number(report?.latitude);
        const lonRaw = Number(report?.longitude);
        const lat = pulseValidCoordinate(latRaw, lonRaw) ? latRaw : null;
        const lon = pulseValidCoordinate(latRaw, lonRaw) ? lonRaw : null;
        const trust = pulseBuildTrust({
            sourceType: 'citizen', title, content,
            city: cityState.city, state: cityState.state,
            latitude: lat, longitude: lon,
            mediaCount: report?.evidence ? 1 : 0
        });
        events.push({
            id: pulseMakeId('CITIZEN', report?.id || `${report?.timestamp}|${content}`),
            sourceType: 'citizen',
            sourceName: 'RakshaNet citizen emergency reports',
            sourceUrl: '/pages/emergency.html',
            sourceId: String(report?.id || ''),
            title,
            content,
            author: report?.reporterName || 'Anonymous citizen',
            publishedAt: report?.timestamp || null,
            ingestedAt: pulseNowIso(),
            city: cityState.city,
            district: cityState.district || null,
            state: cityState.state,
            latitude: Number.isFinite(lat) ? lat : null,
            longitude: Number.isFinite(lon) ? lon : null,
            eventCategory: pulseEventCategory(`${report?.incidentType || ''} ${content}`, String(report?.incidentType || '').toLowerCase() || 'other'),
            verificationStatus: 'UNVERIFIED_CITIZEN',
            trustScore: trust,
            duplicateOf: null,
            media: report?.evidence ? [{ url: report.evidence.url || null, type: report.evidence.type || 'evidence' }] : [],
            metadata: { status: report?.status || null, mentionedPlaces: cityState.mentionedPlaces || [], locationScope: cityState.locationScope || 'SINGLE_AREA', peopleAffected: Number(report?.peopleAffected || 0), injured: Number(report?.injured || 0), trapped: Number(report?.trapped || 0) }
        });
    }
    return events;
}

function pulseHasPossibleDuplicate(event, existing) {
    const targetTime = new Date(event.publishedAt || event.ingestedAt || pulseNowIso()).getTime();
    for (const candidate of existing) {
        if (!candidate) continue;
        if (candidate.sourceType === event.sourceType && candidate.sourceId && event.sourceId && candidate.sourceId === event.sourceId) return candidate;
        // Fuzzy matching is intentionally conservative and stays within the same source class.
        // A new timestamped observation from an official feed is not a duplicate just because the text is similar.
        if (candidate.sourceType !== event.sourceType) continue;
        if (candidate.sourceId && event.sourceId) continue;
        const candidateTime = new Date(candidate.publishedAt || candidate.ingestedAt || '').getTime();
        if (!Number.isFinite(targetTime) || !Number.isFinite(candidateTime)) continue;
        if (Math.abs(targetTime - candidateTime) > 15 * 60 * 1000) continue;
        if (candidate.eventCategory !== event.eventCategory) continue;
        const distance = Number.isFinite(Number(event.latitude)) && Number.isFinite(Number(event.longitude)) && Number.isFinite(Number(candidate.latitude)) && Number.isFinite(Number(candidate.longitude))
            ? haversineKm(Number(event.latitude),Number(event.longitude),Number(candidate.latitude),Number(candidate.longitude)) : null;
        const samePlace = (event.city && candidate.city && pulseNormalizeText(event.city) === pulseNormalizeText(candidate.city)) ||
            (event.district && candidate.district && pulseNormalizeText(event.district) === pulseNormalizeText(candidate.district));
        if (distance !== null ? distance > 50 : !samePlace) continue;
        if (pulseTextSimilarity(`${event.title} ${event.content}`,`${candidate.title} ${candidate.content}`) >= 0.96) return candidate;
    }
    return null;
}

function pulseCorroborated(event, allEvents) {
    for (const candidate of allEvents) {
        if (!candidate || candidate.id === event.id) continue;
        if (!['official', 'public_dataset', 'weather_api'].includes(candidate.sourceType)) continue;
        if (candidate.eventCategory !== event.eventCategory) continue;
        const a = new Date(event.publishedAt || event.ingestedAt || '').getTime();
        const b = new Date(candidate.publishedAt || candidate.ingestedAt || '').getTime();
        if (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) > 12 * 3600 * 1000) continue;
        const stateMatch = event.state && candidate.state && pulseNormalizeText(event.state) === pulseNormalizeText(candidate.state);
        const cityMatch = event.city && candidate.city && pulseNormalizeText(event.city) === pulseNormalizeText(candidate.city);
        if (stateMatch || cityMatch) return true;
    }
    return false;
}

async function pulseIngestLive() {
    const startedAt = Date.now();
    const [sachet, imd, social, web, citizen, sensorStatus, weatherApi] = await Promise.all([
        fetchSachetAlerts(),
        fetchPulseImdCombined(),
        fetchPulseBluesky(),
        fetchPulseGdelt(),
        pulseFetchCitizenReports(),
        getAssamSensorData().then(data => ({ available: Boolean(data?.ok), data, error: null })).catch(error => ({ available: false, data: null, error: error.message })),
        fetchPulseOpenMeteo()
    ]);

    const rawEvents = [
        ...pulseToSachetEvents(sachet),
        ...(imd?.fallback === 'CAP_RSS' ? pulseToImdCapEvents(imd) : pulseToImdEvents(imd)),
        ...pulseToSocialEvents(social),
        ...pulseToWebEvents(web),
        ...pulseToCitizenEvents(citizen),
        ...pulseToWeatherApiEvents(weatherApi),
        ...pulseToNwdpEvents(sensorStatus?.data)
    ];

    rawEvents.forEach(pulseApplyIntegrityScreen);
    await pulseEnrichLocations(rawEvents);

    await pulseRepairStoredNwdpTimestamps();
    const existing = await pulseLoadEvents();
    const working = existing.map(pulseNormalizeStoredEvent);
    await pulseEnrichLocations(working);
    if (!DATABASE_CONFIGURED && working.length) pulseWriteLocal(working);
    let duplicatesDetected = 0;
    let inserted = 0;
    const freshEvents = [];

    for (const event of rawEvents) {
        event.fingerprint = pulseFingerprint(event);
        const duplicate = pulseHasPossibleDuplicate(event, working);
        if (duplicate) {
            duplicatesDetected++;
            continue;
        }
        const corroborated = pulseCorroborated(event, [...working, ...freshEvents]);
        if (corroborated && !['VERIFIED_OFFICIAL', 'SOURCE_VERIFIED'].includes(event.verificationStatus)) {
            event.verificationStatus = 'CORROBORATED';
            event.trustScore = Math.min(100, Math.max(event.trustScore, pulseBuildTrust({
                sourceType: event.sourceType,
                title: event.title,
                content: event.content,
                city: event.city,
                state: event.state,
                latitude: event.latitude,
                longitude: event.longitude,
                mediaCount: event.media?.length || 0,
                corroborated: true
            })));
        }
        freshEvents.push(event);
    }

    for (const event of freshEvents) {
        await pulseUpsertEvent(event);
        inserted++;
    }

    const events = await pulseLoadEvents();
    const officialAlerts = (Array.isArray(sachet?.alerts) ? sachet.alerts.length : 0) + (Array.isArray(imd?.rows) ? imd.rows.length : 0);
    const connectorStatus = (feed, defaultStatus = 'UNAVAILABLE') => {
        if (feed?.available) return 'CONNECTED';
        const message = String(feed?.error || '').toLowerCase();
        if (/\b429\b|rate limit|too many requests/.test(message)) return 'RATE_LIMITED';
        if (/\b401\b|authentication|whitelist/.test(message)) return 'ACCESS_BLOCKED';
        if (/\b403\b|forbidden/.test(message)) return 'ACCESS_BLOCKED';
        if (feed && Array.isArray(feed?.alerts) && feed.alerts.length === 0 && !feed.error) return 'NO_DATA';
        return defaultStatus;
    };
    const storageMode = DATABASE_CONFIGURED && await ensurePulseStorage()
        ? 'PostgreSQL'
        : (await ensurePulseSupabaseStorage() ? 'Supabase REST' : 'Local JSON fallback');
    const nwdpConnectedFeeds = sensorStatus?.data
        ? Object.values(sensorStatus.data.feeds || {}).filter(feed => feed?.connected && feed?.latest).length
        : 0;
    const nwdpAvailableRecords = pulseToNwdpEvents(sensorStatus?.data).length;
    const sourceHealth = [
        { key: 'sachet', label: 'NDMA SACHET', type: 'Official alerts', available: Boolean(sachet?.available), status: connectorStatus(sachet), checkedAt: sachet?.checkedAt, detail: sachet?.error || `${Number(sachet?.alerts?.length || 0)} active alert records` },
        { key: 'imd', label: imd?.sourceName || 'IMD warnings', type: 'Official warnings', available: Boolean(imd?.available), status: connectorStatus(imd), checkedAt: imd?.checkedAt, detail: imd?.apiError ? `${imd?.error || 'CAP RSS fallback active'} · API: ${imd.apiError}` : (imd?.error || `${Number(imd?.rows?.length || 0)} warning rows`) },
        { key: 'bluesky', label: social?.sourceName || 'Public social feeds', type: 'Social stream', available: Boolean(social?.available), status: connectorStatus(social), checkedAt: social?.checkedAt, detail: [social?.error, `${Number(social?.posts?.length || 0)} posts matched`, Number(social?.filteredOut || 0) ? `${Number(social.filteredOut)} irrelevant posts filtered` : null].filter(Boolean).join(' · ') || 'No relevant India-weather posts returned' },
        { key: 'gdelt', label: web?.sourceName || 'Public web / news', type: 'Web/news stream', available: Boolean(web?.available), status: connectorStatus(web), checkedAt: web?.checkedAt, detail: web?.error || `${Number(web?.articles?.length || 0)} matching articles` },
        { key: 'weather_api', label: weatherApi?.sourceName || 'Open-Meteo public API', type: 'Public API observations', available: Boolean(weatherApi?.available), status: connectorStatus(weatherApi), checkedAt: weatherApi?.checkedAt, detail: weatherApi?.error || `${Number(weatherApi?.observations?.length || 0)} location observations` },
        {
            key: 'nwdp',
            label: 'NWDP/NWIC telemetry',
            type: 'Public dataset',
            available: nwdpConnectedFeeds > 0,
            status: nwdpConnectedFeeds > 0 ? 'CONNECTED' : 'NO_DATA',
            checkedAt: sensorStatus?.data?.updatedAt || pulseNowIso(),
            detail: sensorStatus?.error || `${nwdpConnectedFeeds} telemetry feeds with usable observations · ${nwdpAvailableRecords} records available`
        },
        { key: 'citizen', label: 'Citizen reports', type: 'Community', available: Boolean(citizen?.available), status: citizen?.storage === 'LOCAL_FALLBACK' ? (citizen?.reports?.length ? 'LOCAL_FALLBACK' : 'NO_DATA') : connectorStatus(citizen), checkedAt: citizen?.checkedAt, detail: citizen?.error || `${Number(citizen?.reports?.length || 0)} reports available` }
    ];

    sourceHealth.forEach(item => { if (!item.status) item.status = item.available ? 'CONNECTED' : 'UNAVAILABLE'; });

    const counts = events.reduce((acc, event) => {
        acc.total++;
        acc.byCategory[event.eventCategory] = (acc.byCategory[event.eventCategory] || 0) + 1;
        acc.bySource[event.sourceType] = (acc.bySource[event.sourceType] || 0) + 1;
        acc.byVerification[event.verificationStatus] = (acc.byVerification[event.verificationStatus] || 0) + 1;
        if (event.state) acc.byState[event.state] = (acc.byState[event.state] || 0) + 1;
        const hasGps = pulseValidCoordinate(event.latitude, event.longitude);
        if (hasGps) acc.mapped++;
        if (['DERIVED_GEOCODE','DERIVED_DISTRICT_GEOCODE'].includes(event.metadata?.locationBasis)) acc.derivedLocations++;
        if (Array.isArray(event.media) && event.media.length) acc.mediaRecords++;
        if (['NEEDS_REVIEW','LOW_CONFIDENCE','UNVERIFIED_CITIZEN'].includes(event.verificationStatus)) acc.reviewQueue++;
        return acc;
    }, { total: 0, byCategory: {}, bySource: {}, byVerification: {}, byState: {}, mapped: 0, derivedLocations: 0, mediaRecords: 0, reviewQueue: 0 });

    const overview = {
        ok: true,
        checkedAt: pulseNowIso(),
        latencyMs: Date.now() - startedAt,
        counts: {
            ingested: counts.total,
            recordsScreened: rawEvents.length,
            newRecords: inserted,
            duplicateCandidates: duplicatesDetected,
            inserted,
            duplicatesDetected,
            officialAlerts,
            officialAlertSources: { sachet: Array.isArray(sachet?.alerts) ? sachet.alerts.length : 0, imd: Array.isArray(imd?.rows) ? imd.rows.length : 0 },
            socialSignals: counts.bySource.social || 0,
            webSignals: counts.bySource.web || 0,
            citizenReports: counts.bySource.citizen || 0,
            weatherApiObservations: Array.isArray(weatherApi?.observations) ? weatherApi.observations.length : 0,
            publicDatasetRecords: counts.bySource.public_dataset || 0,
            sourceVerified: counts.byVerification.SOURCE_VERIFIED || 0,
            officialVerified: counts.byVerification.VERIFIED_OFFICIAL || 0,
            corroborated: counts.byVerification.CORROBORATED || 0,
            needsReview: counts.reviewQueue,
            lowConfidence: counts.byVerification.LOW_CONFIDENCE || 0
        },
        categoryCounts: counts.byCategory,
        sourceCounts: counts.bySource,
        verificationCounts: counts.byVerification,
        storage: { mode: storageMode, centralized: ['PostgreSQL','Supabase REST'].includes(storageMode), table: storageMode === 'Supabase REST' ? PULSE_SUPABASE_TABLE : null },
        apiObservations: Array.isArray(weatherApi?.observations) ? weatherApi.observations.slice(0, 20) : [],
        datasetObservations: pulseToNwdpEvents(sensorStatus?.data).slice(0, 20).map(event => ({
            title: event.title, sourceName: event.sourceName, sourceUrl: event.sourceUrl, publishedAt: event.publishedAt, station: event.metadata?.station || null,
            district: event.district || null, state: event.state || null, value: event.metadata?.value ?? null, unit: event.metadata?.unit || null, observationType: event.metadata?.observationType || null
        })),
        stateCounts: counts.byState,
        dataQuality: { mappedRecords: counts.mapped, derivedLocationRecords: counts.derivedLocations, recordsWithMedia: counts.mediaRecords },
        sourceHealth,
        sensorSnapshot: sensorStatus?.data ? {
            ok: sensorStatus.data.ok,
            updatedAt: sensorStatus.data.updatedAt,
            source: sensorStatus.data.source,
            connectedFeeds: Object.values(sensorStatus.data.feeds || {}).filter(item => item?.connected).length
        } : null,
        events: events.slice(0, 150),
        disclaimers: [
            'Pulse 2.0 ingests public/authorized sources and labels each record with its source and verification state.',
            'Social and citizen records are not treated as official warnings unless corroborated or manually verified.',
            'A trust score is a screening aid, not proof that a report is true or false.',
            'Official IMD/NDMA/district instructions take precedence over derived analytics.'
        ]
    };

    PULSE_CACHE.timestamp = Date.now();
    PULSE_CACHE.overview = overview;
    return overview;
}

app.get('/api/pulse/overview', async (req, res) => {
    try {
        const force = String(req.query?.refresh || '') === '1';
        if (!force && PULSE_CACHE.overview && Date.now() - PULSE_CACHE.timestamp < PULSE_CACHE_MS) {
            return res.json(PULSE_CACHE.overview);
        }
        return res.json(await pulseIngestLive());
    } catch (error) {
        console.error('Pulse overview error:', error);
        return res.status(502).json({ ok: false, error: 'Unable to build live Pulse 2.0 overview.', details: error.message });
    }
});

app.get('/api/pulse/events', async (req, res) => {
    try {
        const rows = await pulseLoadEvents();
        const from = req.query?.from ? new Date(req.query.from).getTime() : null;
        const to = req.query?.to ? new Date(req.query.to).getTime() : null;
        const event = String(req.query?.event || '').trim().toLowerCase();
        const state = String(req.query?.state || '').trim().toLowerCase();
        const verification = String(req.query?.verification || '').trim().toUpperCase();
        const source = String(req.query?.source || '').trim().toLowerCase();
        const q = pulseNormalizeText(req.query?.q || '');

        const filtered = rows.filter(row => {
            const t = new Date(row.publishedAt || row.ingestedAt || '').getTime();
            if (from && (!Number.isFinite(t) || t < from)) return false;
            if (to && (!Number.isFinite(t) || t > to)) return false;
            if (event && event !== 'all' && String(row.eventCategory || '').toLowerCase() !== event) return false;
            if (state && state !== 'all' && !pulseNormalizeText(`${row.city || ''} ${row.district || ''} ${row.state || ''}`).includes(pulseNormalizeText(state))) return false;
            if (verification && verification !== 'ALL' && String(row.verificationStatus || '').toUpperCase() !== verification) return false;
            if (source && source !== 'all' && String(row.sourceType || '').toLowerCase() !== source) return false;
            if (q && !pulseNormalizeText(`${row.title} ${row.content} ${row.city || ''} ${row.state || ''} ${row.author || ''}`).includes(q)) return false;
            return true;
        }).slice(0, 250);

        return res.json({ ok: true, count: filtered.length, events: filtered });
    } catch (error) {
        return res.status(500).json({ ok: false, error: 'Unable to query Pulse events.', details: error.message });
    }
});

app.patch('/api/pulse/events/:id/verification', requireAdmin, async (req, res) => {
    try {
        const status = String(req.body?.verificationStatus || '').toUpperCase();
        const updated = await pulseUpdateEventStatus(req.params.id, status);
        if (!updated) return res.status(404).json({ ok: false, error: 'Pulse event not found.' });
        return res.json({ ok: true, event: updated });
    } catch (error) {
        return res.status(400).json({ ok: false, error: error.message });
    }
});

// Background ingestion loop. It is opt-out so a deployed instance continuously
// refreshes the national feed without needing a dashboard user to press Refresh.
let pulseAutomationTimer = null;
if (PULSE_AUTOMATION_ENABLED) {
    pulseAutomationTimer = setInterval(() => {
        pulseIngestLive().catch(error => console.warn('Pulse background ingestion failed:', error.message));
    }, PULSE_AUTOMATION_INTERVAL_MS);
    if (typeof pulseAutomationTimer.unref === 'function') pulseAutomationTimer.unref();
}


app.get('/api/pulse/storage/status', async (_req, res) => {
    try {
        const mode = DATABASE_CONFIGURED && await ensurePulseStorage()
            ? 'PostgreSQL'
            : (await ensurePulseSupabaseStorage() ? 'Supabase REST' : 'Local JSON fallback');
        return res.json({ ok:true, mode, centralized:['PostgreSQL','Supabase REST'].includes(mode), table:mode === 'Supabase REST' ? PULSE_SUPABASE_TABLE : (mode === 'PostgreSQL' ? 'public.pulse_events' : null) });
    } catch (error) {
        return res.status(500).json({ ok:false, error:error.message });
    }
});

// =====================================================
// STATIC WEBSITE
// =====================================================

app.use(express.static(ROOT));

app.get('/', (_req, res) => {
    res.sendFile(path.join(ROOT, 'index.html'));
});

// =====================================================
// SERVER
// =====================================================

const PORT = process.env.PORT || 5000;

app.listen(PORT, '0.0.0.0', async () => {
    console.log(`RakshaNet backend running on port ${PORT}`);

    if (DATABASE_CONFIGURED) {
        try {
            const result = await pool.query('SELECT NOW() AS time');
            console.log('PostgreSQL connected:', result.rows[0].time);
        } catch (error) {
            console.warn('PostgreSQL unavailable; local fallbacks remain active:', error.message);
        }
    } else {
        console.log('INFO: DATABASE_URL is not configured; local JSON fallbacks are active.');
    }

    if (DATABASE_CONFIGURED) {
        try {
            await ensurePulseStorage();
        } catch (error) {
            console.warn('Pulse storage initialization skipped:', error.message);
        }

        try {
            await ensureAlertDeliveryTables();
        } catch (error) {
            console.warn('Alert storage initialization skipped:', error.message);
        }
    } else {
        console.log('INFO: Pulse PostgreSQL storage is disabled until DATABASE_URL is configured.');
        console.log('INFO: Alert delivery database tables are disabled until DATABASE_URL is configured.');
    }

    if (ALERT_AUTOMATION_ENABLED && DATABASE_CONFIGURED) {
        console.log(`RakshaNet automated early-warning scanner enabled every ${Math.round(ALERT_AUTOMATION_INTERVAL_MS / 1000)}s.`);
        runAutomatedAlertScan().catch(error => console.warn('Initial automated alert scan failed:', error.message));
        setInterval(() => {
            runAutomatedAlertScan().catch(error => console.warn('Automated alert scan failed:', error.message));
        }, ALERT_AUTOMATION_INTERVAL_MS);
    } else if (ALERT_AUTOMATION_ENABLED && !DATABASE_CONFIGURED) {
        console.log('INFO: Automated alert delivery is paused until database and delivery channels are configured.');
    } else {
        console.log('INFO: Automated early-warning scanner is disabled.');
    }
});
