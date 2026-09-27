/* RakshaNet app/browser early-warning notifications. */
(function () {
    'use strict';

    const API = '';
    const USER_KEY = 'rakshanet_user';

    function getUserId() {
        try {
            const user = JSON.parse(localStorage.getItem(USER_KEY) || 'null');
            return user?.id || null;
        } catch (_) {
            return null;
        }
    }

    async function getConfig() {
        const response = await fetch(`${API}/api/notifications/config`, {
            headers: { Accept: 'application/json' }
        });
        if (!response.ok) throw new Error(`Notification config HTTP ${response.status}`);
        return response.json();
    }

    async function ensurePermission() {
        if (!('Notification' in window)) throw new Error('This browser does not support notifications.');
        if (Notification.permission === 'granted') return 'granted';
        // Permission prompts are deliberately triggered only from a button click.
        return Notification.requestPermission();
    }

    function urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
        const rawData = window.atob(base64);
        return Uint8Array.from([...rawData].map(char => char.charCodeAt(0)));
    }

    async function enablePush() {
        if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
            throw new Error('This browser does not support Web Push.');
        }
        const permission = await ensurePermission();
        if (permission !== 'granted') throw new Error('Notification permission was not granted.');

        const config = await getConfig();
        if (!config.push?.available || !config.push?.publicKey) {
            throw new Error('Server-side Web Push is not configured yet. Run npm run generate:vapid and configure VAPID keys.');
        }

        const registration = await navigator.serviceWorker.ready;
        let subscription = await registration.pushManager.getSubscription();
        if (!subscription) {
            subscription = await registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: urlBase64ToUint8Array(config.push.publicKey)
            });
        }

        const response = await fetch(`${API}/api/notifications/subscribe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: getUserId(),
                subscription: subscription.toJSON()
            })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data?.error || 'Unable to register push subscription.');
        return { ok: true, subscription, config };
    }

    async function testPush() {
        const response = await fetch(`${API}/api/notifications/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: getUserId() })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data?.error || 'Push test failed.');
        return data;
    }

    async function config() {
        return getConfig();
    }

    async function renderStatus(element) {
        if (!element) return;
        try {
            const result = await getConfig();
            const pushReady = result.push?.available;
            const smsReady = result.sms?.available;
            const automation = result.automation?.enabled;
            element.textContent = pushReady || smsReady || automation
                ? `Push ${pushReady ? 'Ready' : 'Not configured'} · SMS ${smsReady ? 'Ready' : 'Not configured'} · Automation ${automation ? 'ON' : 'Standby'}`
                : 'Delivery providers not configured';
            element.dataset.pushReady = pushReady ? 'true' : 'false';
            element.dataset.smsReady = smsReady ? 'true' : 'false';
        } catch (error) {
            element.textContent = `Notification service unavailable: ${error.message}`;
        }
    }

    window.RakshaNetNotifications = {
        getConfig: config,
        enablePush,
        testPush,
        renderStatus,
        getUserId
    };
})();
