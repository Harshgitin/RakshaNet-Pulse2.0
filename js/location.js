/* =========================================================
   RakshaNet Location System
   ========================================================= */
const RakshaLocation = {
    current: null,
    getLocation(successCallback, errorCallback) {
        if (!navigator.geolocation) {
            errorCallback && errorCallback("Geolocation is not supported by this browser.");
            return;
        }
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const location = {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                    accuracy: position.coords.accuracy,
                    timestamp: new Date().toISOString()
                };
                RakshaLocation.current = location;
                localStorage.setItem("rakshanet_location", JSON.stringify(location));
                successCallback && successCallback(location);
            },
            (error) => {
                let message = "Unable to detect your location.";
                if (error.code === 1) message = "Location permission was denied.";
                if (error.code === 2) message = "Your location is currently unavailable.";
                if (error.code === 3) message = "Location request timed out.";
                errorCallback && errorCallback(message);
            },
            { enableHighAccuracy: true, timeout: 12000, maximumAge: 30000 }
        );
    },
    getSavedLocation() {
        try {
            const raw = localStorage.getItem("rakshanet_location");
            return raw ? JSON.parse(raw) : null;
        } catch (_) { return null; }
    }
};
