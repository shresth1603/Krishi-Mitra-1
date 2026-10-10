/**
 * marketService.js
 * Routes mandi data and forecasting through backend — keeps keys server-side.
 */

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const marketService = {
  getMarketPrices: async (state = "Maharashtra", commodity = "Tomato") => {
    const url = `${API_URL}/api/mandi?state=${encodeURIComponent(state)}&commodity=${encodeURIComponent(commodity)}`;
    const res = await fetch(url);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Market API error ${res.status}`);
    }
    const json = await res.json();
    return {
      data:        json.data,
      source:      json.source,
      lastUpdated: json.lastUpdated,
      count:       json.count,
    };
  },

  getMandiForecast: async (state = "Maharashtra", commodity = "Tomato", market = null, horizonDays = 7) => {
    const res = await fetch(`${API_URL}/api/mandi/forecast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state,
        commodity,
        market: market || null,
        horizon_days: horizonDays,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Forecast error ${res.status}`);
    }
    return res.json();
  },

  getForecastStatus: async () => {
    const res = await fetch(`${API_URL}/api/mandi/forecast/status`);
    if (!res.ok) return null;
    return res.json();
  },
};