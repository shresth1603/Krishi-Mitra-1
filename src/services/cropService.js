/**
 * cropService.js
 * Handles communication with Krishi Mitra crop recommendation ML endpoints.
 */

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const cropService = {
  /**
   * Recommend crops based on soil and weather metrics.
   * @param {Object} params - { N, P, K, temperature, humidity, ph, rainfall }
   */
  recommendCrop: async (params) => {
    const res = await fetch(`${API_URL}/api/crop-recommendation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to get crop recommendation." }));
      throw new Error(err.detail || `Server error: ${res.status}`);
    }

    return res.json();
  },

  /**
   * Fetch crop recommendation model information and supported classes.
   */
  getModelInfo: async () => {
    try {
      const res = await fetch(`${API_URL}/api/crop-recommendation/info`);
      if (!res.ok) return null;
      return res.json();
    } catch {
      return null;
    }
  },
};
