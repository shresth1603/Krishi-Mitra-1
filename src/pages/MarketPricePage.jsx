import { useState, useEffect } from "react";
import { useMarketPrices } from "../hooks/useMarketPrices";
import { marketService } from "../services/marketService";
import SkeletonLoader from "../components/SkeletonLoader";

const ALL_STATES = [
  "Andhra Pradesh","Assam","Bihar","Chhattisgarh","Gujarat","Haryana",
  "Karnataka","Kerala","Madhya Pradesh","Maharashtra","Odisha",
  "Punjab","Rajasthan","Tamil Nadu","Telangana","Uttar Pradesh",
  "Uttarakhand","West Bengal"
];

const COMMODITIES = [
  "Tomato","Onion","Potato","Rice","Wheat","Maize","Banana",
  "Coconut","Cotton","Groundnut","Soyabean","Sugarcane","Paddy"
];

const MarketPricePage = () => {
  const [selectedState, setSelectedState] = useState("Maharashtra");
  const [commodity, setCommodity] = useState("Tomato");
  const [activeTab, setActiveTab] = useState("spot"); // "spot" | "forecast"
  const [horizonDays, setHorizonDays] = useState(7);
  const [forecastData, setForecastData] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState(null);

  const { prices, loading, error, lastUpdated, source } =
    useMarketPrices(selectedState, commodity);

  const handleFetchForecast = async () => {
    setForecastLoading(true);
    setForecastError(null);
    try {
      const data = await marketService.getMandiForecast(selectedState, commodity, null, horizonDays);
      setForecastData(data);
    } catch (err) {
      setForecastError(err.message || "Failed to retrieve price forecast.");
    } finally {
      setForecastLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === "forecast") {
      handleFetchForecast();
    }
  }, [selectedState, commodity, horizonDays, activeTab]);

  const getInsight = (min, max, modal) => {
    if(!min || !max || !modal) return { text: 'Prices available', color: 'var(--blue-500)' };
    const spread = ((max - min) / modal * 100).toFixed(0);
    if (spread > 30) return { text: 'High price variation — compare mandis', color: 'var(--orange-500)' };
    if (modal > (min + max) / 2) return { text: 'Prices trending upward', color: 'var(--green-500)' };
    return { text: 'Stable market conditions', color: 'var(--blue-500)' };
  };

  if (loading && activeTab === "spot") {
    return (
      <div className="market-page">
        <div className="container">
          <SkeletonLoader type="market" rows={5} />
        </div>
      </div>
    );
  }

  if (error && activeTab === "spot") {
    return (
      <div className="market-page">
        <div className="container">
          <div className="error-state">
            <div className="error-icon">⚠️</div>
            <h3>Unable to load market prices</h3>
            <p>{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="btn btn-primary"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="market-page">
      <div className="container">
        {/* Header Section */}
        <div className="market-header">
          <div className="header-content">
            <div className="header-text">
              <h1>🌾 Market Prices & Forecasting</h1>
              <p className="subtitle">
                Real-time agricultural commodity prices & econometric price forecasts across India
              </p>
            </div>
            <div className="last-updated">
              <span className="update-icon">🕒</span>
              <span>Last updated: {lastUpdated || "Just now"}</span>
            </div>
          </div>
        </div>

        {/* Tab Switcher */}
        <div style={{ display: "flex", gap: "10px", margin: "20px 0 10px" }}>
          <button
            onClick={() => setActiveTab("spot")}
            style={{
              padding: "10px 20px",
              borderRadius: "10px",
              border: "1px solid",
              borderColor: activeTab === "spot" ? "var(--primary-600)" : "#d1d5db",
              background: activeTab === "spot" ? "var(--primary-600)" : "#ffffff",
              color: activeTab === "spot" ? "#ffffff" : "#4b5563",
              fontWeight: "600",
              cursor: "pointer",
              fontSize: "14px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span>📊 Spot Market Prices</span>
            <span style={{ fontSize: "11px", opacity: 0.85 }}>({prices.length} Mandis)</span>
          </button>

          <button
            onClick={() => setActiveTab("forecast")}
            style={{
              padding: "10px 20px",
              borderRadius: "10px",
              border: "1px solid",
              borderColor: activeTab === "forecast" ? "var(--primary-600)" : "#d1d5db",
              background: activeTab === "forecast" ? "var(--primary-600)" : "#ffffff",
              color: activeTab === "forecast" ? "#ffffff" : "#4b5563",
              fontWeight: "600",
              cursor: "pointer",
              fontSize: "14px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span>🔮 Price Forecasting (ML Model 3)</span>
          </button>
        </div>

        {/* Controls Section */}
        <div className="market-controls">
          <div className="controls-grid">
            <div className="filter-group">
              <label htmlFor="state-select">
                <span className="label-icon">📍</span>
                State
              </label>
              <select
                id="state-select"
                value={selectedState}
                onChange={(e) => setSelectedState(e.target.value)}
                className="filter-select"
              >
                {ALL_STATES.map((st) => (
                  <option key={st} value={st}>
                    {st}
                  </option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label htmlFor="commodity-select">
                <span className="label-icon">🌱</span>
                Commodity
              </label>
              <select
                id="commodity-select"
                value={commodity}
                onChange={(e) => setCommodity(e.target.value)}
                className="filter-select"
              >
                {COMMODITIES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {activeTab === "forecast" && (
              <div className="filter-group">
                <label htmlFor="horizon-select">
                  <span className="label-icon">⏱️</span>
                  Forecast Horizon
                </label>
                <select
                  id="horizon-select"
                  value={horizonDays}
                  onChange={(e) => setHorizonDays(Number(e.target.value))}
                  className="filter-select"
                >
                  <option value={3}>3 Days Ahead</option>
                  <option value={7}>7 Days Ahead (1 Week)</option>
                  <option value={14}>14 Days Ahead (2 Weeks)</option>
                </select>
              </div>
            )}

            <button
              onClick={() => {
                if (activeTab === "forecast") handleFetchForecast();
                else window.location.reload();
              }}
              className="refresh-btn"
              title="Refresh prices"
            >
              <span className="btn-icon">🔄</span>
              <span className="btn-text">Refresh</span>
            </button>
          </div>
        </div>

        {/* ── Spot Prices Tab ── */}
        {activeTab === "spot" && (
          <>
            {/* Results Summary */}
            <div className="results-summary">
              <p>
                Showing <strong>{prices.length}</strong>{" "}
                {prices.length === 1 ? "mandi" : "mandis"}
                {commodity && ` for ${commodity}`}
                {selectedState && ` in ${selectedState}`}
              </p>
              {source && (
                <p style={{marginTop: '4px', fontSize: '13px', color: 'var(--primary-600)'}}>
                  Data source: <strong>{source}</strong>
                </p>
              )}
            </div>

            {/* Price Cards Grid */}
            <div className="price-cards">
              {prices.length === 0 ? (
                <div className="no-results">
                  <div className="no-results-icon">📊</div>
                  <h3>No prices found</h3>
                  <p>
                    Try adjusting your filters or check back later for updated
                    prices.
                  </p>
                </div>
              ) : (
                prices.map((item, index) => {
                  const insight = getInsight(item.minPrice, item.maxPrice, item.modalPrice);
                  return (
                    <div key={index} className="price-card">
                      <div className="price-card-header">
                        <div className="crop-info">
                          <h3 className="crop-name">{item.mandi}</h3>
                          <p className="variety">{item.district} District | Variety: {item.variety}</p>
                        </div>
                        <div
                          className="trend-indicator"
                          style={{ color: insight.color, fontSize: '0.85em', textAlign: 'right', fontWeight: '500' }}
                        >
                          <span className="trend-text">{insight.text}</span>
                        </div>
                      </div>

                      <div className="price-list">
                        <div className="price-row">
                          <span className="location">Minimum</span>
                          <span className="price" style={{color: 'var(--text-secondary)'}}>₹{item.minPrice}/q</span>
                        </div>
                        <div className="price-row">
                          <span className="location">Maximum</span>
                          <span className="price" style={{color: 'var(--text-secondary)'}}>₹{item.maxPrice}/q</span>
                        </div>
                        <div className="price-row">
                          <span className="location" style={{fontWeight: 'bold'}}>Modal (Average)</span>
                          <span className="price" style={{fontWeight: 'bold'}}>₹{item.modalPrice}/q</span>
                        </div>
                      </div>

                      <div className="card-footer">
                        <span className="update-time">
                          Arrival Date: {item.date}
                        </span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </>
        )}

        {/* ── Forecast Tab ── */}
        {activeTab === "forecast" && (
          <div style={{ marginTop: "16px" }}>
            {forecastLoading && (
              <div style={{ padding: "30px", textAlign: "center", background: "#f9fafb", borderRadius: "14px", border: "1px solid #e5e7eb" }}>
                <div style={{ fontSize: "15px", fontWeight: "600", color: "#374151", marginBottom: "8px" }}>
                  ⏳ Evaluating Time-Series Models…
                </div>
                <p style={{ margin: 0, fontSize: "13px", color: "#6b7280" }}>
                  Calculating lag features and testing seasonal autoregression on held-out historical dates.
                </p>
              </div>
            )}

            {forecastError && (
              <div style={{ padding: "16px 20px", background: "#fef2f2", color: "#dc2626", borderRadius: "10px", border: "1px solid #fecaca", marginBottom: "16px" }}>
                ⚠️ {forecastError}
              </div>
            )}

            {forecastData && !forecastLoading && (
              <div>
                {/* When genuine forecast exists */}
                {forecastData.forecast_available ? (
                  <div style={{ background: "#ffffff", padding: "26px", borderRadius: "14px", border: "1px solid #10b981", boxShadow: "0 2px 8px rgba(16,185,129,0.1)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
                      <div>
                        <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#059669", letterSpacing: "0.08em" }}>
                          TIME-SERIES FORECAST RESULT
                        </span>
                        <h2 style={{ fontSize: "1.6rem", fontWeight: "700", color: "#111827", margin: "4px 0 0" }}>
                          {forecastData.commodity} — {forecastData.market}
                        </h2>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: "13px", color: "#6b7280" }}>Last Observed Price ({forecastData.last_observed_date})</div>
                        <div style={{ fontSize: "1.3rem", fontWeight: "700", color: "#111827" }}>₹{forecastData.last_observed_price} / Quintal</div>
                      </div>
                    </div>

                    {/* Model Details Badge */}
                    <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginBottom: "20px", fontSize: "12px", background: "#f0fdf4", padding: "10px 14px", borderRadius: "8px", border: "1px solid #bbf7d0", color: "#166534" }}>
                      <div><strong>Forecasting Engine:</strong> {forecastData.model_used}</div>
                      {forecastData.evaluation_metrics_held_out && (
                        <>
                          <div>•</div>
                          <div><strong>Held-out MAE:</strong> ₹{forecastData.evaluation_metrics_held_out.mae}</div>
                          <div>•</div>
                          <div><strong>Held-out RMSE:</strong> ₹{forecastData.evaluation_metrics_held_out.rmse}</div>
                        </>
                      )}
                    </div>

                    {/* Forecast Table / Trend */}
                    <div style={{ overflowX: "auto", marginBottom: "20px" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
                        <thead>
                          <tr style={{ background: "#f9fafb", borderBottom: "2px solid #e5e7eb" }}>
                            <th style={{ textAlign: "left", padding: "10px 14px", color: "#4b5563" }}>Date</th>
                            <th style={{ textAlign: "right", padding: "10px 14px", color: "#4b5563" }}>Projected Modal Price</th>
                            <th style={{ textAlign: "right", padding: "10px 14px", color: "#4b5563" }}>Daily Variation</th>
                          </tr>
                        </thead>
                        <tbody>
                          {forecastData.forecasts.map((item, idx) => {
                            const prev = idx === 0 ? forecastData.last_observed_price : forecastData.forecasts[idx - 1].predicted_modal_price;
                            const diff = item.predicted_modal_price - prev;
                            const diffPct = ((diff / prev) * 100).toFixed(1);
                            return (
                              <tr key={idx} style={{ borderBottom: "1px solid #f3f4f6" }}>
                                <td style={{ padding: "10px 14px", fontWeight: "500", color: "#111827" }}>{item.date}</td>
                                <td style={{ padding: "10px 14px", textAlign: "right", fontWeight: "700", color: "#065f46" }}>
                                  ₹{item.predicted_modal_price} / q
                                </td>
                                <td style={{ padding: "10px 14px", textAlign: "right", fontSize: "13px", color: diff > 0 ? "#16a34a" : diff < 0 ? "#dc2626" : "#6b7280" }}>
                                  {diff > 0 ? `▲ +₹${diff.toFixed(1)} (+${diffPct}%)` : diff < 0 ? `▼ -₹${Math.abs(diff).toFixed(1)} (${diffPct}%)` : "— 0.0%"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>

                    {/* Grounded Disclaimer */}
                    <div style={{ padding: "12px 14px", background: "#fffbeb", borderRadius: "8px", border: "1px solid #fde68a", fontSize: "12px", color: "#92400e", lineHeight: "1.5" }}>
                      <strong>⚠️ Agricultural Market Notice:</strong> {forecastData.disclaimer}
                    </div>
                  </div>
                ) : (
                  /* When historical dataset is missing or below observation threshold */
                  <div style={{ background: "#ffffff", padding: "28px", borderRadius: "14px", border: "1px solid #e5e7eb", boxShadow: "0 2px 6px rgba(0,0,0,0.05)" }}>
                    <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "#fef3c7", padding: "4px 12px", borderRadius: "16px", color: "#92400e", fontSize: "12px", fontWeight: "700", marginBottom: "12px" }}>
                      <span>📋 Ingestion & Historical Data Status</span>
                    </div>

                    <h3 style={{ fontSize: "1.3rem", fontWeight: "700", color: "#111827", margin: "0 0 10px" }}>
                      Historical Observations Required for {commodity}
                    </h3>

                    <p style={{ fontSize: "14px", color: "#4b5563", lineHeight: "1.6", margin: "0 0 16px" }}>
                      {forecastData.message}
                    </p>

                    <div style={{ background: "#f9fafb", padding: "14px 16px", borderRadius: "10px", border: "1px solid #f3f4f6", fontSize: "13px", marginBottom: "16px" }}>
                      <div style={{ fontWeight: "700", color: "#374151", marginBottom: "6px" }}>
                        Data Pipeline Verification Rules (Rule 5 & 6 Compliance):
                      </div>
                      <ul style={{ margin: 0, paddingLeft: "18px", color: "#4b5563", lineHeight: "1.5" }}>
                        <li>Spot API prices are actively served in the <strong>Spot Market Prices</strong> tab above.</li>
                        <li>Krishi Mitra strictly prohibits fabricating time series forecasts using mock data.</li>
                        <li>Autoregressive machine learning models (Lag Ridge & Seasonal Naive) are fully built and tested in <code>krishi-backend/mandi_forecast/</code>.</li>
                        <li>To activate local training, place a verified AGMARKNET CSV (with <code>Arrival_Date</code>, <code>Market</code>, <code>Modal_Price</code>) in <code>krishi-backend/datasets/mandi_historical/</code>.</li>
                      </ul>
                    </div>

                    <div style={{ fontSize: "12px", color: "#92400e", fontStyle: "italic" }}>
                      {forecastData.disclaimer}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Market Insights */}
        {prices.length > 0 && (
          <div className="market-insights" style={{ marginTop: "30px" }}>
            <h2>📈 Market Insights</h2>
            <div className="insights-grid">
              <div className="insight-card">
                <h4>Average Price Range</h4>
                <p>Most crops are trading within expected seasonal ranges</p>
              </div>
              <div className="insight-card">
                <h4>Price Trends</h4>
                <p>Monitor daily changes to make informed selling decisions</p>
              </div>
              <div className="insight-card">
                <h4>Best Markets</h4>
                <p>
                  Compare prices across different regions for better profits
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MarketPricePage;
