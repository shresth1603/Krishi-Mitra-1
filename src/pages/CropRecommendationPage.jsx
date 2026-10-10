import { useState, useEffect } from "react";
import { cropService } from "../services/cropService";

const PRESETS = [
  {
    name: "Rice / Kharif Alluvial",
    desc: "High N, high humidity & high rainfall",
    values: { N: 90, P: 42, K: 43, temperature: 24, humidity: 82, ph: 6.5, rainfall: 220 },
  },
  {
    name: "Wheat / Rabi Loam",
    desc: "Moderate N, cooler climate & moderate moisture",
    values: { N: 105, P: 55, K: 30, temperature: 21, humidity: 55, ph: 6.8, rainfall: 85 },
  },
  {
    name: "Cotton / Black Soil",
    desc: "Warm temperature, moderate rainfall",
    values: { N: 120, P: 45, K: 20, temperature: 28, humidity: 75, ph: 7.2, rainfall: 75 },
  },
  {
    name: "Apple / Hill Orchard",
    desc: "Cool temperature, balanced NPK, moderate rainfall",
    values: { N: 25, P: 135, K: 200, temperature: 22, humidity: 92, ph: 6.0, rainfall: 110 },
  },
];

const CropRecommendationPage = () => {
  const [formData, setFormData] = useState({
    N: 90,
    P: 42,
    K: 43,
    temperature: 24.5,
    humidity: 80.0,
    ph: 6.5,
    rainfall: 200.0,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    cropService.getModelInfo().then(setModelInfo).catch(() => {});
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value === "" ? "" : parseFloat(value) || value,
    }));
  };

  const handleApplyPreset = (preset) => {
    setFormData(preset.values);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Validation
    const errors = [];
    if (formData.N < 0 || formData.N > 200) errors.push("Nitrogen (N) must be between 0 and 200 kg/ha");
    if (formData.P < 0 || formData.P > 200) errors.push("Phosphorus (P) must be between 0 and 200 kg/ha");
    if (formData.K < 0 || formData.K > 250) errors.push("Potassium (K) must be between 0 and 250 kg/ha");
    if (formData.temperature < 0 || formData.temperature > 55) errors.push("Temperature must be between 0 and 55 °C");
    if (formData.humidity < 0 || formData.humidity > 100) errors.push("Humidity must be between 0 and 100 %");
    if (formData.ph < 0 || formData.ph > 14) errors.push("Soil pH must be between 0 and 14");
    if (formData.rainfall < 0 || formData.rainfall > 4000) errors.push("Rainfall must be between 0 and 4000 mm");

    if (errors.length > 0) {
      setError(errors.join(" • "));
      setLoading(false);
      return;
    }

    try {
      const data = await cropService.recommendCrop({
        N: Number(formData.N),
        P: Number(formData.P),
        K: Number(formData.K),
        temperature: Number(formData.temperature),
        humidity: Number(formData.humidity),
        ph: Number(formData.ph),
        rainfall: Number(formData.rainfall),
      });
      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to obtain crop recommendation. Please check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="crop-rec-page" style={{ padding: "30px 16px 80px", maxWidth: "1050px", margin: "0 auto", color: "#1f2937" }}>
      {/* Title */}
      <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "8px", background: "#ecfdf5", padding: "6px 16px", borderRadius: "20px", color: "#065f46", fontSize: "13px", fontWeight: "600", marginBottom: "12px", border: "1px solid #a7f3d0" }}>
          <span>🌱 ML Model 2</span>
          <span>•</span>
          <span>Random Forest Classification</span>
        </div>
        <h1 style={{ fontSize: "2.2rem", fontWeight: "800", color: "#111827", margin: "0 0 10px" }}>
          Intelligent Crop Recommendation
        </h1>
        <p style={{ color: "#4b5563", fontSize: "1.05rem", maxWidth: "680px", margin: "0 auto", lineHeight: "1.6" }}>
          Provide soil nutrients (N-P-K), soil pH, and weather parameters to receive data-driven crop recommendations for your field.
        </p>
      </div>

      {/* Preset Buttons */}
      <div style={{ background: "#ffffff", padding: "18px 22px", borderRadius: "14px", border: "1px solid #e5e7eb", marginBottom: "26px", boxShadow: "0 1px 3px rgba(0,0,0,0.05)" }}>
        <div style={{ fontSize: "12px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.08em", color: "#6b7280", marginBottom: "10px" }}>
          💡 Quick Presets (Sample Agro-Ecological Profiles)
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleApplyPreset(p)}
              style={{
                background: "#f9fafb",
                border: "1px solid #d1d5db",
                borderRadius: "8px",
                padding: "8px 14px",
                cursor: "pointer",
                fontSize: "13px",
                fontWeight: "500",
                color: "#374151",
                transition: "all 0.15s ease",
              }}
              onMouseOver={(e) => { e.currentTarget.style.borderColor = "#10b981"; e.currentTarget.style.background = "#ecfdf5"; }}
              onMouseOut={(e) => { e.currentTarget.style.borderColor = "#d1d5db"; e.currentTarget.style.background = "#f9fafb"; }}
            >
              <strong>{p.name}</strong> — <span style={{ color: "#6b7280", fontSize: "12px" }}>{p.desc}</span>
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: result ? "1fr 1fr" : "1fr", gap: "24px", alignItems: "start" }}>
        {/* Form Container */}
        <div style={{ background: "#ffffff", padding: "28px", borderRadius: "16px", border: "1px solid #e5e7eb", boxShadow: "0 2px 6px rgba(0,0,0,0.05)" }}>
          <h2 style={{ fontSize: "1.25rem", fontWeight: "700", marginBottom: "18px", color: "#111827" }}>
            Field & Climate Parameters
          </h2>

          <form onSubmit={handleSubmit}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "14px", marginBottom: "16px" }}>
              <div>
                <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#374151", marginBottom: "6px" }}>
                  Nitrogen (N)
                  <span style={{ fontSize: "11px", color: "#6b7280", display: "block" }}>kg/ha (0 - 200)</span>
                </label>
                <input
                  type="number"
                  name="N"
                  step="any"
                  value={formData.N}
                  onChange={handleChange}
                  required
                  style={{ width: "100%", padding: "10px 12px", borderRadius: "8px", border: "1px solid #d1d5db", fontSize: "14px" }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#374151", marginBottom: "6px" }}>
                  Phosphorus (P)
                  <span style={{ fontSize: "11px", color: "#6b7280", display: "block" }}>kg/ha (0 - 200)</span>
                </label>
                <input
                  type="number"
                  name="P"
                  step="any"
                  value={formData.P}
                  onChange={handleChange}
                  required
                  style={{ width: "100%", padding: "10px 12px", borderRadius: "8px", border: "1px solid #d1d5db", fontSize: "14px" }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#374151", marginBottom: "6px" }}>
                  Potassium (K)
                  <span style={{ fontSize: "11px", color: "#6b7280", display: "block" }}>kg/ha (0 - 250)</span>
                </label>
                <input
                  type="number"
                  name="K"
                  step="any"
                  value={formData.K}
                  onChange={handleChange}
                  required
                  style={{ width: "100%", padding: "10px 12px", borderRadius: "8px", border: "1px solid #d1d5db", fontSize: "14px" }}
                />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px", marginBottom: "16px" }}>
              <div>
                <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#374151", marginBottom: "6px" }}>
                  Temperature (°C)
                  <span style={{ fontSize: "11px", color: "#6b7280", display: "block" }}>0 - 55 °C</span>
                </label>
                <input
                  type="number"
                  name="temperature"
                  step="0.1"
                  value={formData.temperature}
                  onChange={handleChange}
                  required
                  style={{ width: "100%", padding: "10px 12px", borderRadius: "8px", border: "1px solid #d1d5db", fontSize: "14px" }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#374151", marginBottom: "6px" }}>
                  Relative Humidity (%)
                  <span style={{ fontSize: "11px", color: "#6b7280", display: "block" }}>0 - 100 %</span>
                </label>
                <input
                  type="number"
                  name="humidity"
                  step="0.1"
                  value={formData.humidity}
                  onChange={handleChange}
                  required
                  style={{ width: "100%", padding: "10px 12px", borderRadius: "8px", border: "1px solid #d1d5db", fontSize: "14px" }}
                />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px", marginBottom: "24px" }}>
              <div>
                <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#374151", marginBottom: "6px" }}>
                  Soil pH (Acidity/Alkalinity)
                  <span style={{ fontSize: "11px", color: "#6b7280", display: "block" }}>3.5 - 9.5 (Neutral: 7)</span>
                </label>
                <input
                  type="number"
                  name="ph"
                  step="0.1"
                  value={formData.ph}
                  onChange={handleChange}
                  required
                  style={{ width: "100%", padding: "10px 12px", borderRadius: "8px", border: "1px solid #d1d5db", fontSize: "14px" }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#374151", marginBottom: "6px" }}>
                  Annual Rainfall (mm)
                  <span style={{ fontSize: "11px", color: "#6b7280", display: "block" }}>0 - 4000 mm</span>
                </label>
                <input
                  type="number"
                  name="rainfall"
                  step="any"
                  value={formData.rainfall}
                  onChange={handleChange}
                  required
                  style={{ width: "100%", padding: "10px 12px", borderRadius: "8px", border: "1px solid #d1d5db", fontSize: "14px" }}
                />
              </div>
            </div>

            {error && (
              <div style={{ padding: "12px 14px", background: "#fef2f2", color: "#dc2626", borderRadius: "8px", border: "1px solid #fecaca", fontSize: "13px", marginBottom: "18px" }}>
                ⚠️ {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              style={{
                width: "100%",
                padding: "14px",
                background: loading ? "#9ca3af" : "linear-gradient(135deg, #059669 0%, #10b981 100%)",
                color: "#ffffff",
                border: "none",
                borderRadius: "10px",
                fontSize: "15px",
                fontWeight: "700",
                cursor: loading ? "not-allowed" : "pointer",
                boxShadow: "0 2px 4px rgba(16,185,129,0.2)",
              }}
            >
              {loading ? "Analyzing Soil & Climate Profiles…" : "🔍 Get Crop Recommendation"}
            </button>
          </form>
        </div>

        {/* Results Container */}
        {result && (
          <div style={{ background: "#ffffff", padding: "28px", borderRadius: "16px", border: "1px solid #10b981", boxShadow: "0 4px 14px rgba(16,185,129,0.15)" }}>
            <div style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.1em", color: "#059669", marginBottom: "6px" }}>
              RECOMMENDED CROP
            </div>

            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "16px" }}>
              <h3 style={{ fontSize: "2rem", fontWeight: "800", color: "#065f46", margin: 0, textTransform: "capitalize" }}>
                🌾 {result.recommended_crop}
              </h3>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#059669" }}>
                {result.confidence}% <span style={{ fontSize: "12px", color: "#6b7280", fontWeight: "500" }}>Confidence</span>
              </div>
            </div>

            {/* Top Alternatives */}
            {result.top_alternatives && result.top_alternatives.length > 0 && (
              <div style={{ marginBottom: "20px", background: "#f9fafb", padding: "14px 16px", borderRadius: "10px", border: "1px solid #f3f4f6" }}>
                <div style={{ fontSize: "12px", fontWeight: "700", color: "#4b5563", marginBottom: "8px", textTransform: "uppercase" }}>
                  Ranked Alternatives
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  {result.top_alternatives.map((alt, i) => (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                      <span style={{ textTransform: "capitalize", fontWeight: "500", color: "#374151" }}>
                        {i + 2}. {alt.crop}
                      </span>
                      <span style={{ fontWeight: "600", color: "#059669" }}>{alt.confidence}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Model Metadata */}
            <div style={{ fontSize: "12px", color: "#6b7280", paddingBottom: "14px", borderBottom: "1px solid #f3f4f6", marginBottom: "14px" }}>
              <div><strong>Model Algorithm:</strong> {result.model_version}</div>
              <div><strong>Validation Accuracy:</strong> {(result.train_accuracy * 100).toFixed(1)}%</div>
            </div>

            {/* Disclaimers & Grounded Limitations */}
            <div style={{ padding: "12px 14px", background: "#fffbeb", borderRadius: "8px", border: "1px solid #fde68a", fontSize: "12px", color: "#92400e", lineHeight: "1.5" }}>
              <div style={{ fontWeight: "700", marginBottom: "4px" }}>⚠️ Agricultural Limitation Notice:</div>
              <p style={{ margin: "0 0 6px" }}>
                {result.disclaimer}
              </p>
              <ul style={{ margin: 0, paddingLeft: "16px" }}>
                {result.limitations?.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CropRecommendationPage;
