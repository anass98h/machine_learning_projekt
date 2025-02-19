import React, { useState, useEffect } from "react";
import axios from "axios";
import GaugeChart from "react-gauge-chart";

function App() {
  // State for the active tab
  const [activeTab, setActiveTab] = useState("Regression"); // Default: Regression

  // Common state
  const [models, setModels] = useState([]); // state for models
  const [selectedModel, setSelectedModel] = useState(""); // state for selected model
  const [error, setError] = useState(""); // state for error
  const [successMessage, setSuccessMessage] = useState(""); // state for success message
  const [file, setFile] = useState(null); // state for the uploaded file
  const [loading, setLoading] = useState(false); // state for loading

  // State for the regression model

  const [score, setScore] = useState(0); // initial normalized score (0-1)
  const [category, setCategory] = useState(""); // state for the category
  const [modelName, setModelName] = useState(""); // state for the model version
  const [healthStatus, setHealthStatus] = useState(""); // state for health status
  const [modelsLoaded, setModelsLoaded] = useState(0); // state for models loaded
  const [refreshing, setRefreshing] = useState(false); // state for refreshing

  // State for the classification model
  const [weakestLink, setWeakestlink] = useState(null); // Per salvare classe e probabilità
  const [selectedCategorizer, setSelectedCategorizer] = useState(""); // Modello di classificazione selezionato
  const [categorizingModels, setCategorizingModels] = useState([]); // Modelli di classificazione

  useEffect(() => {
    Handlerun();
    fetchHealthCheck();
    fetchCategorizingModels();
  }, []);

  // Update handleRun function to handle the models list

  const Handlerun = async () => {
    setLoading(true);
    setError("");
    setSuccessMessage("");
    try {
      const response = await axios.get("http://localhost:8000/models");

      // Update the models list
      setModels(response.data);
      if (response.data.length > 0) {
        setSelectedModel(response.data[0].name);
      }
    } catch (error) {
      console.error("Error fetching models:", error);
      setError(
        "An error occurred while fetching the models. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  // Fetch categorizing models

  const fetchCategorizingModels = async () => {
    try {
      const response = await axios.get(
        "http://localhost:8000/categorizing-models"
      );
      setCategorizingModels(response.data);
      if (response.data.length > 0)
        setSelectedCategorizer(response.data[0].name);
    } catch (err) {
      setError("Failed to fetch categorizing models.");
    }
  };

  // Update handleRefreshModels function
  const handleRefreshModels = async () => {
    setRefreshing(true);
    setError("");
    setSuccessMessage("");
    try {
      // First call the refresh-models endpoint
      const refreshResponse = await axios.post(
        "http://localhost:8000/refresh-models"
      );
      setSuccessMessage(refreshResponse.data.message);

      // Add a delay to ensure backend has time to complete the refresh
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Retry fetching models up to 3 times with increasing delays
      const maxAttempts = 3;

      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          await Handlerun();
          await fetchHealthCheck();
          break; // If successful, exit the loop
        } catch (error) {
          console.error(`Tentativo ${attempt} fallito:`, error);
          if (attempt === maxAttempts) {
            throw error; // If all attempts failed, throw the error
          }
          // Wait longer between each attempt
          await new Promise((resolve) => setTimeout(resolve, attempt * 1000));
        }
      }
    } catch (error) {
      console.error("Error refreshing models:", error);
      if (error.response?.data?.detail) {
        setError(error.response.data.detail);
      } else {
        setError("Failed to refresh models. Please try again.");
      }
    } finally {
      setRefreshing(false);
    }
  };

  const handleRefreshCategorizingModels = async () => {
    setRefreshing(true);
    setError("");
    setSuccessMessage("");
    try {
      // Call the endpoint to refresh categorizing models
      const refreshResponse = await axios.post(
        "http://localhost:8000/refresh-models"
      );
      setSuccessMessage(refreshResponse.data.message);

      // Add a delay to ensure backend updates the categorizing models
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Retry fetching categorizing models up to 3 times with increasing delays
      const maxAttempts = 3;

      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          await fetchCategorizingModels();
          break; // If successful, exit the loop
        } catch (error) {
          console.error(`Attempt ${attempt} failed:`, error);
          if (attempt === maxAttempts) {
            throw error; // If all attempts fail, throw the error
          }
          await new Promise((resolve) => setTimeout(resolve, attempt * 1000));
        }
      }
    } catch (error) {
      console.error("Error refreshing categorizing models:", error);
      if (error.response?.data?.detail) {
        setError(error.response.data.detail);
      } else {
        setError("Failed to refresh categorizing models. Please try again.");
      }
    } finally {
      setRefreshing(false);
    }
  };

  // Update fetchHealthCheck function
  const fetchHealthCheck = async () => {
    try {
      const response = await axios.get("http://localhost:8000/health");
      setHealthStatus(response.data.status);
      setModelsLoaded(response.data.models_loaded);
    } catch (error) {
      console.error("Error fetching health status:", error);
      setError("An error occurred while fetching the health status.");
    }
  };

  const handleFileUpload = async (endpoint) => {
    if (!file) {
      setError("Please upload a file!");
      return;
    }

    // Check for CSV file for both endpoints
    if (!file.name.endsWith(".csv") || file.type !== "text/csv") {
      setError("Invalid file format! Please upload a CSV file.");
      return;
    }

    const modelToUse =
      endpoint === "predict" ? selectedModel : selectedCategorizer;
    if (!modelToUse) {
      setError("Please select a model!");
      return;
    }

    setLoading(true);
    setError("");
    setSuccessMessage("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        `http://localhost:8000/${endpoint}/${modelToUse}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (endpoint === "predict") {
        const data = response.data;
        setCategory(data.category);
        setScore(data.score);
        setModelName(data.model_name);
        setSuccessMessage("Prediction completed successfully!");
      } else if (endpoint === "classify-weakest-link") {
        const data = response.data;
        setWeakestlink(data.weakest_link);
        setModelName(data.model_name);
        setSuccessMessage("Classification completed successfully!");
      }
    } catch (error) {
      console.error("Error during the operation:", error);
      if (error.response?.data?.detail) {
        setError(error.response.data.detail);
      } else {
        setError(
          error.message || "An unknown error occurred. Please try again."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  // Dashboard improvements
  const dashboardStyles = {
    container: {
      padding: "20px",
      backgroundColor: "#f0f2f5",
      minHeight: "100vh",
      fontFamily: "Arial, sans-serif",
    },
    navbar: {
      backgroundColor: "#1e40af",
      padding: "15px 0",
      width: "100%",
      position: "relative",
      marginBottom: "40px",
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    },
    navContent: {
      maxWidth: "1400px",
      margin: "0 auto",
      padding: "0 20px",
    },
    titleContainer: {
      display: "flex",
      flexDirection: "column",
    },
    mainTitle: {
      color: "white",
      fontSize: "28px",
      fontWeight: "bold",
      marginBottom: "8px",
    },
    subTitle: {
      color: "#93c5fd",
      fontSize: "18px",
      fontWeight: "500",
    },
    gridContainer: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: "20px",
      maxWidth: "1400px",
      margin: "0 auto",
      padding: "0 15px",
    },
    card: {
      backgroundColor: "white",
      borderRadius: "10px",
      padding: "20px",
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
      height: "fit-content",
    },
    fullWidthCard: {
      backgroundColor: "white",
      borderRadius: "10px",
      padding: "20px",
      marginTop: "20px",
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
      gridColumn: "1 / -1",
    },
    cardHeader: {
      fontSize: "20px",
      color: "#2d3748",
      marginBottom: "16px",
      fontWeight: "bold",
      borderBottom: "1px solid #e2e8f0",
      paddingBottom: "8px",
    },
    healthContainer: {
      marginBottom: "20px",
    },
    healthStatus: {
      display: "flex",
      alignItems: "center",
      marginBottom: "15px",
      fontSize: "16px",
      padding: "10px",
      backgroundColor: "#f7fafc",
      borderRadius: "6px",
    },
    healthIcon: {
      marginRight: "8px",
      fontSize: "20px",
    },
    modelsList: {
      marginTop: "25px",
      padding: "15px",
      backgroundColor: "#f7fafc",
      borderRadius: "6px",
    },
    modelsHeader: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "15px",
    },
    refreshButton: {
      backgroundColor: "#4299e1",
      color: "white",
      padding: "8px 16px",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "12px",
      transition: "background-color 0.2s",
    },
    refreshButtonSpinning: {
      backgroundColor: "#90cdf4",
      cursor: "not-allowed",
    },
    modelItem: {
      padding: "8px",
      borderBottom: "1px solid #e2e8f0",
      fontSize: "14px",
      display: "flex",
      justifyContent: "space-between",
    },
    modelVersion: {
      color: "#718096",
      fontSize: "12px",
    },
    selectContainer: {
      marginBottom: "16px",
      width: "100%",
    },
    select: {
      width: "100%",
      padding: "12px",
      borderRadius: "6px",
      border: "1px solid #cbd5e0",
      backgroundColor: "white",
      fontSize: "14px",
      marginBottom: "15px",
      boxSizing: "border-box",
    },
    buttonContainer: {
      width: "100%",
      display: "flex",
      flexDirection: "column",
      gap: "10px",
    },
    button: {
      backgroundColor: "#4299e1",
      color: "white",
      padding: "12px 24px",
      border: "none",
      borderRadius: "6px",
      cursor: "pointer",
      fontSize: "14px",
      fontWeight: "500",
      width: "100%",
      transition: "background-color 0.2s",
    },
    buttonDisabled: {
      backgroundColor: "#a0aec0",
      cursor: "not-allowed",
    },
    fileInput: {
      display: "none",
    },
    fileLabel: {
      display: "block",
      padding: "12px 24px",
      backgroundColor: "#48bb78",
      color: "white",
      borderRadius: "6px",
      cursor: "pointer",
      fontSize: "14px",
      marginBottom: "15px",
      textAlign: "center",
      width: "100%",
      boxSizing: "border-box",
    },
    fileName: {
      color: "#4a5568",
      fontSize: "14px",
      marginBottom: "15px",
      display: "block",
      wordBreak: "break-all",
    },
    error: {
      color: "#e53e3e",
      marginTop: "10px",
      fontSize: "14px",
      padding: "10px",
      backgroundColor: "#fed7d7",
      borderRadius: "6px",
    },
    success: {
      color: "#38a169",
      marginTop: "10px",
      fontSize: "14px",
      padding: "10px",
      backgroundColor: "#c6f6d5",
      borderRadius: "6px",
    },
    detailsRow: {
      marginBottom: "12px",
      fontSize: "16px",
      display: "flex",
      justifyContent: "space-between",
      borderBottom: "1px solid #e2e8f0",
      paddingBottom: "8px",
    },
    detailLabel: {
      color: "#4a5568",
      fontWeight: "bold",
    },
    resultsContainer: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
      gap: "20px",
      padding: "20px",
    },
    gaugeContainer: {
      textAlign: "center",
      padding: "20px",
    },
    scoreValue: {
      fontSize: "24px",
      fontWeight: "bold",
      color: "#2d3748",
      marginTop: "15px",
    },
    resultDetails: {
      backgroundColor: "#f7fafc",
      padding: "20px",
      borderRadius: "8px",
    },
    legend: {
      marginTop: "20px",
      padding: "15px",
      backgroundColor: "#f8fafc",
      borderRadius: "8px",
      border: "1px solid #e2e8f0",
    },
    legendTitle: {
      fontSize: "16px",
      fontWeight: "bold",
      color: "#2d3748",
      marginBottom: "12px",
    },
    legendGrid: {
      display: "flex",
      justifyContent: "space-between",
      flexWrap: "wrap",
      gap: "10px",
    },
    legendItem: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      padding: "8px 16px",
      borderRadius: "6px",
      backgroundColor: "white",
      boxShadow: "0 1px 2px rgba(0, 0, 0, 0.05)",
      minWidth: "fit-content",
      flex: "0 1 auto",
    },
    legendDot: {
      width: "12px",
      height: "12px",
      borderRadius: "50%",
      flexShrink: 0,
    },
    tab: {
      padding: "10px 20px",
      cursor: "pointer",
      fontWeight: "bold",
    },
    activeTab: {
      borderBottom: "2px solid #1e40af",
      color: "#1e40af",
    },
  };

  // function for the colors
  const getWeakestLinkColor = (link) => {
    if (["ForwardHead", "ExcessiveForwardLean"].includes(link))
      return "#e53e3e"; // Rosso
    if (["KneeMovesInward", "ArmFallForward"].includes(link)) return "#dd6b20"; // Arancione
    return "#38a169"; // Verde
  };

  const handleTabChange = (tab) => {
    // Reset common states
    setError("");
    setSuccessMessage("");
    setFile(null);

    // Reset results based on tab
    if (tab === "Regression") {
      setScore(0);
      setCategory("");
      setModelName("");
    } else {
      setWeakestlink(null);
      setModelName("");
    }

    // Set the active tab
    setActiveTab(tab);
  };

  return (
    <div
      style={{ display: "flex", flexDirection: "column", alignItems: "center" }}
    >
      <nav style={dashboardStyles.navbar}>
        <div style={dashboardStyles.navContent}>
          <div style={dashboardStyles.titleContainer}>
            <h1 style={dashboardStyles.mainTitle}>
              Project In Data Intensive Systems
            </h1>
            <h2 style={dashboardStyles.subTitle}>
              Automated Movement Assessment
            </h2>
          </div>
        </div>
      </nav>

      {/* Tab Navigation */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          borderBottom: "2px solid #e2e8f0",
          marginBottom: "20px",
          width: "100%",
        }}
      >
        <div
          style={{
            ...dashboardStyles.tab,
            ...(activeTab === "Regression" ? dashboardStyles.activeTab : {}),
          }}
          onClick={() => handleTabChange("Regression")}
        >
          Regression
        </div>
        <div
          style={{
            ...dashboardStyles.tab,
            ...(activeTab === "Classification"
              ? dashboardStyles.activeTab
              : {}),
          }}
          onClick={() => handleTabChange("Classification")}
        >
          Classification
        </div>
      </div>

      {/* tab Regression */}
      {activeTab === "Regression" && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: "20px",
            maxWidth: "1200px",
            width: "100%",
          }}
        >
          {/* Health Status Card */}
          <div style={dashboardStyles.card}>
            <div style={dashboardStyles.healthContainer}>
              <h3 style={dashboardStyles.cardHeader}>System Health</h3>
              <div style={dashboardStyles.healthStatus}>
                <span style={dashboardStyles.healthIcon}>
                  {healthStatus === "healthy" ? "✅" : "❌"}
                </span>
                <span>Status: {healthStatus}</span>
              </div>
              <div style={dashboardStyles.detailsRow}>
                <span style={dashboardStyles.detailLabel}>Models Loaded:</span>
                <span>{modelsLoaded}</span>
              </div>
            </div>
            <div style={dashboardStyles.modelsList}>
              <div style={dashboardStyles.modelsHeader}>
                <h4 style={dashboardStyles.detailLabel}>Available Models</h4>
                <button
                  onClick={handleRefreshModels}
                  disabled={refreshing}
                  style={{
                    ...dashboardStyles.refreshButton,
                    ...(refreshing
                      ? dashboardStyles.refreshButtonSpinning
                      : {}),
                  }}
                >
                  {refreshing ? "Refreshing..." : "Refresh Models"}
                </button>
              </div>
              {models.map((model) => (
                <div key={model.name} style={dashboardStyles.modelItem}>
                  <span>{model.name}</span>
                  <span style={dashboardStyles.modelVersion}>
                    v{model.version}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Make Prediction Card */}
          <div style={dashboardStyles.card}>
            <h3 style={dashboardStyles.cardHeader}>Make Prediction</h3>
            <div style={dashboardStyles.selectContainer}>
              <select
                style={dashboardStyles.select}
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="">Select a Model</option>
                {models.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name} (v{model.version})
                  </option>
                ))}
              </select>
            </div>
            <div style={dashboardStyles.buttonContainer}>
              <label style={dashboardStyles.fileLabel}>
                Choose File
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setFile(e.target.files[0])}
                  style={dashboardStyles.fileInput}
                />
              </label>
              {file && (
                <span style={dashboardStyles.fileName}>
                  Selected: {file.name}
                </span>
              )}
              <button
                onClick={() => handleFileUpload("predict")}
                disabled={loading || !selectedModel}
                style={{
                  ...dashboardStyles.button,
                  ...(loading || !selectedModel
                    ? dashboardStyles.buttonDisabled
                    : {}),
                }}
              >
                {loading ? "Processing..." : "Get Prediction"}
              </button>
            </div>
            {error && <div style={dashboardStyles.error}>{error}</div>}
            {successMessage && (
              <div style={dashboardStyles.success}>{successMessage}</div>
            )}
          </div>
        </div>
      )}

      {/* Prediction Results Card */}
      {activeTab === "Regression" && score > 0 && (
        <div
          style={{
            ...dashboardStyles.fullWidthCard,
            width: "100%", // Changed from grid-column to explicit width
            maxWidth: "1200px",
            margin: "20px auto 0", // Added margin to separate from cards above
          }}
        >
          <div style={dashboardStyles.fullWidthCard}>
            <h3 style={dashboardStyles.cardHeader}>Prediction Results</h3>
            <div style={dashboardStyles.resultsContainer}>
              <div style={dashboardStyles.gaugeContainer}>
                <GaugeChart
                  id="gauge-chart"
                  nrOfLevels={4}
                  colors={["#ff7675", "#ffeaa7", "#74b9ff", "#55efc4"]}
                  percent={score}
                  arcWidth={0.3}
                  textColor="#2d3436"
                  formatTextValue={(value) => `${value}%`}
                  arcsLength={[0.39, 0.3, 0.2, 0.11]}
                />
                <div style={dashboardStyles.scoreValue}>
                  Raw Score: {score.toFixed(2)}
                </div>
              </div>

              <div style={dashboardStyles.resultDetails}>
                <div style={dashboardStyles.detailsRow}>
                  <span style={dashboardStyles.detailLabel}>Model:</span>
                  <span>{modelName}</span>
                </div>
                <div style={dashboardStyles.detailsRow}>
                  <span style={dashboardStyles.detailLabel}>Category:</span>
                  <span>{category}</span>
                </div>
                <div style={dashboardStyles.detailsRow}>
                  <span style={dashboardStyles.detailLabel}>
                    Prediction Score:
                  </span>
                  <span>{score}</span>
                </div>

                {/* Score Legend */}
                <div style={dashboardStyles.legend}>
                  <div style={dashboardStyles.legendTitle}>
                    Score Categories
                  </div>
                  <div style={dashboardStyles.legendGrid}>
                    <div style={dashboardStyles.legendItem}>
                      <div
                        style={{
                          ...dashboardStyles.legendDot,
                          backgroundColor: "#ff7675",
                        }}
                      ></div>
                      <span style={dashboardStyles.legendText}>0-39: Bad</span>
                    </div>
                    <div style={dashboardStyles.legendItem}>
                      <div
                        style={{
                          ...dashboardStyles.legendDot,
                          backgroundColor: "#ffeaa7",
                        }}
                      ></div>
                      <span style={dashboardStyles.legendText}>
                        40-69: Good
                      </span>
                    </div>
                    <div style={dashboardStyles.legendItem}>
                      <div
                        style={{
                          ...dashboardStyles.legendDot,
                          backgroundColor: "#74b9ff",
                        }}
                      ></div>
                      <span style={dashboardStyles.legendText}>
                        70-89: Great
                      </span>
                    </div>
                    <div style={dashboardStyles.legendItem}>
                      <div
                        style={{
                          ...dashboardStyles.legendDot,
                          backgroundColor: "#55efc4",
                        }}
                      ></div>
                      <span style={dashboardStyles.legendText}>
                        90-100: Excellent
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Contenuto del tab Classification */}

      {activeTab === "Classification" && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: "20px",
            maxWidth: "1200px",
            width: "100%",
          }}
        >
          {/* Classification Models List (Left Side) */}
          <div style={dashboardStyles.card}>
            <h3 style={dashboardStyles.cardHeader}>Classification Models</h3>
            <div style={dashboardStyles.modelsHeader}>
              <h4 style={dashboardStyles.detailLabel}>Available Models</h4>
              <button
                onClick={handleRefreshCategorizingModels}
                disabled={refreshing}
                style={{
                  ...dashboardStyles.refreshButton,
                  ...(refreshing ? dashboardStyles.refreshButtonSpinning : {}),
                }}
              >
                {refreshing ? "Refreshing..." : "Refresh Models"}
              </button>
            </div>
            <div style={dashboardStyles.modelsList}>
              {categorizingModels.map((model) => (
                <div key={model.name} style={dashboardStyles.modelItem}>
                  <span>{model.name}</span>
                  <span style={dashboardStyles.modelVersion}>
                    v{model.version}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Model Selection, File Upload & Classification (Right Side) */}
          <div style={dashboardStyles.card}>
            <h3 style={dashboardStyles.cardHeader}>Select Model & Classify</h3>
            <div style={dashboardStyles.selectContainer}>
              <select
                style={dashboardStyles.select}
                value={selectedCategorizer}
                onChange={(e) => setSelectedCategorizer(e.target.value)}
              >
                <option value="">Select a Model</option>
                {categorizingModels.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name} (v{model.version})
                  </option>
                ))}
              </select>
            </div>
            <div style={dashboardStyles.buttonContainer}>
              <label style={dashboardStyles.fileLabel}>
                Choose File
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setFile(e.target.files[0])}
                  style={dashboardStyles.fileInput}
                />
              </label>
              {file && (
                <span style={dashboardStyles.fileName}>
                  Selected: {file.name}
                </span>
              )}
              <button
                onClick={() => handleFileUpload("classify-weakest-link")}
                disabled={loading || !selectedCategorizer}
                style={{
                  ...dashboardStyles.button,
                  ...(loading || !selectedCategorizer
                    ? dashboardStyles.buttonDisabled
                    : {}),
                }}
              >
                {loading ? "Classifying..." : "Classify Weakest Link"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Classification Results Card */}
      {activeTab === "Classification" && weakestLink && (
        <div
          style={{
            ...dashboardStyles.fullWidthCard,
            width: "100%", // Full width to match other cards
            maxWidth: "1200px",
            margin: "20px auto 0", // Consistent margin with regression results
            gridColumn: "1 / -1", // Ensure it spans full width in grid
          }}
        >
          <h3 style={dashboardStyles.cardHeader}>Classification Results</h3>
          <div
            style={{
              padding: "30px", // Reduced padding for better proportions
            }}
          >
            <div style={dashboardStyles.resultDetails}>
              <div style={dashboardStyles.detailsRow}>
                <span style={dashboardStyles.detailLabel}>Model:</span>
                <span>{modelName}</span>
              </div>
              <div style={dashboardStyles.detailsRow}>
                <span style={dashboardStyles.detailLabel}>Weakest Link:</span>
                <span
                  style={{
                    fontWeight: "bold",
                    color: getWeakestLinkColor(weakestLink),
                    fontSize: "18px",
                  }}
                >
                  {weakestLink.replace(/([A-Z])/g, " $1")}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
