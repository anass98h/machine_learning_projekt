import React, { useState } from "react";
import axios from "axios";
import GaugeChart from "react-gauge-chart";

function App() {
  const [score, setScore] = useState(0.9); // initial normalized score (0-1)
  const [category, setCategory] = useState(""); // state for the category
  const [modelName, setModelName] = useState(""); // state for the model version
  const [file, setFile] = useState(null); // state for the uploaded file
  const [loading, setLoading] = useState(false); // state for loading
  const [models, setModels] = useState([]);

  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  // Update handleRun function to handle the models list
  const handleRun = async () => {
    setLoading(true);
    setError("");
    setSuccessMessage("");
    try {
      const response = await axios.get("http://localhost:8000/models");
      const data = response.data;
      console.log(data);

      // Update the models list
      setModels(data);
      setSuccessMessage("Available models loaded successfully!");
    } catch (error) {
      console.error("Error fetching models:", error);
      setError(
        "An error occurred while fetching the models. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  // Update handleFileUpload to handle prediction results
  const handleFileUpload = async () => {
    if (!file) {
      setError("Please upload a file!");
      return;
    }

    setLoading(true);
    setError("");
    setSuccessMessage("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://localhost:8000/predict/linear_regression_model_v1",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const data = response.data;
      console.log(data);

      // Update states with prediction results
      setCategory(data.category);
      setScore(data.score / 100);
      setModelName(data.model_name);
      setSuccessMessage("Prediction completed successfully!");
    } catch (error) {
      console.error("Error during file upload", error);
      setError("An error occurred during file upload. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        textAlign: "center",
        padding: "20px",
        fontFamily: "Arial, sans-serif",
        backgroundColor: "#f4f4f4",
      }}
    >
      <header
        style={{
          marginBottom: "20px",
          padding: "10px",
          backgroundColor: "#6200ea",
          color: "white",
          borderRadius: "10px",
        }}
      >
        <h1>Project: Data Intensive System</h1>
        <p style={{ fontStyle: "italic" }}>Automated Movement Assessment</p>
      </header>

      <div
        style={{
          margin: "20px auto",
          maxWidth: "600px",
          textAlign: "center",
          backgroundColor: "white",
          padding: "20px",
          borderRadius: "10px",
          boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
        }}
      >
        <h2 style={{ color: "#6200ea" }}>Movement Score</h2>

        <GaugeChart
          id="gauge-chart"
          nrOfLevels={5}
          colors={["#FF5F6D", "#FFC371", "#28a745"]}
          percent={score}
          arcWidth={0.3}
          style={{ width: "300px", height: "150px", margin: "0 auto" }}
          textColor="#000000"
        />

        <div
          style={{
            marginTop: "20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <label
            style={{
              margin: "10px",
              padding: "10px 20px",
              backgroundColor: "#0288d1",
              color: "white",
              cursor: "pointer",
              borderRadius: "5px",
              border: "none",
              fontSize: "16px",
              transition: "background-color 0.3s",
            }}
          >
            Choose File
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0])}
              style={{ display: "none" }} // Hide the default input
            />
          </label>
          <div style={{ marginLeft: "10px" }}>
            {file ? (
              <span>{file.name}</span> // Show selected file name
            ) : (
              <span>No file selected</span> // Message when no file is selected
            )}
          </div>
        </div>

        <div>
          <button
            onClick={handleRun}
            disabled={loading}
            style={{
              margin: "10px",
              padding: "10px 20px",
              backgroundColor: loading ? "#ccc" : "#0288d1",
              color: "white",
              cursor: loading ? "not-allowed" : "pointer",
              borderRadius: "5px",
              border: "none",
              fontSize: "16px",
              transition: "background-color 0.3s",
            }}
          >
            {loading ? "Loading..." : "Run"}
          </button>

          <button
            onClick={handleFileUpload}
            disabled={loading}
            style={{
              margin: "10px",
              padding: "10px 20px",
              backgroundColor: loading ? "#ccc" : "#6c757d",
              color: "white",
              cursor: loading ? "not-allowed" : "pointer",
              borderRadius: "5px",
              border: "none",
              fontSize: "16px",
              transition: "background-color 0.3s",
            }}
          >
            {loading ? "Loading..." : "Upload"}
          </button>
        </div>

        {/* Display success or error messages */}
        {error && (
          <div style={{ marginTop: "20px", color: "red", fontWeight: "bold" }}>
            <p>{error}</p>
          </div>
        )}
        {successMessage && (
          <div
            style={{ marginTop: "20px", color: "green", fontWeight: "bold" }}
          >
            <p>{successMessage}</p>
          </div>
        )}

        {models.length > 0 && (
          <div
            style={{
              marginTop: "20px",
              padding: "15px",
              backgroundColor: "#f8f9fa",
              borderRadius: "8px",
            }}
          >
            <h3 style={{ marginBottom: "15px", color: "#6200ea" }}>
              Available Models
            </h3>
            <div
              style={{ display: "flex", flexDirection: "column", gap: "10px" }}
            >
              {models.map((model, index) => (
                <div
                  key={index}
                  style={{
                    padding: "10px",
                    backgroundColor: "white",
                    borderRadius: "5px",
                    boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
                  }}
                >
                  <p>
                    <strong>Name:</strong> {model.name}
                  </p>
                  <p>
                    <strong>Version:</strong> {model.version}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Display prediction results */}
        {category && (
          <div
            style={{
              marginTop: "20px",
              display: "flex",
              justifyContent: "space-around",
              fontSize: "18px",
              width: "100%",
            }}
          >
            <div
              style={{
                flex: 1,
                margin: "0 10px",
                padding: "15px",
                backgroundColor: "#f8f9fa",
                borderRadius: "8px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <strong style={{ marginBottom: "8px" }}>Category</strong>
              <span>{category}</span>
            </div>
            <div
              style={{
                flex: 1,
                margin: "0 10px",
                padding: "15px",
                backgroundColor: "#f8f9fa",
                borderRadius: "8px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <strong style={{ marginBottom: "8px" }}>Score</strong>
              <span>{(score * 100).toFixed(1)}%</span>
            </div>
            <div
              style={{
                flex: 1,
                margin: "0 10px",
                padding: "15px",
                backgroundColor: "#f8f9fa",
                borderRadius: "8px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <strong style={{ marginBottom: "8px" }}>Model Name</strong>
              <span>{modelName}</span>
            </div>
          </div>
        )}
      </div>

      <footer style={{ marginTop: "20px", color: "#555" }}>
        <p>&copy; 2025 Movement Assessment System</p>
      </footer>
    </div>
  );
}

export default App;