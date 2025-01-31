import React, { useState } from "react";
import GaugeChart from "react-gauge-chart";

function App() {
  const [score, setScore] = useState(0.9); // initial normalized score (0-1)
  const [category, setCategory] = useState(""); // state for the category
  const [modelVersion, setModelVersion] = useState(""); // state for the model version
  const [file, setFile] = useState(null); // state for the uploaded file
  const [loading, setLoading] = useState(false); // state for loading

  // Function to handle the run button
  const handleRun = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/models");
      if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log(data);

      // Update the state with the received data
      setScore(data.score / 100); // Normalize the score (0-1) 
      setCategory(data.category);
      setModelVersion(data.model_version);

      alert(JSON.stringify(data));
    } catch (error) {
      console.error("Error during prediction:", error);
      alert(error);
    } finally {
      setLoading(false);
    }
  };

  // Function to handle the file upload
  const handleFileUpload = async () => {
    if (!file) {
      alert("Please upload a file!");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict/model_name", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log(data); // Log the response data
      alert(`Category: ${data.category}, Score: ${data.score}`);
    } catch (error) {
      console.error("Error during file upload", error);
      alert(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px", fontFamily: "Arial, sans-serif", backgroundColor: "#f4f4f4" }}>
      <header style={{ marginBottom: "20px", padding: "10px", backgroundColor: "#6200ea", color: "white", borderRadius: "10px" }}>
        <h1>Project: Data Intensive System</h1>
        <p style={{ fontStyle: "italic" }}>Automated Movement Assessment</p>
      </header>

      <div style={{ margin: "20px auto", maxWidth: "600px", textAlign: "center", backgroundColor: "white", padding: "20px", borderRadius: "10px", boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)" }}>
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

        <div style={{ marginTop: "20px", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <label
            style={{
              margin: "10px",
              padding: "10px 20px",
              backgroundColor: "#6c757d",
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
              style={{ display: "none" }} // Nasconde l'input predefinito
            />
          </label>
          <div style={{ marginLeft: "10px" }}>
            {file ? (
              <span>{file.name}</span> // Mostra il nome del file selezionato
            ) : (
              <span>No file selected</span> // Messaggio quando nessun file è selezionato
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

        {/* Display category and model version */}
        {category && (
          <div style={{ marginTop: "20px", fontSize: "18px" }}>
            <p><strong>Category:</strong> {category}</p>
            <p><strong>Model Version:</strong> {modelVersion}</p>
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