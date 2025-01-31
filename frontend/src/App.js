import React, { useState } from "react";
import GaugeChart from "react-gauge-chart";

function App() {
  const [score, setScore] = useState(0.5); // Valore iniziale normalizzato (0-1)

  const handleRun = () => {
    const newScore = Math.random(); // Simula un nuovo punteggio casuale
    setScore(newScore);
  };

  const handleUpdate = () => {
    alert("Caricamento dati... (da implementare)");
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
          nrOfLevels={20}
          colors={["#FF5F6D", "#FFC371", "#28a745"]}
          percent={score}
          arcWidth={0.3}
          style={{ width: "300px", height: "150px", margin: "0 auto" }}
          textColor="#000000"
          formatTextValue={(value) => `${Math.round(value * 100)}`}
        />
        <div style={{ marginTop: "20px" }}>
          <button
            onClick={handleRun}
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
            onMouseOver={(e) => (e.target.style.backgroundColor = "#0277bd")}
            onMouseOut={(e) => (e.target.style.backgroundColor = "#0288d1")}
          >
            Run
          </button>
          <button
            onClick={handleUpdate}
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
            onMouseOver={(e) => (e.target.style.backgroundColor = "#5a6268")}
            onMouseOut={(e) => (e.target.style.backgroundColor = "#6c757d")}
          >
            Update
          </button>
        </div>
      </div>
      <footer style={{ marginTop: "20px", color: "#555" }}>
        <p>&copy; 2025 Movement Assessment System</p>
      </footer>
    </div>
  );
}

export default App;
