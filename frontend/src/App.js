import React, { useState } from "react";
import GaugeChart from "react-gauge-chart";

function App() {
  const [score, setScore] = useState(0.9); // Valore iniziale normalizzato (0-1)
  const [category, setCategory] = useState(""); // Stato per la categoria
  const [modelVersion, setModelVersion] = useState(""); // Stato per la versione del modello
  const [file, setFile] = useState(null); // Stato per il file caricato
  const [loading, setLoading] = useState(false); // Stato per il caricamento

  // Funzione per gestire la predizione (Run)
  const handleRun = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/models");
      if (!response.ok) {
        throw new Error(`Errore HTTP! Stato: ${response.status}`);
      }

      const data = await response.json();
      console.log(data); // Aggiungi questa riga per vedere cosa ricevi

      // Imposta i dati ricevuti
      setScore(data.score / 100); // Normalizza il punteggio a un valore tra 0 e 1
      setCategory(data.category);
      setModelVersion(data.model_version);

      alert(JSON.stringify(data));
    } catch (error) {
      console.error("Errore durante la predizione:", error);
      alert(error);
    } finally {
      setLoading(false);
    }
  };

  // Funzione per gestire l'upload del file
  const handleFileUpload = async () => {
    if (!file) {
      alert("Per favore, carica un file!");
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
        throw new Error(`Errore HTTP! Stato: ${response.status}`);
      }

      const data = await response.json();
      console.log(data); // Logga la risposta per vedere cosa ricevi
      alert(`Categoria: ${data.category}, Punteggio: ${data.score}`);
    } catch (error) {
      console.error("Errore durante l'upload del file:", error);
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
        //formatTextValue={(value) => `${Math.round(value * 100)}%`}
        />

        <div style={{ marginTop: "20px" }}>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files[0])}
            style={{ margin: "10px" }}
          />

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
            {loading ? "Caricamento..." : "Run"}
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
            {loading ? "Caricamento..." : "Upload"}
          </button>
        </div>

        {/* Display category and model version */}
        {category && (
          <div style={{ marginTop: "20px", fontSize: "18px" }}>
            <p><strong>Categoria:</strong> {category}</p>
            <p><strong>Versione del modello:</strong> {modelVersion}</p>
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
