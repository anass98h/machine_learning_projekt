import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "./App";
import axios from "axios";

// Mock delle chiamate API
jest.mock("axios");

describe("App Component", () => {
  beforeEach(() => {
    axios.get.mockResolvedValue({ data: [] });
    axios.post.mockResolvedValue({ data: { message: "Success" } });
  });

  test("renders the application", () => {
    render(<App />);
    expect(screen.getByText(/Project In Data Intensive Systems/i)).toBeInTheDocument();
  });

  test("switches between Regression and Classification tabs", () => {
    render(<App />);

    // Controlla che il tab iniziale sia Regression
    expect(screen.getByText("Regression")).toHaveStyle("border-bottom: 2px solid #1e40af");

    // Clicca su "Classification"
    fireEvent.click(screen.getByText("Classification"));

    // Controlla che il tab attivo sia cambiato
    expect(screen.getByText("Classification")).toHaveStyle("border-bottom: 2px solid #1e40af");
  });

  test("fetches models and health status on mount", async () => {
    render(<App />);

    // Aspetta che le chiamate API siano state effettuate
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith("http://localhost:8000/models");
      expect(axios.get).toHaveBeenCalledWith("http://localhost:8000/health");
      expect(axios.get).toHaveBeenCalledWith("http://localhost:8000/categorizing-models");
    });
  });

  test("allows selecting a model", async () => {
    axios.get.mockResolvedValueOnce({
      data: [{ name: "TestModel", version: "1.0" }],
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Select a Model")).toBeInTheDocument();
    });

    // Recupera il select
    const select = screen.getByRole("combobox");

    // Aspetta che le opzioni vengano caricate
    await waitFor(() => {
      expect(select.options.length).toBeGreaterThan(1);
    });

    // Simula la selezione del modello
    fireEvent.change(select, { target: { value: "TestModel" } });

    // Controlla che il valore sia stato aggiornato
    await waitFor(() => {
      expect(select.value).toBe("TestModel");
    });
  });


  test("handles file upload", async () => {
    render(<App />);

    const file = new File(["sample content"], "test.csv", { type: "text/csv" });

    // Simula l'upload del file
    fireEvent.change(screen.getByLabelText(/Choose File/i), {
      target: { files: [file] },
    });

    await waitFor(() => {
      expect(screen.getByText(/Selected: test.csv/i)).toBeInTheDocument();
    });
  });

  test("disables 'Get Prediction' button until model and file are selected", async () => {
    axios.get.mockResolvedValueOnce({
      data: [{ name: "TestModel", version: "1.0" }],
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Select a Model/i)).toBeInTheDocument();
    });

    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "TestModel" } });

    const file = new File(["sample content"], "test.csv", { type: "text/csv" });
    fireEvent.change(screen.getByLabelText(/Choose File/i), {
      target: { files: [file] },
    });

    await waitFor(() => {
      const button = screen.getByRole("button", { name: /get prediction/i });
      expect(button).toBeEnabled();
    });
  });

  test("calls API when refresh models is clicked", async () => {
    render(<App />);
    const refreshButton = screen.getByText("Refresh Models");

    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith("http://localhost:8000/refresh-models");
    });
  });
});
