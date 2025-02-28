import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "./App";
import axios from "axios";

// Mock API calls
jest.mock("axios");

describe("App Component", () => {
  // Reset mock API responses before each test
  beforeEach(() => {
    jest.useFakeTimers(); // Mocks timers to prevent hanging tests
    axios.get.mockResolvedValue({ data: [] });
    axios.post.mockResolvedValue({ data: { message: "Success" } });
  });
  //run after each tests to clean up
  afterEach(() => {
    jest.clearAllMocks(); // Clears all mocks after each test
    jest.restoreAllMocks(); // Restores original implementations
    jest.useRealTimers(); // Ensures there are no pending timers
    // Ensure all pending API calls are resolved
    return new Promise((resolve) => setTimeout(resolve, 0));
  });


  test("renders the application", () => {
    render(<App />);
    // Check if the application title is displayed
    expect(screen.getByText(/Project In Data Intensive Systems/i)).toBeInTheDocument();
  });

  test("switches between Regression and Classification tabs", () => {
    render(<App />);

    // Verify that "Regression" is the default active tab
    expect(screen.getByText("Regression")).toHaveStyle("border-bottom: 2px solid #1e40af");

    // Click on "Classification" tab
    fireEvent.click(screen.getByText("Classification"));

    // Ensure "Classification" tab is now active
    expect(screen.getByText("Classification")).toHaveStyle("border-bottom: 2px solid #1e40af");
  });

  test("fetches models and health status on mount", async () => {
    render(<App />);

    // Wait for API calls to be triggered on component mount
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith("http://localhost:8000/models");
      expect(axios.get).toHaveBeenCalledWith("http://localhost:8000/health");
      expect(axios.get).toHaveBeenCalledWith("http://localhost:8000/categorizing-models");
    });
  });

  test("allows selecting a model", async () => {
    // Mock API response with a test model
    axios.get.mockResolvedValueOnce({
      data: [{ name: "TestModel", version: "1.0" }],
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Select a Model")).toBeInTheDocument();
    });

    // Get the model selection dropdown
    const select = screen.getByRole("combobox");

    // Wait until models are loaded in the dropdown
    await waitFor(() => {
      expect(select.options.length).toBeGreaterThan(1);
    });

    // Simulate selecting a model
    fireEvent.change(select, { target: { value: "TestModel" } });

    // Verify that the selected model value is updated
    await waitFor(() => {
      expect(select.value).toBe("TestModel");
    });
  });

  test("handles file upload", async () => {
    render(<App />);

    const file = new File(["sample content"], "test.csv", { type: "text/csv" });

    // Simulate file upload
    fireEvent.change(screen.getByLabelText(/Choose File/i), {
      target: { files: [file] },
    });

    // Verify that the uploaded file name is displayed
    await waitFor(() => {
      expect(screen.getByText(/Selected: test.csv/i)).toBeInTheDocument();
    });
  });

  test("disables 'Get Prediction' button until model and file are selected", async () => {
    // Mock API response with a test model
    axios.get.mockResolvedValueOnce({
      data: [{ name: "TestModel", version: "1.0" }],
    });

    render(<App />);

    // Wait for the model selection dropdown to be available
    await waitFor(() => {
      expect(screen.getByText(/Select a Model/i)).toBeInTheDocument();
    });

    // Get the select dropdown
    const select = screen.getByRole("combobox");

    // Wait for the options to be populated before interacting
    await waitFor(() => {
      expect(select.options.length).toBeGreaterThan(1);
    });

    // Simulate selecting the model
    fireEvent.change(select, { target: { value: "TestModel" } });

    // Wait for the state update to reflect in the dropdown
    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveValue("TestModel");
    });

    // Simulate file upload
    const file = new File(["sample content"], "test.csv", { type: "text/csv" });
    fireEvent.change(screen.getByLabelText(/Choose File/i), {
      target: { files: [file] },
    });

    // Ensure file selection is reflected in the UI
    await waitFor(() => {
      expect(screen.getByText(/Selected: test.csv/i)).toBeInTheDocument();
    });

    // Wait for the button to become enabled
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /get prediction/i })).not.toBeDisabled();
    });
  });


  test("calls API when refresh models is clicked", async () => {
    render(<App />);
    const refreshButton = screen.getByText("Refresh Models");

    // Click the "Refresh Models" button
    fireEvent.click(refreshButton);

    // Verify that the refresh API call was made
    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith("http://localhost:8000/refresh-models");
    });
  });
});
