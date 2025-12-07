import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "../App";

vi.mock("../lib/ipc", () => ({
  runTraining: vi.fn().mockResolvedValue("job-1"),
  stopTraining: vi.fn().mockResolvedValue(undefined),
  fetchLogs: vi.fn().mockResolvedValue("job started"),
}));

describe("Desktop App", () => {
  it("gates access with auth token", () => {
    render(<App />);
    expect(screen.getByText(/Auth Required/)).toBeInTheDocument();

    const input = screen.getByLabelText(/Auth token/i);
    fireEvent.change(input, { target: { value: "token" } });
    fireEvent.click(screen.getByText(/Continue/));

    expect(screen.getByText(/Ministral Desktop/)).toBeInTheDocument();
  });

  it("renders controls and starts a job", async () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText(/Auth token/i), { target: { value: "token" } });
    fireEvent.click(screen.getByText(/Continue/));

    fireEvent.click(screen.getByText(/Run/));

    await waitFor(() => {
      expect(screen.getByTestId("logs").textContent).toContain("Starting job");
    });
    expect(screen.getByText(/Dataset/)).toBeInTheDocument();
  });
});

