import { useEffect, useMemo, useState } from "react";
import { fetchLogs, runTraining, stopTraining } from "./lib/ipc";
import { AnnealingPreset, DashboardMetrics, JobState } from "./types";

const defaultPreset: AnnealingPreset = {
  name: "Default",
  datasetPath: "my_dataset.jsonl",
  outputDir: "ministral-3b-finetuned",
  cycles: 3,
  initialTemperature: 1.5,
  minTemperature: 0.3,
  coolingRate: 0.85,
  heatingRate: 1.05,
  stabilizationSteps: 1,
  quality: {
    minLength: 32,
    minCoherence: 0.4,
    minDiversity: 0.3,
    maxRepetition: 0.4,
  },
  enableAnnealing: true,
};

function usePresets() {
  const [presets, setPresets] = useState<AnnealingPreset[]>(() => {
    if (typeof window === "undefined") return [defaultPreset];
    const stored = window.localStorage.getItem("annealing-presets");
    if (stored) return JSON.parse(stored) as AnnealingPreset[];
    return [defaultPreset];
  });

  const save = (preset: AnnealingPreset) => {
    const next = [...presets.filter((p) => p.name !== preset.name), preset];
    setPresets(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("annealing-presets", JSON.stringify(next));
    }
  };

  return { presets, save };
}

export default function App() {
  const { presets, save } = usePresets();
  const [authToken, setAuthToken] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [form, setForm] = useState<AnnealingPreset>(defaultPreset);
  const [job, setJob] = useState<JobState>({ id: null, status: "idle" });
  const [logs, setLogs] = useState<string[]>(["Ready to run"]);
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    datasetSize: 0,
    avgQuality: 0,
    cyclesCompleted: 0,
    removedFlags: {},
  });

  useEffect(() => {
    if (job.status !== "running" || !job.id) return;
    const interval = setInterval(async () => {
      try {
        const next = await fetchLogs(job.id!);
        if (next) {
          setLogs((prev) => [...prev.slice(-400), next]);
        }
      } catch (err) {
        setLogs((prev) => [...prev, `log fetch failed: ${err}`]);
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [job]);

  const handleRun = async () => {
    setLogs((prev) => [...prev, "Starting job..."]);
    setJob({ id: null, status: "running" });
    try {
      const jobId = await runTraining({ ...form, authToken });
      setJob({ id: jobId, status: "running" });
      setLogs((prev) => [...prev, `Job ${jobId} started.`]);
    } catch (err) {
      setJob({ id: null, status: "error", lastMessage: String(err) });
      setLogs((prev) => [...prev, `Failed to start: ${err}`]);
    }
  };

  const handleStop = async () => {
    if (!job.id) return;
    try {
      await stopTraining(job.id);
      setJob({ id: null, status: "stopped" });
      setLogs((prev) => [...prev, "Job stopped"]);
    } catch (err) {
      setLogs((prev) => [...prev, `Failed to stop: ${err}`]);
    }
  };

  const handleSavePreset = () => {
    const name = prompt("Preset name", form.name) || form.name;
    save({ ...form, name });
    setLogs((prev) => [...prev, `Preset '${name}' saved.`]);
  };

  const handleLoadPreset = (name: string) => {
    const preset = presets.find((p) => p.name === name);
    if (preset) {
      setForm(preset);
      setLogs((prev) => [...prev, `Preset '${name}' loaded.`]);
    }
  };

  const derivedStats = useMemo(
    () => ({
      datasetSize: metrics.datasetSize || 0,
      quality: metrics.avgQuality || 0,
      cycles: metrics.cyclesCompleted || 0,
    }),
    [metrics]
  );

  if (!authenticated) {
    return (
      <div className="app-shell" style={{ display: "flex", placeItems: "center" }}>
        <div className="panel" style={{ width: 360, margin: "0 auto" }}>
          <h3>Auth Required</h3>
          <p>Enter any non-empty token to unlock the desktop UI.</p>
          <label className="controls-grid">
            <span>Auth token</span>
            <input
              aria-label="Auth token"
              value={authToken}
              onChange={(e) => setAuthToken(e.target.value)}
              style={{ padding: 8, borderRadius: 6, border: "1px solid #1f2937" }}
            />
          </label>
          <button
            className="btn"
            style={{ marginTop: 12, width: "100%" }}
            onClick={() => setAuthenticated(Boolean(authToken.trim()))}
          >
            Continue
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="header">
        <div>
          <div style={{ fontWeight: 700 }}>Ministral Desktop</div>
          <div style={{ fontSize: 12, color: "#9ca3af" }}>
            Self-annealing controls, presets, and live logs
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn" onClick={handleRun} disabled={job.status === "running"}>
            {job.status === "running" ? "Running..." : "Run"}
          </button>
          <button className="btn secondary" onClick={handleStop} disabled={!job.id}>
            Stop
          </button>
          <button className="btn secondary" onClick={handleSavePreset}>
            Save preset
          </button>
        </div>
      </header>

      <aside className="sidebar">
        <h4>Controls</h4>
        <div className="controls-grid">
          <label>
            Dataset path
            <input
              value={form.datasetPath}
              onChange={(e) => setForm({ ...form, datasetPath: e.target.value })}
            />
          </label>
          <label>
            Output dir
            <input
              value={form.outputDir}
              onChange={(e) => setForm({ ...form, outputDir: e.target.value })}
            />
          </label>
          <label>
            Enable annealing
            <input
              type="checkbox"
              checked={form.enableAnnealing}
              onChange={(e) => setForm({ ...form, enableAnnealing: e.target.checked })}
            />
          </label>
          <label>
            Cycles
            <input
              type="number"
              min={1}
              value={form.cycles}
              onChange={(e) => setForm({ ...form, cycles: Number(e.target.value) })}
            />
          </label>
          <label>
            Initial temperature
            <input
              type="number"
              step={0.1}
              value={form.initialTemperature}
              onChange={(e) =>
                setForm({ ...form, initialTemperature: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Min temperature
            <input
              type="number"
              step={0.1}
              value={form.minTemperature}
              onChange={(e) => setForm({ ...form, minTemperature: Number(e.target.value) })}
            />
          </label>
          <label>
            Cooling rate
            <input
              type="number"
              step={0.01}
              value={form.coolingRate}
              onChange={(e) => setForm({ ...form, coolingRate: Number(e.target.value) })}
            />
          </label>
          <label>
            Heating rate
            <input
              type="number"
              step={0.01}
              value={form.heatingRate}
              onChange={(e) => setForm({ ...form, heatingRate: Number(e.target.value) })}
            />
          </label>
          <label>
            Stabilization steps
            <input
              type="number"
              min={1}
              value={form.stabilizationSteps}
              onChange={(e) =>
                setForm({ ...form, stabilizationSteps: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Min length
            <input
              type="number"
              min={1}
              value={form.quality.minLength}
              onChange={(e) =>
                setForm({
                  ...form,
                  quality: { ...form.quality, minLength: Number(e.target.value) },
                })
              }
            />
          </label>
          <label>
            Min coherence
            <input
              type="number"
              step={0.05}
              value={form.quality.minCoherence}
              onChange={(e) =>
                setForm({
                  ...form,
                  quality: { ...form.quality, minCoherence: Number(e.target.value) },
                })
              }
            />
          </label>
          <label>
            Min diversity
            <input
              type="number"
              step={0.05}
              value={form.quality.minDiversity}
              onChange={(e) =>
                setForm({
                  ...form,
                  quality: { ...form.quality, minDiversity: Number(e.target.value) },
                })
              }
            />
          </label>
          <label>
            Max repetition
            <input
              type="number"
              step={0.05}
              value={form.quality.maxRepetition}
              onChange={(e) =>
                setForm({
                  ...form,
                  quality: { ...form.quality, maxRepetition: Number(e.target.value) },
                })
              }
            />
          </label>
          <label>
            Presets
            <select onChange={(e) => handleLoadPreset(e.target.value)} value="">
              <option value="">Load preset...</option>
              {presets.map((preset) => (
                <option key={preset.name} value={preset.name}>
                  {preset.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </aside>

      <main className="panel">
        <div className="cards">
          <div className="card">
            <div className="section-title">Dataset</div>
            <div>{derivedStats.datasetSize} samples</div>
          </div>
          <div className="card">
            <div className="section-title">Avg quality</div>
            <div>{derivedStats.quality.toFixed(3)}</div>
          </div>
          <div className="card">
            <div className="section-title">Cycles completed</div>
            <div>{derivedStats.cycles}</div>
          </div>
          <div className="card">
            <div className="section-title">Job state</div>
            <div>{job.status}</div>
          </div>
        </div>

        <div style={{ marginTop: 12 }}>
          <div className="section-title">Live logs</div>
          <div className="logs" data-testid="logs">
            {logs.join("\n")}
          </div>
        </div>

        <div style={{ marginTop: 12 }}>
          <div className="section-title">Annealing flags removed</div>
          <div className="cards">
            {Object.entries(metrics.removedFlags).map(([flag, count]) => (
              <div key={flag} className="card">
                <strong>{flag}</strong>: {count}
              </div>
            ))}
            {Object.keys(metrics.removedFlags).length === 0 && (
              <div className="card">No flags yet</div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

