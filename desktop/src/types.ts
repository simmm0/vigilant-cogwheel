export type AnnealingPreset = {
  name: string;
  datasetPath: string;
  outputDir: string;
  cycles: number;
  initialTemperature: number;
  minTemperature: number;
  coolingRate: number;
  heatingRate: number;
  stabilizationSteps: number;
  quality: {
    minLength: number;
    minCoherence: number;
    minDiversity: number;
    maxRepetition: number;
  };
  enableAnnealing: boolean;
};

export type JobState = {
  id: string | null;
  status: "idle" | "running" | "stopped" | "error";
  lastMessage?: string;
};

export type DashboardMetrics = {
  datasetSize: number;
  avgQuality: number;
  cyclesCompleted: number;
  removedFlags: Record<string, number>;
};

