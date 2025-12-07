#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Deserialize;
use std::{
    collections::HashMap,
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::State;

#[derive(Default)]
struct AppState {
    jobs: Mutex<HashMap<String, Child>>,
    logs: Mutex<HashMap<String, String>>,
}

#[derive(Deserialize, Clone)]
struct RunRequest {
    dataset_path: String,
    output_dir: String,
    cycles: u32,
    initial_temperature: f64,
    min_temperature: f64,
    cooling_rate: f64,
    heating_rate: f64,
    stabilization_steps: u32,
    quality: QualityConfig,
    enable_annealing: bool,
    auth_token: Option<String>,
}

#[derive(Deserialize, Clone)]
struct QualityConfig {
    min_length: u32,
    min_coherence: f64,
    min_diversity: f64,
    max_repetition: f64,
}

impl RunRequest {
    fn to_args(&self) -> Vec<String> {
        let mut args = vec![
            "--dataset".into(),
            self.dataset_path.clone(),
            "--output-dir".into(),
            self.output_dir.clone(),
        ];
        if self.enable_annealing {
            args.push("--enable-annealing".into());
            args.push("--anneal-cycles".into());
            args.push(self.cycles.to_string());
            args.push("--anneal-initial-temp".into());
            args.push(self.initial_temperature.to_string());
            args.push("--anneal-min-temp".into());
            args.push(self.min_temperature.to_string());
            args.push("--anneal-cooling-rate".into());
            args.push(self.cooling_rate.to_string());
            args.push("--anneal-heating-rate".into());
            args.push(self.heating_rate.to_string());
            args.push("--anneal-stabilization-steps".into());
            args.push(self.stabilization_steps.to_string());
            args.push("--quality-min-length".into());
            args.push(self.quality.min_length.to_string());
            args.push("--quality-min-coherence".into());
            args.push(self.quality.min_coherence.to_string());
            args.push("--quality-min-diversity".into());
            args.push(self.quality.min_diversity.to_string());
            args.push("--quality-max-repetition".into());
            args.push(self.quality.max_repetition.to_string());
        }
        args
    }
}

#[tauri::command]
async fn run_training(state: State<'_, AppState>, payload: RunRequest) -> Result<String, String> {
    if payload.auth_token.as_deref().unwrap_or("").is_empty() {
        return Err("missing auth token".into());
    }

    let job_id = format!(
        "job-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis()
    );

    let mut cmd = Command::new("python");
    cmd.current_dir("..");
    cmd.arg("-m").arg("cli.main");
    for arg in payload.to_args() {
        cmd.arg(arg);
    }
    cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    {
        let mut logs = state.logs.lock().map_err(|e| e.to_string())?;
        logs.insert(job_id.clone(), "job started\n".into());
    }

    state
        .jobs
        .lock()
        .map_err(|e| e.to_string())?
        .insert(job_id.clone(), child);

    Ok(job_id)
}

#[tauri::command]
fn stop_training(state: State<'_, AppState>, job_id: String) -> Result<(), String> {
    if let Some(mut child) = state.jobs.lock().map_err(|e| e.to_string())?.remove(&job_id) {
        child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn tail_logs(state: State<'_, AppState>, job_id: String) -> Result<String, String> {
    let logs = state.logs.lock().map_err(|e| e.to_string())?;
    Ok(logs.get(&job_id).cloned().unwrap_or_default())
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![run_training, stop_training, tail_logs])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

