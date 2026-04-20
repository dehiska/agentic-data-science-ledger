/**
 * pythonBridge.ts
 *
 * Spawns the agentic-ds-ledger CLI as a subprocess and parses JSON stdout.
 *
 * Protocol:
 *   python -m agentic_ds_ledger.cli <subcommand> [args]
 *
 * Every invocation writes exactly one JSON line to stdout.
 * Errors arrive as: {"error": "...", "detail": "..."}
 */

import * as cp from "child_process";
import * as path from "path";
import * as vscode from "vscode";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface ParseResult {
  entry_id: number;
  file: string;
  models_found: number;
  metrics_found: number;
  preprocessing_found: number;
  models: Array<{ name: string; family?: string }>;
  metrics: Array<{ name: string }>;
  preprocessing: Array<{ name: string; type?: string }>;
}

export interface ProjectEntry {
  id: number;
  name: string;
  description: string;
  file_count: number;
  created_at: string;
}

export interface LedgerEntry {
  id: number;
  file_path: string;
  file_type: string;
  models: Array<{ name: string; family?: string }>;
  metrics: Array<{ name: string }>;
  preprocessing: Array<{ name: string }>;
  status: string;
  timestamp: string;
  project_name?: string;
}

// ── Config helpers ─────────────────────────────────────────────────────────────

function getConfig() {
  const cfg = vscode.workspace.getConfiguration("agenticLedger");
  const wsFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? ".";

  let dbPath = cfg.get<string>("dbPath", "").trim();
  if (!dbPath) {
    dbPath = path.join(wsFolder, "ledger.db");
  }

  return {
    pythonPath: cfg.get<string>("pythonPath", "python") || "python",
    dbPath,
    defaultProject: cfg.get<string>("defaultProject", "Default") || "Default",
    autoParseOnSave: cfg.get<boolean>("autoParseOnSave", true),
    wsFolder,
  };
}

// ── Core subprocess runner ─────────────────────────────────────────────────────

function runCli(args: string[]): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const { pythonPath, wsFolder } = getConfig();

    let stdout = "";
    let stderr = "";

    const proc = cp.spawn(pythonPath, ["-m", "agentic_ds_ledger.cli", ...args], {
      cwd: wsFolder,
    });

    proc.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    proc.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));

    proc.on("close", (code) => {
      const trimmed = stdout.trim();
      if (!trimmed) {
        reject(
          new Error(
            `Python process produced no output (exit ${code}).\n` +
              `stderr: ${stderr || "(empty)"}\n\n` +
              `Make sure agentic-ds-ledger is installed:\n  pip install -e /path/to/agentic-ds-ledger`
          )
        );
        return;
      }
      try {
        const result = JSON.parse(trimmed);
        if (result && typeof result === "object" && "error" in result) {
          reject(new Error(`${result.error}`));
        } else {
          resolve(result);
        }
      } catch {
        reject(
          new Error(
            `Could not parse Python output as JSON.\nOutput: ${trimmed}\nstderr: ${stderr}`
          )
        );
      }
    });

    proc.on("error", (err) => {
      reject(
        new Error(
          `Failed to start Python (${pythonPath}): ${err.message}\n` +
            `Check the agenticLedger.pythonPath setting.`
        )
      );
    });
  });
}

// ── Public API ─────────────────────────────────────────────────────────────────

export async function parseFile(
  filePath: string,
  projectName?: string
): Promise<ParseResult> {
  const { dbPath, defaultProject } = getConfig();
  const project = projectName ?? defaultProject;
  const result = await runCli(["parse", filePath, "--project", project, "--db", dbPath]);
  return result as ParseResult;
}

export async function getProjects(): Promise<ProjectEntry[]> {
  const { dbPath } = getConfig();
  const result = (await runCli(["projects", "--db", dbPath])) as {
    projects: ProjectEntry[];
  };
  return result.projects ?? [];
}

export async function getEntries(
  projectName: string,
  n = 50
): Promise<LedgerEntry[]> {
  const { dbPath } = getConfig();
  const result = (await runCli([
    "entries",
    "--project", projectName,
    "--n", String(n),
    "--db", dbPath,
  ])) as { entries: LedgerEntry[] };
  return result.entries ?? [];
}

export async function createProject(
  projectName: string,
  description = ""
): Promise<{ project: string; id: number }> {
  const { dbPath } = getConfig();
  const result = await runCli([
    "create",
    "--project", projectName,
    "--description", description,
    "--db", dbPath,
  ]);
  return result as { project: string; id: number };
}

export async function getVersion(): Promise<string> {
  const result = (await runCli(["version"])) as { version: string };
  return result.version ?? "unknown";
}

export { getConfig };
