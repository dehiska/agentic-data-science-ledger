/**
 * extension.ts — Activation point for the Agentic DS Ledger VS Code extension.
 *
 * Registers:
 *   - TreeView: "DS Ledger" in Explorer sidebar
 *   - onDidSaveTextDocument    (catches .py, .docx)
 *   - onDidSaveNotebookDocument (catches .ipynb — separate VS Code event)
 *   - Commands: parseFile, createProject, openDashboard, refreshTree
 */

import * as vscode from "vscode";
import { LedgerTreeProvider } from "./treeView";
import { parseFile, createProject, getVersion, getConfig, ParseResult } from "./pythonBridge";

const SUPPORTED_EXTS = new Set([".ipynb", ".py", ".docx", ".doc"]);

export function activate(context: vscode.ExtensionContext): void {
  // ── Tree View ──────────────────────────────────────────────────────────────
  const treeProvider = new LedgerTreeProvider();
  const treeView = vscode.window.createTreeView("agenticLedgerView", {
    treeDataProvider: treeProvider,
    showCollapseAll: true,
  });
  context.subscriptions.push(treeView);

  // Verify Python on activation (non-blocking)
  getVersion()
    .then((v) => {
      console.log(`[DS Ledger] agentic-ds-ledger v${v} found`);
    })
    .catch(() => {
      vscode.window
        .showWarningMessage(
          "DS Ledger: agentic-ds-ledger package not found. Install it to enable parsing.",
          "Show Setup Instructions"
        )
        .then((choice) => {
          if (choice === "Show Setup Instructions") {
            vscode.window.showInformationMessage(
              "Run: pip install -e /path/to/agentic-ds-ledger\n" +
                "Then set agenticLedger.pythonPath to the correct interpreter."
            );
          }
        });
    });

  // ── Command: Parse Current File ────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand("agenticLedger.parseFile", async () => {
      const filePath = _activeFilePath();
      if (!filePath) {
        vscode.window.showWarningMessage("DS Ledger: No active file to parse.");
        return;
      }
      const ext = _ext(filePath);
      if (!SUPPORTED_EXTS.has(ext)) {
        vscode.window.showWarningMessage(
          `DS Ledger: Unsupported file type "${ext}". Supported: .ipynb, .py, .docx`
        );
        return;
      }
      await _doParse(filePath, treeProvider);
    })
  );

  // ── Command: Create Project ────────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand("agenticLedger.createProject", async () => {
      const name = await vscode.window.showInputBox({
        prompt: "New project name",
        placeHolder: "e.g. LightGBM Experiments",
        validateInput: (v) => (v.trim() ? undefined : "Name cannot be empty"),
      });
      if (!name) return;

      try {
        const result = await createProject(name.trim());
        vscode.window.showInformationMessage(
          `DS Ledger: Project "${result.project}" ready (id=${result.id})`
        );
        treeProvider.refresh();
      } catch (err) {
        vscode.window.showErrorMessage(`DS Ledger: ${err}`);
      }
    })
  );

  // ── Command: Open Dashboard ────────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand("agenticLedger.openDashboard", () => {
      vscode.env.openExternal(vscode.Uri.parse("http://localhost:8501"));
    })
  );

  // ── Command: Refresh Tree ──────────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand("agenticLedger.refreshTree", () => {
      treeProvider.refresh();
    })
  );

  // ── Auto-parse on save: .py / .docx ───────────────────────────────────────
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (doc) => {
      if (!getConfig().autoParseOnSave) return;
      const ext = _ext(doc.uri.fsPath);
      if (!SUPPORTED_EXTS.has(ext) || ext === ".ipynb") return; // .ipynb handled below
      await _doParse(doc.uri.fsPath, treeProvider);
    })
  );

  // ── Auto-parse on save: .ipynb ─────────────────────────────────────────────
  // Note: .ipynb saves do NOT fire onDidSaveTextDocument in VS Code —
  // they use onDidSaveNotebookDocument instead.
  context.subscriptions.push(
    vscode.workspace.onDidSaveNotebookDocument(async (nb) => {
      if (!getConfig().autoParseOnSave) return;
      await _doParse(nb.uri.fsPath, treeProvider);
    })
  );
}

export function deactivate(): void {
  // Nothing to clean up — subprocess is short-lived per call
}

// ── Helpers ────────────────────────────────────────────────────────────────────

async function _doParse(
  filePath: string,
  treeProvider: LedgerTreeProvider
): Promise<void> {
  const { defaultProject } = getConfig();

  try {
    const result: ParseResult = await parseFile(filePath, defaultProject);

    const modelNames =
      result.models.length > 0
        ? result.models.map((m) => m.name).join(", ")
        : "no models detected";

    vscode.window.showInformationMessage(
      `DS Ledger: ${modelNames} — entry #${result.entry_id} added to "${defaultProject}"`
    );
    treeProvider.refresh();
  } catch (err) {
    vscode.window.showErrorMessage(`DS Ledger parse error: ${err}`);
  }
}

function _activeFilePath(): string | undefined {
  // Try text editor first, then notebook editor
  return (
    vscode.window.activeTextEditor?.document.uri.fsPath ??
    vscode.window.activeNotebookEditor?.notebook.uri.fsPath
  );
}

function _ext(filePath: string): string {
  const lastDot = filePath.lastIndexOf(".");
  return lastDot >= 0 ? filePath.slice(lastDot).toLowerCase() : "";
}
