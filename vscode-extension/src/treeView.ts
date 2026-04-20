/**
 * treeView.ts — Projects → Files → Entries TreeView provider.
 *
 * Tree structure:
 *   📁 MyProject  (3 files)
 *     📓 lgbm_model.ipynb  (2 entries)
 *       #5  LightGBM | Ensemble  [2026-04-17]
 *       #3  LightGBM | Ensemble  [2026-04-16]
 *     🐍 preprocess.py  (1 entry)
 *       #1  StandardScaler | Scaling  [2026-04-15]
 */

import * as vscode from "vscode";
import {
  getProjects,
  getEntries,
  ProjectEntry,
  LedgerEntry,
} from "./pythonBridge";

// ── Node types ─────────────────────────────────────────────────────────────────

class PlaceholderNode extends vscode.TreeItem {
  readonly kind = "placeholder" as const;
  constructor(message: string) {
    super(message, vscode.TreeItemCollapsibleState.None);
    this.contextValue = "placeholder";
    this.iconPath = new vscode.ThemeIcon("info");
  }
}

type AnyNode = ProjectNode | FileNode | EntryNode | PlaceholderNode;

class ProjectNode extends vscode.TreeItem {
  readonly kind = "project" as const;
  constructor(public readonly project: ProjectEntry) {
    super(
      `${project.name}  (${project.file_count ?? 0} files)`,
      vscode.TreeItemCollapsibleState.Collapsed
    );
    this.tooltip = project.description || project.name;
    this.contextValue = "agenticProject";
    this.iconPath = new vscode.ThemeIcon("beaker");
  }
}

class FileNode extends vscode.TreeItem {
  readonly kind = "file" as const;
  constructor(
    public readonly filePath: string,
    public readonly projectName: string,
    public readonly entryCount: number
  ) {
    super(filePath, vscode.TreeItemCollapsibleState.Collapsed);
    this.description = `${entryCount} entr${entryCount === 1 ? "y" : "ies"}`;
    this.tooltip = filePath;
    this.contextValue = "agenticFile";
    // Icon by file type
    const ext = filePath.split(".").pop()?.toLowerCase();
    if (ext === "ipynb") {
      this.iconPath = new vscode.ThemeIcon("notebook");
    } else if (ext === "py") {
      this.iconPath = new vscode.ThemeIcon("symbol-method");
    } else {
      this.iconPath = new vscode.ThemeIcon("file-text");
    }
  }
}

class EntryNode extends vscode.TreeItem {
  readonly kind = "entry" as const;
  constructor(public readonly entry: LedgerEntry) {
    const modelNames =
      (entry.models ?? []).map((m) => m.name).join(", ") || "—";
    const label = `#${entry.id}  ${modelNames}`;
    super(label, vscode.TreeItemCollapsibleState.None);

    const ts = (entry.timestamp ?? "").slice(0, 16);
    this.description = ts;
    this.tooltip = [
      `File: ${entry.file_path}`,
      `Models: ${modelNames}`,
      `Metrics: ${(entry.metrics ?? []).map((m) => m.name).join(", ") || "—"}`,
      `Preprocessing: ${(entry.preprocessing ?? []).map((p) => p.name).join(", ") || "—"}`,
      `Status: ${entry.status ?? "—"}`,
    ].join("\n");
    this.contextValue = "agenticEntry";
    this.iconPath = new vscode.ThemeIcon("symbol-event");
  }
}

// ── Provider ───────────────────────────────────────────────────────────────────

export class LedgerTreeProvider
  implements vscode.TreeDataProvider<AnyNode>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    AnyNode | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  // Cache entries per project name to avoid re-fetching every expand
  private _cache = new Map<string, LedgerEntry[]>();

  refresh(): void {
    this._cache.clear();
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: AnyNode): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: AnyNode): Promise<AnyNode[]> {
    // Root level — list all projects
    if (!element) {
      try {
        const projects = await getProjects();
        if (projects.length === 0) {
          return [_placeholder("No projects yet. Use 'Ledger: Create Project'.")];
        }
        return projects.map((p) => new ProjectNode(p));
      } catch (err) {
        vscode.window.showWarningMessage(`DS Ledger: ${err}`);
        return [_placeholder("Could not load projects. Check Python setup.")];
      }
    }

    // Project level — group entries by file
    if (element instanceof ProjectNode) {
      try {
        const entries = await this._fetchEntries(element.project.name);
        if (entries.length === 0) {
          return [_placeholder("No files parsed yet.")];
        }
        const byFile = new Map<string, LedgerEntry[]>();
        for (const e of entries) {
          const key = e.file_path;
          if (!byFile.has(key)) byFile.set(key, []);
          byFile.get(key)!.push(e);
        }
        return [...byFile.entries()].map(
          ([fp, es]) => new FileNode(fp, element.project.name, es.length)
        );
      } catch {
        return [];
      }
    }

    // File level — list entries for this file
    if (element instanceof FileNode) {
      const all = this._cache.get(element.projectName) ?? [];
      return all
        .filter((e) => e.file_path === element.filePath)
        .map((e) => new EntryNode(e));
    }

    return [];
  }

  private async _fetchEntries(projectName: string): Promise<LedgerEntry[]> {
    if (!this._cache.has(projectName)) {
      const entries = await getEntries(projectName, 200);
      this._cache.set(projectName, entries);
    }
    return this._cache.get(projectName)!;
  }
}

function _placeholder(message: string): PlaceholderNode {
  return new PlaceholderNode(message);
}
