"""
GitHub Integration — fetch notebooks from GitHub, push JSON summaries.

Requires GITHUB_TOKEN env var (scope: repo).

Usage:
    from src.github_integration import GitHubIntegration
    gh = GitHubIntegration(token)
    notebooks = gh.list_notebooks("user/repo")
    nb = gh.get_notebook("user/repo", "notebooks/exp1.ipynb")
    gh.push_summary("user/repo", "notebooks/exp1.ipynb", summary_dict)
"""

import base64
import json
import os
from typing import Dict, List, Optional


class GitHubIntegration:
    def __init__(self, github_token: Optional[str] = None):
        token = github_token or os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN is required for GitHub integration.")
        from github import Github
        self.github = Github(token)
        self._token = token

    # ── Notebooks ──────────────────────────────────────────────────────────────

    def list_notebooks(self, repo_name: str, branch: str = "main") -> List[Dict]:
        repo = self.github.get_repo(repo_name)
        notebooks = []
        try:
            self._walk_repo(repo, "", branch, notebooks)
        except Exception as e:
            print(f"[GitHub] Error listing notebooks: {e}")
        return notebooks

    def _walk_repo(self, repo, path: str, branch: str, results: List):
        contents = repo.get_contents(path, ref=branch)
        if not isinstance(contents, list):
            contents = [contents]
        for item in contents:
            if item.type == "dir":
                self._walk_repo(repo, item.path, branch, results)
            elif item.name.endswith(".ipynb"):
                results.append({
                    "name": item.name,
                    "path": item.path,
                    "sha": item.sha,
                    "url": item.html_url,
                })

    def get_notebook(self, repo_name: str, file_path: str, branch: str = "main") -> Optional[Dict]:
        repo = self.github.get_repo(repo_name)
        try:
            contents = repo.get_contents(file_path, ref=branch)
            return {
                "content": base64.b64decode(contents.content).decode("utf-8"),
                "path": contents.path,
                "sha": contents.sha,
            }
        except Exception as e:
            print(f"[GitHub] Error fetching {file_path}: {e}")
            return None

    # ── Summaries ──────────────────────────────────────────────────────────────

    def push_summary(
        self,
        repo_name: str,
        notebook_path: str,
        summary: Dict,
        branch: str = "main",
    ) -> bool:
        repo = self.github.get_repo(repo_name)
        summary_name = os.path.basename(notebook_path).replace(".ipynb", "_summary.json")
        summary_path = f"summaries/{summary_name}"
        content = json.dumps(summary, indent=2, default=str)

        try:
            # Update existing file if it already exists
            existing = repo.get_contents(summary_path, ref=branch)
            repo.update_file(
                summary_path,
                f"Update summary for {notebook_path}",
                content,
                existing.sha,
                branch=branch,
            )
        except Exception:
            # Create new
            try:
                repo.create_file(
                    summary_path,
                    f"Add summary for {notebook_path}",
                    content,
                    branch=branch,
                )
            except Exception as e:
                print(f"[GitHub] Error pushing summary: {e}")
                return False
        return True
