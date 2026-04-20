"""
cli.py — Command-line interface for agentic-ds-ledger.

Used by the VS Code extension subprocess bridge. All output is a single
JSON line on stdout. Errors: {"error": "...", "detail": "..."} + exit 1.

Commands:
    version
    projects     [--db PATH]
    create       --project NAME [--db PATH] [--description TEXT]
    parse        FILE_PATH --project NAME [--db PATH]
    entries      --project NAME [--n INT] [--db PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback


def _out(data) -> None:
    print(json.dumps(data, default=str), flush=True)


def _err(msg: str, detail: str = "") -> None:
    print(json.dumps({"error": msg, "detail": detail}), flush=True)


def main():
    parser = argparse.ArgumentParser(
        prog="agentic-ledger",
        description="Agentic DS Ledger CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # version
    sub.add_parser("version", help="Print version as JSON")

    # projects
    p_projects = sub.add_parser("projects", help="List all projects")
    p_projects.add_argument("--db", default="ledger.db")

    # create
    p_create = sub.add_parser("create", help="Create or retrieve a project")
    p_create.add_argument("--project", required=True)
    p_create.add_argument("--description", default="")
    p_create.add_argument("--db", default="ledger.db")

    # parse
    p_parse = sub.add_parser("parse", help="Parse a file and store result")
    p_parse.add_argument("file_path")
    p_parse.add_argument("--project", required=True)
    p_parse.add_argument("--db", default="ledger.db")

    # entries
    p_entries = sub.add_parser("entries", help="List ledger entries for a project")
    p_entries.add_argument("--project", required=True)
    p_entries.add_argument("--n", type=int, default=20)
    p_entries.add_argument("--db", default="ledger.db")

    args = parser.parse_args()

    try:
        if args.command == "version":
            from . import __version__
            _out({"version": __version__})

        elif args.command == "projects":
            from .database import LocalDatabase
            db = LocalDatabase(args.db)
            _out({"projects": db.get_projects()})
            db.close()

        elif args.command == "create":
            from .ledger import Ledger
            ledger = Ledger(args.project, db_path=args.db)
            _out({"project": args.project, "id": ledger.project_id, "created": True})

        elif args.command == "parse":
            from .ledger import Ledger
            ledger = Ledger(args.project, db_path=args.db)
            # suppress inline IPython rendering in CLI context
            result = ledger._mcp.parse_file(str(__import__("pathlib").Path(args.file_path).resolve()))
            result["file_path"] = args.file_path
            entry_id = ledger._mcp.store_in_db(result, project_id=ledger.project_id)
            _out({
                "entry_id": entry_id,
                "file": args.file_path,
                "models_found": len(result.get("models", [])),
                "metrics_found": len(result.get("metrics", [])),
                "preprocessing_found": len(result.get("preprocessing", [])),
                "models": result.get("models", []),
                "metrics": result.get("metrics", []),
                "preprocessing": result.get("preprocessing", []),
            })

        elif args.command == "entries":
            from .ledger import Ledger
            ledger = Ledger(args.project, db_path=args.db)
            entries = ledger.entries(n=args.n)
            # Strip raw_text / pip_freeze from output (too large for subprocess JSON)
            for e in entries:
                e.pop("raw_text", None)
                if isinstance(e.get("environment"), dict):
                    e["environment"] = {
                        k: v for k, v in e["environment"].items()
                        if k != "pip_freeze"
                    }
            _out({"entries": entries})

        else:
            _err("Unknown command. Run with --help for usage.")
            sys.exit(1)

    except Exception as exc:
        _err(str(exc), traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
