"""agentic_ds_ledger — Public API.

Quick start (Colab or local):
    from agentic_ds_ledger import Ledger

    ledger = Ledger("MyProject")
    summary = ledger.parse("model.ipynb")
    ledger.show()
"""

from .ledger import Ledger
from .database import LocalDatabase, get_db
from .mcp_server import MCPServer

__version__ = "0.1.0"
__all__ = ["Ledger", "LocalDatabase", "get_db", "MCPServer"]
