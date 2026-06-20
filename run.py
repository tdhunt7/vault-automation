#!/usr/bin/env python3
"""
Vault Linker — run this to launch the TUI.

Usage:
    python run.py              Launch the interactive TUI
    python run.py --reindex    Force re-embed all notes, then launch
"""

import sys

def main():
    from vault_linker.config import VAULT_PATH
    if not VAULT_PATH.exists():
        print(f"Error: vault path does not exist: {VAULT_PATH}")
        print("Set the VAULT_PATH environment variable to your Obsidian vault, e.g.:")
        print("  export VAULT_PATH=~/ObsidianVaults/MyVault")
        sys.exit(1)

    if "--reindex" in sys.argv:
        from vault_linker.config import CACHE_FILE, TITLE_CACHE_FILE
        for cf in (CACHE_FILE, TITLE_CACHE_FILE):
            if cf.exists():
                cf.unlink()
        print("Caches cleared. Will re-embed on launch.")

    from vault_linker.app import run
    run()


if __name__ == "__main__":
    main()
