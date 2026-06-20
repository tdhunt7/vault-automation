#!/usr/bin/env python3
"""
Vault Linker — run this to launch the TUI.

Usage:
    python run.py              Launch the interactive TUI
    python run.py --reindex    Force re-embed all notes, then launch
"""

import sys

def main():
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
