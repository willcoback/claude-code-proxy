# Claude Code Proxy - Main Package
"""
Automatically discover and register all model strategy implementations.
This ensures that new providers are registered without manual import in main.py.
"""

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import List

def _discover_strategies():
    """Discover and import all converter modules in provider subdirectories."""
    # Get the directory of this package
    package_dir = Path(__file__).parent

    # List of subdirectories that contain provider implementations
    # We'll look for directories that contain a converter.py file
    provider_dirs = []
    for item in package_dir.iterdir():
        if item.is_dir() and not item.name.startswith('__'):
            # Check if converter.py exists in this directory
            converter_file = item / "converter.py"
            if converter_file.exists():
                provider_dirs.append(item.name)

    # Import each converter module
    for provider_dir in provider_dirs:
        module_name = f"proxy.{provider_dir}.converter"
        try:
            importlib.import_module(module_name)
            # The module's StrategyFactory.register() will be executed on import
        except ImportError as e:
            # Log but don't crash - might be a development directory
            print(f"Warning: Failed to import {module_name}: {e}", file=sys.stderr)
        except Exception as e:
            # Catch any other errors during import/registration
            print(f"Warning: Error in {module_name}: {e}", file=sys.stderr)

# Run discovery when this package is imported
_discover_strategies()
