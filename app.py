"""
HuggingFace Spaces entry point.

Launches the same Gradio UI as examples/build_UI.py but configured
for hosted deployment (binds to 0.0.0.0:7860).
"""

import importlib.util
import sys
from pathlib import Path

# Ensure src/ is importable in the Space environment
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import create_demo from examples/build_UI.py without needing __init__.py
spec = importlib.util.spec_from_file_location(
    "build_UI", str(Path(__file__).parent / "examples" / "build_UI.py")
)
build_ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_ui)

demo = build_ui.create_demo()
demo.launch(server_name="0.0.0.0", server_port=7860)
