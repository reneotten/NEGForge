"""Frontend regression: execute real cells in a fresh, current-CPython kernel.

Requires the installed extension and ``negforge[notebook]`` dependencies.
"""
from base64 import b64decode
from io import BytesIO
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "negforge_demo.ipynb"


def test_gallery_import_preserves_backend():
    # A subprocess avoids inheriting pytest's Matplotlib/IPython state.
    subprocess.run(
        [sys.executable, "-c", """
import importlib.util
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot
before = matplotlib.get_backend()
spec = importlib.util.spec_from_file_location('demo', 'scripts/render_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
assert matplotlib.get_backend() == before, (before, matplotlib.get_backend())
"""],
        cwd=ROOT, check=True,
    )


def test_notebook_displays_five_real_pngs(tmp_path, monkeypatch):
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    from PIL import Image

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 6
    assert all(cell.execution_count is None and cell.outputs == [] for cell in code)

    # Exercise even a headless shell; the notebook must opt into rich rendering.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    manager = KernelManager(kernel_name="python3")
    # Never silently use a globally registered kernel's different interpreter.
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=300,
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    with client.setup_kernel(cleanup_kc=True):
        client.execute()
    nbformat.write(notebook, tmp_path / "negforge_demo.executed.ipynb")

    assert [cell.execution_count for cell in code] == list(range(1, 7))
    counts = []
    for cell in code:
        pngs = [
            output.data["image/png"] for output in cell.outputs
            if output.output_type in ("display_data", "execute_result")
            and "image/png" in output.data
        ]
        counts.append(len(pngs))
        for encoded in pngs:
            raw = b64decode(encoded, validate=True)
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")
            with Image.open(BytesIO(raw)) as image:
                assert image.format == "PNG"
                assert image.width >= 600 and image.height >= 300
                image.load()  # Decode pixels, not just a signature/header check.
                assert any(low != high for low, high in image.convert("RGB").getextrema())
    print(f"Executed {len(code)} cells; displayed PNGs per cell: {counts}; total: {sum(counts)}")
    assert counts == [0, 1, 1, 1, 1, 1], "Every figure cell must emit one real image/png"
    # Executing must not dirty the output-free source notebook.
    source = nbformat.read(NOTEBOOK, as_version=4)
    assert all(cell.execution_count is None and not cell.outputs
               for cell in source.cells if cell.cell_type == "code")
