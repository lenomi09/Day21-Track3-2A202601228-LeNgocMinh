#!/usr/bin/env python3
"""Generate kaggle/*.ipynb from notebooks/*.py (jupytext py:percent).

Same idea as build_colab.py, different bootstrap: Kaggle starts with torch and a
working dir already in place, but hands out 2x T4 by default and the lab does not
need multi-GPU (HARDWARE-GUIDE.md), so the bootstrap pins CUDA_VISIBLE_DEVICES=0
before torch is ever imported.

Usage: python scripts/build_kaggle.py    (needs jupytext)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks"
OUT = ROOT / "kaggle"

# Student's own fork — results and REPORT.md live here, not on the template repo.
REPO_URL = "https://github.com/lenomi09/Day21-Track3-2A202601228-LeNgocMinh.git"
REPO_DIR = "Day21-Track3-2A202601228-LeNgocMinh"

BOOTSTRAP = f"""# Setup (chạy ô này trước)
# Kaggle cấp GPU T4 x2 theo mặc định; lab này không cần multi-GPU (HARDWARE-GUIDE.md),
# nên khoá về 1 GPU TRƯỚC KHI import torch, tránh device_map="auto" tự chia model
# ra cả hai card.
import os, subprocess, sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO = "{REPO_URL}"
if not os.path.exists("{REPO_DIR}"):
    subprocess.run(["git", "clone", "-q", REPO], check=True)
os.chdir("{REPO_DIR}")
subprocess.run(["git", "pull", "-q"], check=False)
sys.path.insert(0, "src")

# Install from requirements.txt, NOT a copied list — same one-source-of-truth reason
# as the Colab bootstrap (see scripts/build_colab.py). torch is preinstalled on
# Kaggle and requirements.txt pins it compatibly, so that line is a no-op.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
               check=True)

os.environ.setdefault("COMPUTE_TIER", "T4")
import torch
print("commit :", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip())
print("GPU    :", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE — bật Settings > Accelerator > GPU T4 x2")
print("visible GPUs:", torch.cuda.device_count(), "(phải là 1)")
"""


def _stamp_cell_ids(raw: dict) -> None:
    """Deterministic cell ids so `make kaggle` on unchanged sources is a no-op diff.

    Same rationale as build_colab.py's _stamp_cell_ids: nbformat mints a random id
    per cell otherwise, which turns every regeneration into pure id-churn noise that
    nobody reviews (that noise is exactly how F-18 — stale bootstrap in committed
    notebooks — survived a review on the Colab side).
    """
    for i, cell in enumerate(raw.get("cells", [])):
        body = "".join(cell.get("source", []))
        cell["id"] = hashlib.sha1(f"{i}\x00{body}".encode()).hexdigest()[:8]


def _write(dest: pathlib.Path, nb) -> None:
    import jupytext
    jupytext.write(nb, dest, fmt="ipynb")
    raw = json.loads(dest.read_text(encoding="utf-8"))
    _stamp_cell_ids(raw)
    dest.write_text(json.dumps(raw, ensure_ascii=True, indent=1), encoding="utf-8")


def _run_all_notebook():
    """Hand-built, like colab/Lab21_RUN_ALL.ipynb — not derived from a single .py."""
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Lab 21 — Fine-tuning LLMs · RUN ALL (Kaggle, T4)\n\n"
            "Trước khi chạy: Settings (panel phải) -> **Accelerator: GPU T4 x2**, "
            "**Internet: On** (cần verify số điện thoại).\n\n"
            "| Ô | Làm gì | Thời gian |\n"
            "|---|---|---|\n"
            "| 1 | clone + install | ~1-2 phút |\n"
            "| 2 | smoke: import + unit test | ~30 giây |\n"
            "| 3 | **core pipeline NB1 -> NB5** | ~100-130 phút |\n"
            "| 4 | gatekeeper + in kết quả | ~10 giây |\n\n"
            "Quota GPU miễn phí Kaggle: ~30 giờ/tuần, mỗi session tối đa ~9-12 giờ — "
            "đủ cho 1 lần chạy full pipeline. Sau khi xong, tải `results/` và "
            "`adapters/correct/` về (hoặc Save Version -> Save & Run All để giữ output)."
        ),
        nbformat.v4.new_code_cell(BOOTSTRAP),
        nbformat.v4.new_code_cell(
            "# 2. Smoke — imports, seed data, unit tests (no GPU needed)\n"
            "!python scripts/verify.py --smoke"
        ),
        nbformat.v4.new_code_cell(
            "# 3. Core pipeline — NB1 -> NB5\n"
            "# EVAL_LIMIT rút ngắn cả hai tập eval: để trống = full run (nộp được),\n"
            "# 8 = smoke pass nhanh. STAGES cho phép resume sau khi 1 stage fail.\n"
            "import os\n\n"
            "COMPUTE_TIER = \"T4\"                    # CPU | LAPTOP | T4 | BIGGPU\n"
            "EVAL_LIMIT   = \"\"                      # \"\" | \"4\" | \"8\" | \"16\" | \"25\"\n"
            "STAGES       = \"nb1 nb2 nb3 nb4 nb5\"\n\n"
            "os.environ[\"COMPUTE_TIER\"] = COMPUTE_TIER\n"
            "if EVAL_LIMIT:\n"
            "    os.environ[\"EVAL_LIMIT\"] = EVAL_LIMIT\n"
            "else:\n"
            "    os.environ.pop(\"EVAL_LIMIT\", None)\n\n"
            "from labkit import device\n"
            "print(device.banner(), \"\\n\")\n\n"
            "!python scripts/colab_run.py {STAGES}"
        ),
        nbformat.v4.new_code_cell(
            "# 4. Gatekeeper + results\n"
            "!python scripts/verify.py\n"
            "print(\"\\n================ results/ ================\")\n"
            "!ls -la results/\n"
            "!echo && echo \"---- runs.csv ----\" && cat results/runs.csv 2>/dev/null\n"
            "!echo && echo \"---- verdict.json ----\" && cat results/verdict.json 2>/dev/null"
        ),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def main() -> int:
    try:
        import jupytext  # noqa: F401
    except ImportError:
        print("jupytext not installed:  pip install jupytext", file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    made = []

    for src in sorted(SRC.glob("*.py")):
        import jupytext
        import nbformat
        nb = jupytext.read(src, fmt="py:percent")
        nb.cells.insert(0, nbformat.v4.new_code_cell(BOOTSTRAP))
        dest = OUT / f"Lab21_{src.stem}.ipynb"
        _write(dest, nb)
        made.append(dest.name)

    run_all_dest = OUT / "Lab21_RUN_ALL.ipynb"
    _write(run_all_dest, _run_all_notebook())
    made.append(run_all_dest.name)

    print(f"wrote {len(made)} notebooks to kaggle/:")
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
