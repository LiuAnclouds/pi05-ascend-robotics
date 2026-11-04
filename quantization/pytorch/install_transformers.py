"""Install the validated OpenPI transformer replacements in the active Python env."""

import shutil
import sys
from pathlib import Path

import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from include.project_paths import OPENPI_SOURCE


def main() -> None:
    """Copy project replacements to transformers 4.53.3; no model files are changed.

    Input: the bundled OpenPI replacement directory and active Python environment.
    Output: patched transformers modules in that environment's site-packages.
    """
    destination = Path(transformers.__file__).parent
    source = OPENPI_SOURCE / "openpi/models_pytorch/transformers_replace"
    if transformers.__version__ != "4.53.3":
        raise RuntimeError("Expected transformers 4.53.3 from requirements.txt")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    check = destination / "models/siglip/check.py"
    text = check.read_text()
    check.write_text(
        text.replace(
            'transformers.__version__ == "4.53.2"', 'transformers.__version__ in ("4.53.2", "4.53.3")'
        )
    )
    print("OpenPI transformer replacements installed in", destination)


if __name__ == "__main__":
    main()
