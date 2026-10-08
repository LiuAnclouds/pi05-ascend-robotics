"""Install the validated OpenPI transformer replacements in the active Python env."""

from pathlib import Path
import shutil

import transformers


def main() -> None:
    """Copy project replacements to transformers 4.53.3; no model files are changed.

    Input: the bundled OpenPI replacement directory and active Python environment.
    Output: patched transformers modules in that environment's site-packages.
    """
    root = Path(__file__).resolve().parents[1]
    destination = Path(transformers.__file__).parent
    source = root / "openpi/src/openpi/models_pytorch/transformers_replace"
    if transformers.__version__ != "4.53.3":
        raise RuntimeError("Expected transformers 4.53.3 from requirements.txt")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    check = destination / "models/siglip/check.py"
    text = check.read_text()
    check.write_text(text.replace('transformers.__version__ == "4.53.2"',
                                 'transformers.__version__ in ("4.53.2", "4.53.3")'))
    print("OpenPI transformer replacements installed in", destination)


if __name__ == "__main__":
    main()
