# Third-party code

- `openpi/` is the existing board snapshot based on Physical Intelligence
  OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`. Existing Ascend
  mask/CumSum/position-encoding changes in `pi0_pytorch.py` and the Python 3.10
  timezone fix in `shared/download.py` are retained without further changes.
  It is not an unmodified upstream checkout. See `openpi/LICENSE` and
  `openpi/LICENSE_GEMMA.txt`.
- `runtime/acllite/` comes from Ascend open-source samples. Its Apache-2.0
  license is included as `runtime/acllite/LICENSE`.
- `setup_env.sh` installs the OpenPI transformer replacements and extends their
  version check to Transformers 4.53.3, matching the verified board environment.
  The checked-in OpenPI source is not modified during installation.
- Model weights and tokenizer files are external deployment assets and are not
  redistributed by this repository.
