"""Packaging regressions using fake downloads; no devices or network access."""

import errno
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from export import download_models as installer
from runtime.preprocessing import encode_task_state


class PackagingTests(unittest.TestCase):
    def test_missing_assets_fail_before_devices(self):
        """Missing OMs/tokenizer fail clearly without opening cameras or motors."""
        import contextlib
        import io
        import json
        from runtime import realtime_inference as entry

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['realtime_inference.py', '--task', 'test', '--output', str(root / 'result.json')]
            for option in ('--part1-om', '--part2-om', '--tokenizer', '--stats'):
                argv.extend([option, str(root / 'missing')])
            with patch('sys.argv', argv), patch.object(entry, 'CameraPair') as cameras, \
                    patch.object(entry, 'connect_piper') as piper, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(), 1)
                cameras.assert_not_called()
                piper.assert_not_called()
            report = json.loads((root / 'result.json').read_text())
            self.assertEqual(report['summary']['last_stage'], 'model_files')

    def test_tokens_match_previous_encoding(self):
        """Shared encoding preserves the original quantization and token layout."""
        tokenizer = SimpleNamespace(encode=lambda text, add_bos: [1] + list(text.encode()))
        q01, q99 = np.zeros(7, np.float32), np.ones(7, np.float32)
        for task in ('pick_up\nthe can', 'long ' * 100):
            state = np.linspace(-1, 2, 7, dtype=np.float32)
            normalized = np.clip(2 * (state - q01) / np.maximum(q99 - q01, 1e-8) - 1, -1, 1)
            bins = np.digitize(normalized, np.linspace(-1, 1, 257)[:-1]).astype(np.int64) - 1
            cleaned = task.strip().replace('_', ' ').replace('\n', ' ')
            ids = tokenizer.encode(f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: ", add_bos=True)[:200]
            tokens, mask = encode_task_state(tokenizer, task, state, q01, q99)
            np.testing.assert_array_equal(tokens[0, :len(ids)], ids)
            self.assertEqual(int(mask.sum()), len(ids))
            self.assertFalse(tokens[0, len(ids):].any())

    def test_download_handles_separate_mounts(self):
        """Replacement stages beside its target, not across Docker bind mounts."""
        original_replace = installer.os.replace

        def replace(source, target):
            if Path(source).parent != Path(target).parent:
                raise OSError(errno.EXDEV, 'different filesystem')
            original_replace(source, target)

        def download(**kwargs):
            target = Path(kwargs['local_dir']) / kwargs['filename']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'new asset')

        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *_args, **_kw: SimpleNamespace(sha='abc')),
                              hf_hub_download=download)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(installer, 'PROJECT_ROOT', Path(directory)), \
                patch.dict('sys.modules', {'huggingface_hub': hub}), \
                patch.object(installer.os, 'replace', side_effect=replace):
            installer.download_models('owner/repo', 'main')
            for name in installer.RUNTIME_FILES:
                self.assertEqual((Path(directory) / name).read_bytes(), b'new asset')
            self.assertFalse(list(Path(directory).rglob('.download-*')))

    def test_failed_download_preserves_installed_bundle(self):
        """Incomplete downloads must not replace any existing model assets."""
        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *_args, **_kw: SimpleNamespace(sha='abc')),
                              hf_hub_download=lambda **_kw: (_ for _ in ()).throw(RuntimeError('offline')))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(installer, 'PROJECT_ROOT', Path(directory)), \
                patch.dict('sys.modules', {'huggingface_hub': hub}):
            target = Path(directory) / installer.RUNTIME_FILES[0]
            target.parent.mkdir(parents=True)
            target.write_bytes(b'existing asset')
            with self.assertRaises(RuntimeError):
                installer.download_models('owner/repo', 'main')
            self.assertEqual(target.read_bytes(), b'existing asset')


if __name__ == '__main__':
    unittest.main()
