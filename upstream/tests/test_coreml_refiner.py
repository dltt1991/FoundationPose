import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys


class CoreMLRefinerTests(unittest.TestCase):
    def test_cli_rejects_cuda_coreml_before_loading_device(self):
        script = Path(__file__).resolve().parents[1] / 'run_portable.py'
        result = subprocess.run([sys.executable, str(script), '--device', 'cuda',
                                 '--refiner-backend', 'coreml', '--check'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Core ML requires cpu or mps', result.stderr)

    def test_missing_manifest_is_clear_and_does_not_require_coreml_import(self):
        from coreml_refiner import CoreMLRefiner
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(FileNotFoundError, 'manifest.json'):
                CoreMLRefiner(path)

    def test_rejects_export_from_other_weights(self):
        from coreml_refiner import CoreMLRefiner
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / 'manifest.json').write_text(json.dumps(dict(validated=True, checkpoint_sha256='wrong', batches={})))
            checkpoint = root / 'checkpoint.pth'
            checkpoint.write_bytes(b'weights')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                CoreMLRefiner(root, checkpoint)

    def test_rejects_failed_numeric_validation(self):
        from coreml_refiner import CoreMLRefiner
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / 'manifest.json').write_text(json.dumps(dict(validated=False)))
            with self.assertRaisesRegex(ValueError, 'validation'):
                CoreMLRefiner(root)


if __name__ == '__main__':
    unittest.main()
