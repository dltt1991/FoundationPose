"""Optional native Core ML runtime for the unchanged RefineNet architecture."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stage_profile import stage


def checkpoint_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class CoreMLRefiner(torch.nn.Module):
    def __init__(self, directory, checkpoint=None, compute_units='ALL'):
        super().__init__()
        self.directory = Path(directory)
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        if not manifest.get('validated'):
            raise ValueError('Core ML export has not passed numeric validation.')
        if checkpoint is not None and checkpoint_digest(checkpoint) != manifest.get('checkpoint_sha256'):
            raise ValueError('Core ML export checkpoint differs from the selected PyTorch checkpoint.')
        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError('Install optional runtime: uv pip install --python .venv/bin/python coremltools==9.0') from exc
        if compute_units not in ('ALL', 'CPU_AND_GPU', 'CPU_AND_NE', 'CPU_ONLY'):
            raise ValueError('Unknown Core ML compute units.')
        self._models = {}
        for batch, item in manifest['batches'].items():
            path = self.directory / item['compiled']
            if not path.is_dir():
                raise FileNotFoundError(f'Compiled Core ML model is missing: {path}')
            self._models[int(batch)] = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, compute_units))
        self.manifest = manifest
        self.eval()

    def forward(self, A, B):
        if A.shape != B.shape or A.ndim != 4 or tuple(A.shape[1:]) != (6, 160, 160):
            raise ValueError('Core ML refiner expects matching N×6×160×160 inputs.')
        batch = len(A)
        if batch not in self._models:
            raise ValueError(f'Batch {batch} was not exported; available batches: {sorted(self._models)}')
        with stage('inputs_to_cpu'):
            a = np.ascontiguousarray(A.detach().cpu().numpy(), dtype=np.float32)
            b = np.ascontiguousarray(B.detach().cpu().numpy(), dtype=np.float32)
        with stage('native_predict'):
            outputs = self._models[batch].predict({'A': a, 'B': b})
        if any(np.asarray(outputs[key]).shape != (batch, 3) for key in ('trans', 'rot')):
            raise RuntimeError('Unexpected native refiner output shape.')
        with stage('outputs_to_device'):
            return {key: torch.as_tensor(np.asarray(outputs[key]), dtype=torch.float32, device=A.device)
                    for key in ('trans', 'rot')}
