"""Select an inference device without loading portable-only dependencies."""
import os
import torch


def select_device():
    name = os.environ.get('FOUNDATIONPOSE_DEVICE', 'auto')
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name not in ('cpu', 'mps', 'cuda'):
        raise ValueError('FOUNDATIONPOSE_DEVICE must be auto, cpu, mps or cuda')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; use --device cpu to diagnose.')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    return name


DEVICE = select_device()

