"""Export verified, fixed-batch Core ML refiners without changing original weights.

Use the separate export environment documented in docs/native-prototype.md.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import platform

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


class RefinerOutputs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, A, B):
        outputs = self.model(A, B)
        return outputs['trans'], outputs['rot']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'weights/coreml-refiner-fp32')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--precision', choices=['float32', 'float16'], default='float32')
    parser.add_argument('--atol', type=float, default=1e-4)
    args = parser.parse_args()
    if args.batch_size < 1 or not np.isfinite(args.atol) or args.atol <= 0:
        parser.error('Batch size and validation tolerance must be positive.')
    if np.lib.NumpyVersion(np.__version__) >= '2.4.0':
        parser.error('coremltools 9 scalar conversion requires numpy<2.4; use the isolated export environment.')
    os.environ['FOUNDATIONPOSE_DEVICE'] = 'cpu'
    from omegaconf import OmegaConf
    from learning.models.refine_network import RefineNet
    from coreml_refiner import checkpoint_digest
    import coremltools as ct
    logging.getLogger().setLevel(logging.WARNING)
    checkpoint = ROOT / 'weights/2023-10-28-18-33-37/model_best.pth'
    cfg = OmegaConf.load(checkpoint.parent / 'config.yml')
    if cfg['c_in'] != 6 or cfg['rot_rep'] != 'axis_angle':
        parser.error('This prototype supports the released six-channel axis-angle refiner only.')
    model = RefineNet(cfg=cfg, c_in=cfg.c_in).eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('model', state))
    exported = RefinerOutputs(model).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    batches = sorted({1, args.batch_size, 252 % args.batch_size} - {0})
    manifest = dict(validated=False, checkpoint_sha256=checkpoint_digest(checkpoint),
                    precision=args.precision, atol=args.atol, batch_size=args.batch_size,
                    torch=torch.__version__, coremltools=ct.__version__, numpy=np.__version__,
                    platform=platform.platform(), batches={})
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        for batch in batches:
            torch.manual_seed(0)
            A, B = torch.rand(batch, 6, 160, 160), torch.rand(batch, 6, 160, 160)
            with torch.inference_mode():
                expected = model(A, B)
                traced = torch.jit.trace(exported, (A, B), check_trace=False)
            precision = ct.precision.FLOAT32 if args.precision == 'float32' else ct.precision.FLOAT16
            converted = ct.convert(traced, inputs=[ct.TensorType(name='A', shape=A.shape), ct.TensorType(name='B', shape=B.shape)],
                                   outputs=[ct.TensorType(name='trans', dtype=np.float32), ct.TensorType(name='rot', dtype=np.float32)],
                                   convert_to='mlprogram', minimum_deployment_target=ct.target.macOS15,
                                   compute_precision=precision, skip_model_load=True)
            package = args.output / f'refiner_b{batch}.mlpackage'
            compiled = args.output / f'refiner_b{batch}.mlmodelc'
            converted.save(str(package))
            ct.models.utils.compile_model(str(package), destination_path=str(compiled))
            native = ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_GPU)
            predicted = native.predict({'A': A.numpy(), 'B': B.numpy()})
            errors = {key: float(np.max(np.abs(predicted[key]-expected[key].numpy()))) for key in ('trans', 'rot')}
            manifest['batches'][str(batch)] = dict(package=package.name, compiled=compiled.name, max_abs_error=errors)
            (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            if any(not np.isfinite(value) or value > args.atol for value in errors.values()):
                raise RuntimeError(f'Batch {batch} failed numeric validation: {errors}; export remains disabled.')
            print(f'Exported batch={batch} precision={args.precision} errors={errors}', flush=True)
        manifest['validated'] = True
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    print(f'Validated export: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
