"""Real architecture tests; no pretrained downloads needed."""
import os
import unittest
import torch
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet
from learning.models.score_network import ScoreNetMultiPair


class NetworkTests(unittest.TestCase):
    @torch.inference_mode()
    def test_score_chunking_preserves_cross_candidate_attention(self):
        torch.manual_seed(0)
        model = ScoreNetMultiPair(OmegaConf.create({'use_BN': True}), c_in=6).eval()
        a, b = torch.rand(4, 6, 160, 160), torch.rand(4, 6, 160, 160)
        old = os.environ.get('FOUNDATIONPOSE_BATCH_SIZE')
        try:
            os.environ['FOUNDATIONPOSE_BATCH_SIZE'] = '4'
            full = model(a, b, L=4)['score_logit']
            os.environ['FOUNDATIONPOSE_BATCH_SIZE'] = '1'
            chunked = model(a, b, L=4)['score_logit']
            torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-4)
            if torch.backends.mps.is_available():
                result = model.to('mps')(a.to('mps'), b.to('mps'), L=4)['score_logit'].cpu()
                torch.testing.assert_close(full, result, atol=1e-4, rtol=1e-3)
        finally:
            if old is None:
                os.environ.pop('FOUNDATIONPOSE_BATCH_SIZE', None)
            else:
                os.environ['FOUNDATIONPOSE_BATCH_SIZE'] = old

    @torch.inference_mode()
    def test_refine_cpu_and_mps_forward_agree(self):
        torch.manual_seed(0)
        model = RefineNet(OmegaConf.create({'use_BN': True, 'rot_rep': 'axis_angle'}), c_in=6).eval()
        a, b = torch.rand(1, 6, 160, 160), torch.rand(1, 6, 160, 160)
        expected = model(a, b)
        self.assertEqual(expected['trans'].shape, (1, 3))
        if torch.backends.mps.is_available():
            actual = model.to('mps')(a.to('mps'), b.to('mps'))
            for key in ('rot', 'trans'):
                torch.testing.assert_close(expected[key], actual[key].cpu(), atol=1e-4, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
