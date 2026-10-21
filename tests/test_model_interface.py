import pytest
import torch
import torch.nn.functional as F
from torch import nn
from model import model, VLIFNet
from sr_runtime import SRAdapter, restore
from spikingjelly.activation_based.neuron import LIFNode


@pytest.mark.parametrize('scale', (2, 3, 4))
def test_adapter_padding_and_tiled_coverage_for_small_odd_images(scale):
    adapter = SRAdapter(nn.Identity(), scale)
    x = torch.rand(1, 3, 7, 13)
    expected = F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False)
    torch.testing.assert_close(restore(adapter, x, tile_size=16, overlap=4), expected)
    torch.testing.assert_close(restore(adapter, x, tile_size=0), expected)


def test_native_sr_factory_interface():
    class NativeSR(nn.Module):
        def forward(self, x):
            return x.repeat_interleave(3, -2).repeat_interleave(3, -1)
    adapter = SRAdapter(NativeSR(), 3, 'native', 8)
    x = torch.rand(1, 3, 11, 17)
    expected = NativeSR()(x)
    torch.testing.assert_close(restore(adapter, x, 8, 2), expected)
    with pytest.raises(ValueError, match='check bicubic/native'):
        SRAdapter(nn.Identity(), 3, 'native')(x)


def test_uploaded_network_backward_state_reset_and_architecture():
    torch.manual_seed(6)
    network = model()
    assert network.T == 4 and not network.use_refinement
    assert len(network.encoder_level2) == 4
    assert len(network.encoder_level3) == 8
    assert len(network.decoder_level3) == 2
    assert len(network.decoder_level2) == 2
    assert network.patch_embed.proj.out_channels == 48
    adapter = SRAdapter(network, 2)
    adapter.train()
    x = torch.rand(2, 3, 12, 12)
    prediction = adapter(x)
    assert prediction.shape == (2, 3, 24, 24)
    (prediction - torch.rand_like(prediction)).square().mean().backward()
    grads = [p.grad for p in network.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    adapter.reset()
    assert all(isinstance(module.v, float) for module in network.modules() if isinstance(module, LIFNode))
    adapter.eval()
    image = torch.rand(1, 3, 13, 17)
    one = restore(adapter, image, tile_size=24, overlap=8)
    restore(adapter, torch.rand_like(image), tile_size=24, overlap=8)
    two = restore(adapter, image, tile_size=24, overlap=8)
    torch.testing.assert_close(one, two, rtol=0, atol=0)
    assert one.shape == (1, 3, 26, 34)
    with pytest.raises(ValueError, match='T=4'):
        VLIFNet(T=8)


def test_attention_registration_and_tdbn_equivalence():
    from model import MultiDimensionalAttention, ThresholdDependentBatchNorm2d
    attention = MultiDimensionalAttention(T=4, C=48, reduction_t=4, reduction_c=16)
    assert isinstance(attention, nn.Module)
    assert set(attention.state_dict()) == {'ta.sharedMLP.0.weight', 'ta.sharedMLP.2.weight',
                                           'ca.sharedMLP.0.weight', 'ca.sharedMLP.2.weight',
                                           'sa.conv.weight'}
    batch_norm = ThresholdDependentBatchNorm2d(alpha=.7, v_th=.15, num_features=3)
    reference = nn.BatchNorm2d(3)
    reference.load_state_dict(batch_norm.state_dict())
    x = torch.randn(4, 2, 3, 5, 7)
    expected = reference(x.flatten(0, 1)).reshape_as(x)
    torch.testing.assert_close(batch_norm(x), expected, rtol=0, atol=0)
