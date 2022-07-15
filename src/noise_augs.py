import numpy as np
import torch


class GaussianNoise(torch.nn.Module):
    """Add gaussian noise to the stft wave.

    Args:
        wave_shape (tuple): shape of generated gaussian noise,
        which will be added to raw waveform
    """

    def __init__(self, wave_shape):
        assert isinstance(wave_shape, tuple)
        self.wave_shape = wave_shape

    def __call__(self, wave):
        print(self.wave_shape)
        white_noise_wave = np.random.randn(self.wave_shape[0], self.wave_shape[1])
        noisy_wave = np.add(wave, white_noise_wave)

        return noisy_wave


class AddConst(torch.nn.Module):
    """Add constant to the stft wave.

    Args:
        constant (int): constant that will be added to the waveform
        position (int): at which frequency noise will be added
    """

    def __init__(self, constant=64, position=16):
        assert isinstance(constant, int)
        assert isinstance(position, int)

        self.constant = constant
        self.position = position

    def __call__(self, wave):
        wave[:self.position, :] = wave[:self.position, :] + self.constant

        return wave


class ZeroElements(torch.nn.Module):
    """Zero out some elements in stft wave.

    Args:
        zero_over_slice (tuple): which frequencies to zero out
    """

    def __init__(self, zero_over_slice=(16, 128)):
        assert isinstance(zero_over_slice, tuple)
        self.zero_over_slice = zero_over_slice

    def __call__(self, wave):
        wave[self.zero_over_slice[0]:self.zero_over_slice[1], :] = 0

        return wave
