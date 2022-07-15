import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.noise_augs import GaussianNoise, AddConst, ZeroElements


class AudioDataset(Dataset):
    """Audio dataset."""

    def __init__(self, data, n_fft=2048):
        """
        Args:
            data (list): List of paths to data.
            n_fft (int): Length of the windowed signal
        Returns:
            result (dict): Dictionary with raw and noisy waveforms
        """
        self.data = data
        self.transform = None
        self.nfft = n_fft

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _normalize(waveform):
        """
        Normalizes waveform, so it has zero mean and unit variance
        :param waveform: waveform to be normalized
        :return: normalized waveform
        """
        waveform_mean = waveform - np.mean(waveform)
        result = waveform_mean / np.std(waveform_mean)
        return result

    @staticmethod
    def pad_or_truncate(waveform, length=65000):
        """
        Pads or truncates waveform, depending on its length
        :param waveform: waveform to be padded or truncated
        :param length: length waveform will be padded or truncated to
        :return: waveform of a given length
        """
        if waveform.shape[0] > length:
            return torch.narrow(torch.from_numpy(waveform), 0, 0, length)
        elif waveform.shape[0] <= length:
            padding = length - waveform.shape[0]
            padding_tensor = torch.zeros(padding)
            return torch.concat((torch.from_numpy(waveform), padding_tensor))

    def __getitem__(self, idx):
        raw_waveform, sample_rate = librosa.load(self.data[idx])
        resampled = librosa.resample(raw_waveform, orig_sr=sample_rate, target_sr=16000)
        raw_waveform = librosa.util.fix_length(resampled, size=len(resampled) + self.nfft // 2)
        stft_wave = librosa.stft(raw_waveform, n_fft=self.nfft)

        self.transform = torch.nn.Sequential(GaussianNoise(stft_wave.shape),
                                             AddConst(constant=64, position=16),
                                             ZeroElements(zero_over_slice=(16,128)))

        transformed_sample = self.transform(stft_wave)
        transformed_waveform = librosa.istft(transformed_sample, length=len(raw_waveform))

        result = {"raw_waveform": self.pad_or_truncate(self._normalize(raw_waveform)),
                  "transformed_waveform": self.pad_or_truncate(self._normalize(transformed_waveform))}
        return result


def collate_fn(batch):
    """
    Collate function
    :param batch: batch
    :return: Dictionary with raw and noisy waveforms
    """
    raw_waveforms = (torch.stack([(d['raw_waveform']) for d in batch if d]))
    transformed_waveforms = (torch.stack([(d['transformed_waveform']) for d in batch if d]))

    raw_waveforms = raw_waveforms.unsqueeze(-1)
    transformed_waveforms = transformed_waveforms.unsqueeze(-1)

    result = {
        "raw_waveforms": torch.permute(raw_waveforms, (0, 2, 1)).float(),
        "transformed_waveforms": torch.permute(transformed_waveforms, (0, 2, 1)).float()
    }

    return result


def get_dataloader(data, batch_size):
    """
    Function to get dataloader
    :param data: data for dataset
    :param collate_fn: collate function
    :param batch_size: batch_size
    :return: dataloader (iter)
    """
    dataloader = DataLoader(AudioDataset(data), batch_size=batch_size,
                            shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    return dataloader

