import librosa
from scipy import signal
import numpy as np


class FEATS:

    def __init__(self):
        self.rescale = True
        self.preemphasize = True
        self.rescaling_max = 0.999
        self.n_fft = 2048
        self.hop_length = 100,  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
        self.win_length = 400,  # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
        self.magnitude_power = 2.
        self.ref_level_db = 20
        self.min_level_db = -100
        self.signal_normalization = True
        self.allow_clipping_in_normalization = True,  # Only relevant if mel_normalization = True
        self.symmetric_mels = True,  # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
        self.max_abs_value = 4.
        self.sample_rate = 8000  # 22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
        self.preemphasis = 0.97  # filter coefficient.
        self.fmin = 50
        self.fmax = 4000
        self.num_mels = 80

    def __preemphasis(self, wav):
        if self.preemphasize:
            return signal.lfilter([1, self.preemphasis], [1], wav)
        return wav

    def __stft(self, y):
        #return librosa.core.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
        #                         pad_mode='constant')
        return librosa.core.stft(y, n_fft=self.n_fft, pad_mode='constant')

    def __normalize(self, S):
        if self.allow_clipping_in_normalization:
            if self.symmetric_mels:
                return np.clip((2 * self.max_abs_value) * (
                        (S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value,
                               -self.max_abs_value, self.max_abs_value)
            else:
                return np.clip(self.max_abs_value * ((S - self.min_level_db) / (-self.min_level_db)), 0,
                               self.max_abs_value)

        assert (S.max() <= 0) and (S.min() - self.min_level_db >= 0)
        if self.symmetric_mels:
            return (2 * self.max_abs_value) * (
                    (S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value
        else:
            return self.max_abs_value * ((S - self.min_level_db) / (-self.min_level_db))

    def __amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def __build_mel_basis(self):
        assert self.fmax <= self.sample_rate // 2
        return librosa.filters.mel(self.sample_rate, self.n_fft, n_mels=self.num_mels,
                                   fmin=self.fmin, fmax=self.fmax)

    def __linear_to_mel(self, spectogram):
        mel_basis = self.__build_mel_basis()
        return np.dot(mel_basis, spectogram)

    def __melspectrogram(self, wav):
        # D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
        D = self.__stft(self.__preemphasis(wav))
        S = self.__amp_to_db(self.__linear_to_mel(np.abs(D) ** self.magnitude_power)) - self.ref_level_db

        if self.signal_normalization:
            return self.__normalize(S)
        return S

    def wav_to_mfccs(self, wav_file):
        # 1. Load the audio as a waveform `y`
        #    Store the sampling rate as `sr`
        y, sr = librosa.load(wav_file, sr=8000)


        wav = self.__preemphasis(y)

        # rescale wav
        if self.rescale:
            wav = wav / np.abs(wav).max() * self.rescaling_max

        # Assert all audio is in [-1, 1]
        if (wav > 1.).any() or (wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav))

        # Compute the mel scale spectrogram from the wav
        mel_spectrogram = self.__melspectrogram(wav).astype(np.float32)

        mfcc = librosa.feature.mfcc(mel_spectrogram)

        '''mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(0.0125*sr), n_fft=int(0.04*sr), fmin=0.0, n_mels=80)        '''

        return mfcc
