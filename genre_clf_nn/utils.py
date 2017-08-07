'''
    Branden Carrier
    07/25/2017
'''

import ast
import itertools
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder


class DataProcessor(object):

    @staticmethod
    def _get_splits(audio_dir, audio_file_ext, train_percent, valid_percent):
        '''
            This function assigns a list of files to training, validation, and
            testing sets.

            Args
                dir_: The directory in which all files to be assigned to subsets
                      are located.
                file_ext: The file extension of to-be-assigned files.
                train_percent: The percentage of files to be assigned to the
                               training set.
                valid_percent: The percentage of files to be assigned to the
                               validation set.

            Returns
                train_files: A list of paths (relative to the project root) of
                             the files assigned to the training set.
                valid_files: A list of paths (relative to the project root)
                             of the files assigned to the validation set.
                test_files: A list of paths (relative to the project root)
                            of the files assigned to the testing set.
        '''

        # Build files by walking through each subdirectory of self._audio_dir
        # and appending to files the relative path of all files in
        # self._audio_dir
        files = []
        for root, _, dir_files in os.walk(audio_dir):
            for filename in dir_files:
                if filename.endswith(audio_file_ext):
                    filepath = os.path.join(root, filename)
                    files.append(filepath)

        # Shuffle the order of the files so that the following
        # training-validation-testing assignment is random.
        random.shuffle(files)

        # Build the trainining, validation, and testing sets as follows:
        # the first self._train_percent of files is assigned to the training
        # set; the following block (which makes up self._valid_percent of files)
        # is assigned to the validation set; what remains (which makes up
        # 1 - (self._train_percent + self._valid_percent) of files) is assigned
        # to the testing set.
        end_train_idx = int(train_percent * len(files))
        end_valid_idx = end_train_idx + int((valid_percent * len(files)))

        train_files = files[:end_train_idx]
        valid_files = files[end_train_idx:end_valid_idx]
        test_files = files[end_valid_idx:]

        return train_files, valid_files, test_files

    def __init__(self, config):
        self._config = config
        self._train_files, self._valid_files, self._test_files = \
            self._get_splits(self._config.audio_dir,
                             self._config.audio_file_ext,
                             self._config.train_percent,
                             self._config.valid_percent)

    def print_config(self):
        my_attribs = [a for a in dir(self.config) if not a.startswith('__')]
        for attrib in my_attribs:
            print(attrib, getattr(self.config, attrib))

    def load_features(self):
        assert self._config.meta_dir is not None

        filepath = self._config.meta_dir + 'features.csv'
        features = pd.read_csv(filepath, index_col = 0, header = [0, 1, 2])

        return features

    def load_echonest(self):
        assert self._config.meta_dir is not None

        filepath = self._config.meta_dir + 'echonest.csv'
        echonest = pd.read_csv(filepath, index_col = 0, header = [0, 1, 2])

        return echonest

    def load_genres(self):
        assert self._config.meta_dir is not None

        filepath = self._config.meta_dir + 'genres.csv'
        genres = pd.read_csv(filepath, index_col = 0)

        return genres

    def load_tracks(self):
        assert self._config.meta_dir is not None

        filepath = self._config.meta_dir + 'tracks.csv'
        tracks = pd.read_csv(filepath, index_col = 0, header = [0, 1])

        # Safely evaluating strings containing Python values from
        # untrusted sources
        for item in [('track', 'tags'),
                     ('album', 'tags'),
                     ('artist', 'tags'),
                     ('track', 'genres'),
                     ('track', 'genres_all')]:
            tracks[item] = tracks[item].map(ast.literal_eval)

        # Convert to datetime
        for item in [('track', 'date_created'),
                     ('track', 'date_recorded'),
                     ('album', 'date_created'),
                     ('album', 'date_released'),
                     ('artist', 'date_created'),
                     ('artist', 'active_year_begin'),
                     ('artist', 'active_year_end')]:
            tracks[item] = pd.to_datetime(tracks[item])

        # Convert to ordered category
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            dtype = 'category',
            categories = ('small', 'medium', 'large'),
            ordered = True
        )

        # Convert to unordered category
        for item in [('track', 'genre_top'),
                     ('track', 'license'),
                     ('album', 'type'),
                     ('album', 'information'),
                     ('artist', 'bio')]:
            tracks[item] = tracks[item].astype('category')

        return tracks

    # @staticmethod
    # def _get_audio_ts(filepaths, sr):
    #     audio_ts_list = []
    #     for filepath in filepaths:
    #         X, sr = librosa.load(filepath, sr)
    #         audio_ts_list.append(X)
    #
    #     return audio_ts_list
    #
    # @staticmethod
    # def _extract_mean_features(audio_ts, **kwargs):
    #     stft = np.abs(
    #         librosa.stft(audio_ts)
    #     )
    #     mfccs = np.mean(
    #         librosa.feature.mfcc(y=audio_ts, sr=kwargs['sr']).T,
    #         axis=0
    #     )
    #     chroma = np.mean(
    #         librosa.feature.chroma_stft(S=stft, sr=kwargs['sr']).T,
    #         axis=0
    #     )
    #     mel = np.mean(
    #         librosa.feature.melspectrogram(audio_ts, sr=kwargs['sr']).T,
    #         axis=0
    #     )
    #     contrast = np.mean(
    #         librosa.feature.spectral_contrast(S=stft, sr=kwargs['sr']).T,
    #         axis=0
    #     )
    #     tonnetz = np.mean(
    #         librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_ts),
    #                                 sr=kwargs['sr']).T,
    #         axis=0
    #     )
    #
    #     return mfccs, chroma, mel, contrast, tonnetz
    #
    # @classmethod
    # def batch_generator(cls, filepaths, get_label_function, batch_size):
    #     filepath_batches = [iter(filepaths)] * batch_size
    #     for filepath_batch in itertools.zip_longest(*filepath_batches):
    #
    #         audio_ts_batch = cls._get_audio_ts(filepath_batch,
    #                                            sr=cls.FEAT_EXTR_PARAMS['sr'])
    #
    #         # 173 is the the number of columns to which the concatentation of
    #         # features expands
    #         features_mat, labels_vec = np.empty((batch_size, 173)), np.empty(0)
    #
    #         for filepath, audio_ts in zip(filepath_batch, audio_ts_batch):
    #             ith_features = np.concatenate([
    #                 cls._extract_features(audio_ts=audio_ts,
    #                                       **cls.FEATURE_EXTR_PARAMS)
    #             ])
    #             features_mat = np.stack([features_mat, ith_features])
    #             filename = filepath.split('/')[-1].replace(cls.AUDIO_FILE_EXT, '')
    #             labels_vec = np.append(labels_vec, get_label_function(filename))
    #
    #         yield features_mat, labels_vec

    def _get_datagen(self, filepaths, label_function):
        sr = self._config.feature_extr_params['sr']
        n_fft = self._config.feature_extr_params['n_fft']
        n_mels = self._config.feature_extr_params['n_mels']
        audio_file_ext = self._config.audio_file_ext

        for filepath in filepaths:
            audio_ts, _ = librosa.load(filepath, sr)

            filename = filepath.split('/')[-1].replace(audio_file_ext, '')
            label = label_function(filename)

            if label is not None:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_ts, sr=sr, n_fft=n_fft, n_mels=n_mels
                )

                yield (mel_spec, label)

    def get_train_datagen(self, label_function):
        return self._get_datagen(self._train_files, label_function)

    def get_valid_datagen(self, label_function):
        return self._get_datagen(self._valid_files, label_function)


class Visualizer(object):

    PLOT_NAMES = {
        'chroma': 'Chromagram',
        'cqt_hz': 'Constant-Q power spectrogram (Hz)',
        'cqt_note': 'Constant-Q power spectrogram (Note)',
        'linear': 'Linear-frequency power spectrogram',
        'log': 'Log-frequency power spectrogram',
        'mel': 'Mel spectrogram',
        'tempo': 'Tempogram'
    }
    VALID_SPEC_TYPE = PLOT_NAMES.keys()

    _FIGURE_KWARGS_KEYS = ['dpi', 'figsize']

    def __init__(self):
        pass

    @staticmethod
    def _get_gram_plot_data(audio_ts, spec_type, **kwargs):
        data = None

        if spec_type in ['linear', 'log']:
            spec_data = librosa.stft(audio_ts)
            data = librosa.amplitude_to_db(spec_data, ref=np.max)
        elif spec_type == 'mel':
            spec_data = librosa.feature.melspectrogram(audio_ts,
                                                       sr=kwargs['sr'],
                                                       n_mels=128)
            data = librosa.power_to_db(spec_data, ref=np.max)
        elif spec_type in ['cqt_note', 'cqt_hz']:
            data = librosa.amplitude_to_db(librosa.cqt(audio_ts, kwargs['sr']),
                                           ref=np.max)
        elif spec_type == 'chroma':
            data = librosa.feature.chroma_cqt(y=audio_ts, sr=kwargs['sr'])
        elif spec_type == 'tempo':
            data = librosa.feature.tempogram(y=audio_ts, sr=kwargs['sr'])
        else:
            raise ValueError('Unrecognized type')

        return data

    @classmethod
    def _plot_gram(cls, audio_ts, spec_type, grayscale, **kwargs):
        assert spec_type in cls.VALID_SPEC_TYPE

        fig = plt.figure(**{key: kwargs[key] for key in kwargs
                            if key in cls._FIGURE_KWARGS_KEYS})
        data = cls._get_gram_plot_data(audio_ts, spec_type, **kwargs)
        librosa.display.specshow(data=data,
                                 cmap=('gray_r' if grayscale else 'inferno'),
                                 x_axis='time',
                                 y_axis=spec_type)
        if spec_type in ['chroma', 'tempo']:
            plt.colorbar()
        else:
            plt.colorbar(format='%+2.0f dB')
        plt.title(cls.PLOT_NAMES[spec_type])

    @classmethod
    def _plot_gram_mult(cls, names, audio_ts_list, spec_type, grayscale,
                        **kwargs):
        assert isinstance(names, list) and isinstance(audio_ts_list)

        fig = plt.figure(**{key: kwargs[key] for key in kwargs
                            if key in cls._FIGURE_KWARGS_KEYS})
        for i, (name, audio_ts) in enumerate(zip(names, audio_ts_list)):
            plt.subplot(len(audio_ts_list), 1, i + 1)
            data = cls._get_gram_plot_data(audio_ts, spec_type=spec_type,
                                           **kwargs)
            librosa.display.specshow(data=data,
                                     cmap=('gray_r' if grayscale else 'inferno'),
                                     x_axis='time',
                                     y_axis=spec_type)
            plt.title(name.title())
            if spec_type in ['chroma', 'tempo']:
                plt.colorbar()
            else:
                plt.colorbar(format='%+2.0f dB')
        plt.suptitle(cls.PLOT_NAMES[spec_type], x=0.5, y=0.915, fontsize=18)

    @classmethod
    def show_gram_plot(cls, audio_ts, spec_type, grayscale=False, **kwargs):
        cls._plot_gram(audio_ts, spec_type, grayscale, **kwargs)
        plt.show()

    @classmethod
    def show_gram_plots(cls, names, audio_ts_list, spec_type, grayscale=False,
                        **kwargs):
        cls._plot_gram_mult(names, audio_ts_list, spec_type, grayscale,
                            **kwargs)
        plt.show()

    @classmethod
    def save_gram_plot(cls, audio_ts, spec_type, savepath, grayscale=False,
                       **kwargs):
        cls._plot_gram(audio_ts, spec_type, grayscale, **kwargs)
        plt.savefig(savepath)

    @classmethod
    def save_gram_plots(cls, names, audio_ts_list, spec_type, savepath,
                        grayscale=False, **kwargs):
        cls._plot_gram_mult(names, audio_ts_list, spec_type, grayscale,
                            **kwargs)
        plt.savefig(savepath)


class AudioPlayer(object):
    pass
