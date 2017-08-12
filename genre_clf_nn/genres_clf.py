#!/usr/bin/python

"""
    Branden Carrier
    07/25/2017

    This purpose of this script is to build a neural network that classifies a
    song"s genre based on its audio.
"""
import ast
import librosa
import numpy as np
import os
import pandas as pd

from keras.layers import Dense, Dropout, Input
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers.pooling import (
    GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D
)
from keras.utils.data_utils import Sequence
from sklearn.preprocessing import LabelEncoder


SEED = 54645

AUDIO_DIR = "../data/fma_large/"
META_DIR = "../data/fma_metadata/"

TRAIN_PERC = 0.8
VALID_PERC = 0.1

SR = 22050
N_FFT = 2048
N_MELS = 128


class DataSequence(Sequence):

    def __init__(self, audio_dir, meta_dir, batch_size):
        self.audio_dir, self.meta_dir = audio_dir, meta_dir
        self.label_df, self.encoder = self._get_genre_labels()
        self.n_classes = len(self.encoder.classes_)
        self.batch_size = batch_size


    def _get_genre_labels(self):
        assert self.meta_dir is not None

        filepath = os.path.join(self.meta_dir, "tracks.csv")
        tracks = pd.read_csv(filepath, index_col = 0, header = [0, 1])

        for item in [("track", "genres"), ("track", "genres_all")]:
            tracks[item] = tracks[item].map(ast.literal_eval)

        tracks = tracks["track"]
        tracks = tracks.reindex(index=tracks["genre_top"].dropna().index,
                                method=None)

        le = LabelEncoder()
        genres = pd.DataFrame(data={"name": tracks["genre_top"],
                                    "label": le.fit_transform(tracks["genre_top"])},
                              index=tracks.index)
        genres.reset_index(inplace=True)

        return genres, le


    def _get_x_value(self, idx):
        track_id = str(self.label_df.ix[idx, "track_id"])
        filename = ("0" * (6 - len(track_id))) + track_id + ".mp3"
        parent_name = filename[:3]
        filepath = os.path.join(self.audio_dir, parent_name, filename)

        audio_ts, _ = librosa.load(filepath, sr)
        mel_spec = librosa.feature.melspectrogram(y=audio_ts, sr=sr, n_fft=n_fft, n_mels=n_mels)

        return mel_spc


    def _get_y_value(self, idx):
        return self.label_df.ix[idx, "label"]


    def __len__(self):
        self.label_df.shape[0]


    def __getitem__(self, idx):
        batch_idx = list(range((idx * self.batch_size), ((idx + 1) * self.batch_size)))

        return np.array([_get_x_value(item) for item in batch_idx]), \
               np.array([_get_y_value(item) for item in batch_idx])


class Data(object):

    def __init__(self, audio_dir, meta_dir):
        self.audio_dir, self.meta_dir = audio_dir, meta_dir
        self.label_df, self.encoder = self._get_genre_labels()
        self.n_classes = len(self.encoder.classes_)


    def _get_genre_labels(self):
        assert self.meta_dir is not None

        filepath = os.path.join(self.meta_dir, "tracks.csv")
        tracks = pd.read_csv(filepath, index_col = 0, header = [0, 1])

        for item in [("track", "genres"), ("track", "genres_all")]:
            tracks[item] = tracks[item].map(ast.literal_eval)

        tracks = tracks["track"]
        tracks = tracks.reindex(index=tracks["genre_top"].dropna().index,
                                method=None)

        le = LabelEncoder()
        genres = pd.DataFrame(data={"name": tracks["genre_top"],
                                    "label": le.fit_transform(tracks["genre_top"])},
                              index=tracks.index)

        return genres, le


    def generator(self, df, sr, n_fft, n_mels, audio_file_ext):
        for track_id, row in self.label_df.iterrows():
            track_id = str(track_id)
            filename = ("0" * (6 - len(track_id))) + track_id + ".mp3"
            parent_name = filename[:3]
            filepath = os.path.join(self.audio_dir, parent_name, filename)

            audio_ts, _ = librosa.load(filepath, sr)
            mel_spec = librosa.feature.melspectrogram(y=audio_ts, sr=sr, n_fft=n_fft, n_mels=n_mels)

            yield (mel_spec.T, row["label"])


    def get_generators(self, train_perc, valid_perc, sr, n_fft, n_mels, audio_file_ext=".mp3"):
        np.random.seed(SEED)
        train_size = round(train_perc * self.label_df.shape[0])
        valid_size = round(valid_perc * self.label_df.shape[0])
        train_idx = np.random.choice(a=self.label_df.index, size=train_size)
        valid_idx = np.random.choice(a=[idx for idx in self.label_df.index if idx not in train_idx],
                                    size=valid_size)
        train = self.label_df.reindex(index=train_idx, method=None)
        valid = self.label_df.reindex(index=valid_idx, method=None)

        train_gen = self.generator(train, sr, n_fft, n_mels, audio_file_ext)
        valid_gen = self.generator(valid, sr, n_fft, n_mels, audio_file_ext)

        return train_gen, valid_gen


def architect_model(input_shape, n_classes):
    inp = Input(shape=input_shape, dtype="float32")

    x = Conv1D(input_shape[1] * 2, kernel_size=4, padding="causal", activation="relu")(inp)
    x = MaxPooling1D(pool_size=4)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(input_shape[1] * 2, kernel_size=4, padding="causal", activation="relu")(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(input_shape[1] * 2, kernel_size=4, padding="causal", activation="relu")(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Dropout(0.25)(x)

    x = Concatenate()([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])

    # Fully connected
    x = Dense(2048, activation="relu")(x)
    x = Dense(2048, activation="relu")(x)
    out = Dense(n_classes, activation="softmax")(x)

    return Model(inputs=inp, outputs=out)


if __name__ == "__main__":
    # 1291 = (song_duration * sampling_rate) / hop_length
    #data = Data(AUDIO_DIR, META_DIR)
    data_seq = DataSequence(AUDIO_DIR, META_DIR, batch_size=64)
    model = architect_model(input_shape=(1291, N_MELS), n_classes=data_seq.n_classes)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit_generator(data_seq, steps_per_epoch=4000)
    #train_gen, valid_gen = data.get_generators(TRAIN_PERC, VALID_PERC, SR, N_FFT, N_MELS)
    # model.fit_generator(generator=data,
    #                     steps_per_epoch=round(TRAIN_PERC * data.label_df.shape[0]),
    #                     epochs=5,
    #                     validation_data=valid_gen,
    #                     validation_steps=round(VALID_PERC * data.label_df.shape[0]))

    with open("model.yaml", "w+") as f:
        f.write(model.to_yaml())
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
