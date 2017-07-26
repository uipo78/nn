#!/usr/bin/python

'''
    Branden Carrier
    07/25/2017

    This purpose of this script is to build a neural network that classifies a
    song's genre based on its audio.
'''

import keras
import librosa
import utils

from keras import (
    Dense,
    GRU,
    Sequential
)


# Global constants
AUDIO_DIR = '../data/fma_large/'
META_DIR = '../data/fma_metadata/'

SAMPLING_RATE = 44100
FFT_WINDOW_LEN = 2048
HOP_LENGTH = 512

BATCH_SIZE = 128


def format_audio(img_path,
                 sr = SAMPLING_RATE,
                 n_fft = FFT_WINDOW_LEN,
                 hop_length = HOP_LENGTH):

    audio_time_series, _ = librosa.load(img_path,
                                        sr = sr)
    S = librosa.feature.melspectrogram(y = audio_time_series,
                                       sr = sr,
                                       n_fft = n_fft,
                                       hop_length = hop_length)

    return S.T


def make_model_input_generator(files, batch_size = BATCH_SIZE):
    to_stack = []
    for filepath in files:
        formatted_audio = format_audio(filepath)
        to_stack.append(formatted_audio)

        if len(to_stack) == batch_size:
            yield np.stack(to_stack, axis = 0)


def main():

    # Instantiate model
    model = Sequential()

    # Add first layer
    model.add(GRU(

    ))

    # Add hidden layers

    # Add final layer
    model.add(Dense(
        output_dim = SOMETHING,
        activation = SOMETHING
    ))

    # Compile model
    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    model.fit_generator()

if __name__ == '__main__':
    main()
