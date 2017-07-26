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


def format_audio(img_path, sr = 44100, n_fft = 2048, hop_length = 512):
    audio_time_series, _ = librosa.load(img_path,
                                        sr = sr)
    S = librosa.feature.melspectrogram(y = audio_time_series,
                                       sr = sr,
                                       n_fft = n_fft,
                                       hop_length = hop_length)

    return S.T


def make_model_input_generator(files, batch_size = 128):
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
