#!/usr/bin/python

'''
    Branden Carrier
    07/25/2017

    This purpose of this script is to build a neural network that classifies a
    song's genre based on its audio.
'''
import numpy as np
import tensorflow as tf
import time

from keras.layers import Dense, Dropout, Input
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers.pooling import (
    GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D
)
from sklearn.preprocessing import LabelEncoder

from utils import DataProcessor



class Config(object):
    audio_dir = '../data/fma_large/'
    audio_file_ext = '.mp3'
    batch_size = 64
    feature_extr_params = {
        'sr': 22050,
        'hop_length': 512,
        'n_fft': 2048,
        'n_mels': 128
    }
    meta_dir = '../data/fma_metadata/'
    train_percent = 0.8
    valid_percent = 0.1
    test_percent = 1 - train_percent - valid_percent


if __name__ == '__main__':
    config = Config()

    data_processor = DataProcessor(config)

    tracks = data_processor.load_tracks()
    tracks = tracks['track']

    le = LabelEncoder()
    le.fit([genre for genre in tracks['genre_top'].unique() if genre != np.nan])

    train_datagen = data_processor.get_train_datagen(
        lambda track_id: le.transform(tracks.ix[track_id, 'genre_top'])
                         if tracks.ix[track_id, 'genre_top']
                         else None
    )
    valid_datagen = data_processor.get_valid_datagen(
        lambda track_id: le.transform(tracks.ix[track_id, 'genre_top'])
                         if tracks.ix[track_id, 'genre_top']
                         else None
    )

#===============================================================================
    n_mels = config.feature_extr_params['n_mels']
    n_classes = le.classes_.shape[0]

    # 1291 = (song_duration * sampling_rate) / hop_length
    inp = Input(shape=(1291, n_mels), dtype='float32', name='inp')

    # conv0
    h = Conv1D(n_mels * 2, kernel_size=4, padding='causal', activation='relu')(inp)
    h = MaxPooling1D(pool_size=4)(h)
    h = Dropout(0.25)(h)

    # conv1
    h = Conv1D(n_mels * 2, kernel_size=4, padding='causal', activation='relu')(h)
    h = MaxPooling1D(pool_size=2)(h)
    h = Dropout(0.25)(h)

    # conv2
    h = Conv1D(n_mels * 4, kernel_size=4, padding='causal', activation='relu')(h)
    h = MaxPooling1D(pool_size=2)(h)
    h = Dropout(0.25)(h)

    # Global pooling and concatenation
    concat = Concatenate()([GlobalAveragePooling1D()(h), GlobalMaxPooling1D()(h)])

    # Fully connected
    full = Dense(2048, activation='relu')(concat)

    out = Dense(n_classes, activation='softmax')(full)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(generator=train_datagen,
                        steps_per_epoch=len(data_processor._train_files) / config.batch_size,
                        validation_data=valid_datagen,
                        validation_steps=len(data_processor._valid_files) / config.batch_size)
