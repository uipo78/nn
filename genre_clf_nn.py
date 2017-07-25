#!/usr/bin/env python3

'''
    Branden Carrier
    07/25/2017

    This purpose of this script is to build a neural network that classifies a
    song's genre based on its audio.
'''

import os
import utils
import random


class NameItSomethingGood(object):

    def __init__(self,
                 audio_dir = 'data/fma_large/',
                 meta_dir = 'data/fma_metadata/',
                 train_percent = 0.8,
                 valid_percent = 0.1):

        self.audio_dir = audio_dir
        self.meta_dir = meta_dir
        self.train_percent = train_percent
        self.valid_percent = valid_percent
        self.test_percent = 1 - train_percent - valid_percent
        self._train_files, self._valid_files, self._test_files = _get_split()


    def _get_split(self):
        '''
            Purpose -
                This function assigns a set of audio files to a training,
                validation and testing set

            Input -
                self.audio_dir: The directory containing the set of directories
                                holding the mp3 files.
            Output:
                train_files: List of relative paths of the files assigned to the
                             training set.
                valid_files: List of relative paths of the files assigned to the
                             validation set.
                test_files: List of relative paths of the files assigned to the
                            testing set.
        '''

        # Build the list mp3s by walking through each subdirectory of
        # self.audio_dir and appending to mp3s the relative path of all mp3
        # files
        mp3s = []
        for root, _, files in os.walk(self.audio_dir):
            for names in files:
                if name.endswith('.mp3'):
                    filepath = os.path.join(root, name)
                    mp3s.append(filepath)

        # Shuffle the order of the files in mp3s so that the following
        # training-validation-testing assignment is random.
        random.shuffle(mp3s)

        # Build the trainining, validation, and testing sets as follows:
        # the first _TRAIN_PERCENT of mp3s is assignedd to the training
        # set; the following block (which makes up _VALID_PERCENT of mp3s)
        # is assigned to the validation set; what remains (which makes up
        # _TEST_PERCENT of mp3s) is assigned to the testing set.
        end_train_idx = round(self.train_percent * len(mp3s))
        end_valid_idx = end_train_idx + round((self.valid_percent * len(mp3s)))

        train_files = mp3s[:end_train_idx]
        valid_files = mp3s[end_train_idx:end_valid_idx]
        test_files = mp3s[end_valid_idx:]

        return train_files, valid_files, test_files


# Get metadata from fma_metadata
tracks = utils.load(META_DIR + 'tracks.csv')
genres = utils.load(META_DIR + 'genres.csv')
features = utils.load(META_DIR + 'features.csv')
echonest = utils.load(META_DIR + 'echonest.csv').echonest # .echonest remove useless top-most column


# Get audio data from fma_large
