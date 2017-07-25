#!/usr/bin/env python3

'''
    Branden Carrier
    07/25/2017

    This purpose of this script is to build a neural network that classifies a
    song's genre based on its audio.
'''

import os
import pandas as pd
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


def load(filepath):
    '''
        This function is an exact copy of the load function found in the
        utils.py script from the FMA Dataset repository.
    '''

    filename = os.path.basename(filepath)

    if filename.endswith('features.csv'):
        return pd.read_csv(filepath, index_col = 0, header = [0, 1, 2])

    elif filename.endswith('echonest.csv'):
        return pd.read_csv(filepath, index_col = 0, header = [0, 1, 2])

    elif filename.endswith('genres.csv'):
        return pd.read_csv(filepath, index_col = 0)

    elif filename.endswith('tracks.csv'):
        tracks = pd.read_csv(filepath, index_col = 0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

    else:
        raise ValueError('Unrecognized file name passed')

# Get audio data from fma_large
