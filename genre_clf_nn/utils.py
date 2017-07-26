'''
    Branden Carrier
    07/25/2017
'''

import ast
import os
import pandas as pd
import random


class NameItSomethingGood(object):

    def __init__(self,
                 audio_dir,
                 meta_dir,
                 train_percent = 0.8,
                 valid_percent = 0.1):

        self.audio_dir = audio_dir
        self.meta_dir = meta_dir
        self.train_percent = train_percent
        self.valid_percent = valid_percent
        self.test_percent = 1 - train_percent - valid_percent
        self.train_files, self.valid_files, self.test_files = _get_splits()
        self.features, self.echonest, self.genres, self.tracks = _load_meta_csvs()


    def _get_splits(self):
        '''
            Purpose -
                This function assigns a set of audio files to training,
                validation and testing sets

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
            for filename in files:
                if filename.endswith('.mp3'):
                    filepath = os.path.join(root, filename)
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


    def _load_meta_csvs(self):
        '''
            The main contents of this function are a copy of the load function
            found in utils.py from the FMA Dataset repository
        '''

        features, echonest, genres, tracks =  None, None, None, None

        for root, _, files in os.walk(self.meta_dir):
            for filename in files:

                filepath = os.path.join(root, filename)

                if filename.endswith('features.csv'):
                    features = pd.read_csv(filepath,
                                           index_col = 0,
                                           header = [0, 1, 2])

                elif filename.endswith('echonest.csv'):
                    echonest = pd.read_csv(filepath,
                                           index_col = 0,
                                           header = [0, 1, 2])

                elif filename.endswith('genres.csv'):
                    genres = pd.read_csv(filepath,
                                         index_col = 0)

                elif filename.endswith('tracks.csv'):
                    tracks = pd.read_csv(filepath,
                                         index_col = 0,
                                         header = [0, 1])

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

        # Alert the user if any one of the DataFrames to be returned was
        # not made
        if None in [features, echonest, genres, tracks]:
            csv_names = ['features', 'echonest', 'genres', 'tracks']
            is_none = [name for name in csv_names if eval(name) is None]
            print('Warning - A DataFrame was not made for' + \
                  'the following csvs: ' + ', '.join(is_none))

        return features, echonest, genres, tracks
