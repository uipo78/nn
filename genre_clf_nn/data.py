import ast
import librosa
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class _Split(Dataset):

    def __init__(self, audio_dir, df, sound_transformer, sr):
        self.audio_dir = audio_dir
        self.df = df
        self.sound_transformer = sound_transformer
        self.sr = sr

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        track_id = str(self.df.loc[idx, "track_id"])
        filename = ("0" * (6 - len(track_id))) + track_id + ".mp3"
        parent_name = filename[:3]
        filepath = os.path.join(self.audio_dir, parent_name, filename)
        audio_ts, _ = librosa.load(filepath, sr=self.sr)

        return self.sound_transformer(audio_ts), self.df.loc[idx, "genre_top"]


class AllData(object):

    def __init__(self, audio_dir, meta_dir, seed, sound_transformer, sr, train_perc):
        self.seed = seed
        df, self.encoder = self._get_df_and_encoder(meta_dir)

        X = df[[col for col in df.columns if col != "genre_top"]]
        y = df.genre_top

        self.n_genres = y.unique().shape[0]

        X_train, X_valid_test, _, y_valid_test = train_test_split(
            X, y, train_size=train_perc, random_state=self.seed, stratify=y
        )
        X_valid, X_test, _, _ = train_test_split(
            X_valid_test, y_valid_test, train_size=0.5, random_state=self.seed, stratify=y_valid_test
        )

        training = df.iloc[X_train.index, :].copy().reset_index()
        validation = df.iloc[X_valid.index, :].copy().reset_index()
        testing = df.iloc[X_test.index, :].copy().reset_index()

        self.training = _Split(audio_dir, training, sound_transformer, sr)
        self.validation = _Split(audio_dir, validation, sound_transformer, sr)
        self.testing = _Split(audio_dir, testing, sound_transformer, sr)

    @staticmethod
    def _get_df_and_encoder(meta_dir):
        filepath = os.path.join(meta_dir, "tracks.csv")
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        for item in [("track", "genres"), ("track", "genres_all")]:
            tracks[item] = tracks[item].map(ast.literal_eval)

        tracks = tracks["track"]
        tracks = tracks.reindex(index=tracks["genre_top"].dropna().index,
                                method=None)
        tracks.reset_index(inplace=True)

        le = LabelEncoder()
        tracks["genre_top"] = le.fit_transform(tracks["genre_top"])

        return tracks, le
