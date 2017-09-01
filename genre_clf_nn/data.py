import ast
import librosa
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


class _Split(Dataset):

    def __init__(self, audio_dir, df, sound_transformer):
        self.audio_dir = audio_dir
        self.df = df
        self.sound_transformer = sound_transformer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        track_id = str(self.df.loc[idx, "track_id"])
        filename = ("0" * (6 - len(track_id))) + track_id + ".mp3"
        parent_name = filename[:3]
        filepath = os.path.join(self.audio_dir, parent_name, filename)
        audio_ts, _ = librosa.load(filepath, sr=SR)

        return sound_transformer(audio_ts), self.df.loc[idx, "genre_top"]


class Data(object):

    def __init__(self, meta_dir, sound_transformer, train_perc, sr=22050):
        df, self.encoder = self._get_genre_labels_and_encoder(meta_dir)

        X = df[[col for col in df.columns if col != "genre_top"]]
        y = df.genre_top

        self.n_genres = y.unique().shape[0]

        skf = StratifiedKFold(n_splits=2)
        train_idx, valid_idx = next(skf.split(X, y))

        self.training_data = _Split(audio_dir, df.iloc[train_index, :], sound_transformer)
        self.validation_data = _Split(audio_dir, df.iloc[valid_index, :], sound_transformer)

    @staticmethod
    def _get_genre_labels_and_encoder(meta_dir):
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
