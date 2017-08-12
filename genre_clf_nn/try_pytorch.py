import ast
import librosa
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


SEED = 54645

AUDIO_DIR = "../data/fma_large/"
META_DIR = "../data/fma_metadata/"

TRAIN_PERC = 0.8
VALID_PERC = 0.1

SR = 22050
N_FFT = 2048
N_MELS = 128


class Data(Dataset):

    _SR = 22050

    def _get_genre_labels_and_encoder(self):
        assert self.meta_dir is not None

        filepath = os.path.join(self.meta_dir, 'tracks.csv')
        tracks = pd.read_csv(filepath, index_col = 0, header = [0, 1])

        for item in [('track', 'genres'), ('track', 'genres_all')]:
            tracks[item] = tracks[item].map(ast.literal_eval)

        tracks = tracks["track"]
        tracks = tracks.reindex(index=tracks["genre_top"].dropna().index,
                                method=None)
        tracks.reset_index(inplace=True)

        le = LabelEncoder()
        tracks["genre_top"] = le.fit_transform(tracks["genre_top"])

        return tracks, le


    def __init__(self, audio_dir, meta_dir, transformer, train_perc, valid_perc):
        self.audio_dir, self.meta_dir = audio_dir, meta_dir
        self.label_df, self.encoder = self._get_genre_labels_and_encoder()
        self.transformer = transformer
        self.train_perc, self.valid_perc = train_perc, valid_perc


    def __len__(self):
        return self.label_df.shape[0]


    def __getitem__(self, idx):
        track_id = self.label_df.ix[idx, "track_id"]
        filename = ("0" * (6 - len(str(track_id)))) + str(track_id) + ".mp3"
        parent_name = filename[:3]
        filepath = os.path.join(self.audio_dir, parent_name, filename)
        audio_ts, _ = librosa.load(filepath, sr=self._SR)
        sample = {"X": self.transformer(audio_ts), "y": self.label_df.ix[idx, "genres"]}

        return sample


class GenreNet(nn.Module):

    def __init__(self, input_shape=(1, 128, 1291)):
        super(GenreNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 10, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels = 10, out_channels = 20, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels = 20, out_channels = 20, kernel_size=4)

        in_features = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(in_features=in_features, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=2048)


    def _get_conv_output(self, shape):
        batch_size = 1
        input_ = autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._conv_layers(input_)
        n_size = output_feat.data.view(batch_size, -1).size(1)

        return n_size


    def _conv_layers(self, x):
        x = F.dropout(F.max_pool1d(input=F.relu(self.conv1(x)), kernel_size=4), p=0.25)
        x = F.dropout(F.max_pool1d(input=F.relu(self.conv2(x)), kernel_size=2), p=0.25)
        x = F.dropout(F.max_pool1d(input=F.relu(self.conv3(x)), kernel_size=2), p=0.25)

        return x


    def forward(self, x):
        x = self._conv_layers(x)
        x = torch.cat([F.max_pool1d(input=x, kernel_size=x.size(0)),
                       F.avg_pool1d(input=x, kernel_size=x.size(0))])
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        return x


def train(model, data, loss_function, optimizer, lr_scheduler, init_lr, n_epochs=10):
    since = time.time()
    best_model = model
    best_accuracy = 0.0

    for epoch in range(n_epochs):
        print("Epoch {} / {}".format(epoch + 1, n_epochs))
        print("=" * 20)

        for phase in ["train", "validate"]:
            if phase == "train":
                optimizer = lr_scheduler(optimizer, epoch, init_lr)

            running_loss = 0.0
            running_corrects = 0

            data_loader = DataLoader(dataset=data, batch_size=50)

            for d in data_loader:
                inputs, labels = d
                inputs, labels = autograd.Variable(inputs.cuda()), autograd.Variable(labels.cuda())

                optimizer.zero_grad()

                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = loss_function(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / SOMETHING
            epoch_accuracy = running_corrects / SOMETHIGN

            print("{} Loss: {:.4f} Accuracy: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

            if phase == "validate" and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("best validation accuracy: {;.4}".format(best_accuracy))

    return best_model


def exponential_learning_rate_scheduler(optimizer, epoch, init_lr, decay_factor=0.1, lr_decay_epoch=2):
    lr = init_lr * (decay_factor ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return optimizer


if __name__ == "__main__":
    model = GenreNet()
    data = Data(audio_dir=AUDIO_DIR,
                meta_dir=META_DIR,
                transformer=lambda audio_ts: librosa.feature.melspectrogram(y=audio_ts,
                                                                            sr=SR,
                                                                            n_fft=N_FFT,
                                                                            n_mels=N_MELS),
                train_perc=TRAIN_PERC,
                valid_perc=VALID_PERC)
    lr = 1e-4
    fitted_model = train(model=GenreNet(),
                         data=data,
                         loss_function=nn.CrossEntropyLoss(),
                         optimizer=optim.Adam(model.parameters(), lr),
                         lr_scheduler=exponential_learning_rate_scheduler,
                         init_lr=lr)

    fitted_model.save_state_dict("best_model.pt")
