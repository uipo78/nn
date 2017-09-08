import copy
import librosa
import os
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from sys import platform

from data import AllData
from models import GenreNet


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def exp_lr_scheduler(epoch, init_lr, optimizer, decay_factor=0.1, lr_decay_epoch=2):
    lr = init_lr * (decay_factor ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return optimizer


def train_model(all_data, init_lr, loss_function, model, n_epochs, optimizer):
    since = time.time()

    best_model = model
    best_accuracy = 0.0

    for epoch in range(n_epochs):
        print("Epoch {} / {}".format(epoch + 1, n_epochs))
        print("=" * 20)

        for phase in ["train", "validate"]:
            if phase == "train":
                data = all_data.training
                optimizer = exp_lr_scheduler(epoch, init_lr, optimizer)
                model.train(True)
            else:
                data = all_data.validation
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(data):
                if i % 10 == 0:
                    print("Data point", i)

                if inputs is None and labels is None:
                    continue

                inputs, labels = \
                    autograd.Variable(FloatTensor([inputs])), \
                    autograd.Variable(LongTensor([int(labels)]))

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = loss_function(outputs[0], autograd.Variable(torch.cuda.LongTensor([5])))

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data)
            epoch_accuracy = running_corrects / len(data)

            print("{} Loss: {:.4f} Accuracy: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

            if phase == "validate" and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation accuracy: {:.4}".format(best_accuracy))

    return best_model


def main():
    all_data = AllData(audio_dir="../data/fma_large/",
                       meta_dir="../data/fma_metadata/",
                       seed=55645840,
                       sound_transformer=(
                           lambda audio_ts: librosa.feature.melspectrogram(y=audio_ts,
                                                                           sr=22050,
                                                                           n_fft=2048,
                                                                           n_mels=128)
                       ),
                       sr=22050,
                       train_perc=0.8)

    if torch.cuda.is_available():
        model = GenreNet(n_classes=all_data.n_genres).cuda()
    else:
        model = GenreNet(n_classes=all_data.n_genres)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    fitted_model = train_model(all_data, 1e-4, loss_function, model, 1, optimizer)
    fitted_model.save_state_dict("best_model.pt")
    
    if platform == "linux" or platform == "linux2":
        os.system(
            "play --no-show-progress --null --channels 1 synth 1 sine 840")


if __name__ == "__main__":
    main()
