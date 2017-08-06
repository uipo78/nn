#!/usr/bin/python

'''
    Branden Carrier
    07/25/2017

    This purpose of this script is to build a neural network that classifies a
    song's genre based on its audio.
'''
import tensorflow as tf
import time

from models import Model
from utils import DataProcessor

from sklearn.preprocessing import LabelEncoder


class Config(object):
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 50
    learning_rate = 1e-4


class GenresClassifier(Model):

    @staticmethod
    def make_convo_relu_pool_layer():
        pass

    def add_placeholders(self):
        input_shape = (self.config.batch_size, self.config.n_features)
        labels_shape = (self.config.batch_size, self.config.n_classes)
        self.input_placeholder = tf.placeholder(tf.float32, shape=input_shape)
        self.labels_placeholder = tf.placeholder(tf.int32, shape=labels_shape)

    def add_prediction_op(self):
        b = tf.Variable(tf.zeros((self.config.batch_size, self.config.n_classes)))
        W = tf.Variable(tf.zeros((self.config.n_features, self.config.n_classes)))
        pred = softmax(tf.matmul(a=self.input_placeholder, b=W) + b)

        return pred

    def add_loss_op(self, pred):
        pass

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        training_op = optimizer.minimize(loss)

        return train_op

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {self.input_placeholder: inputs_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def get_datagen(self):
        #TODO FINISH
        assert self.config.meta_dir is not None, 'No path specified for meta_dir'

        data_processor = DataProcessor(self.config.audio_dir,
                                       self.config.meta_dir)
        tracks = data_processor.load_tracks()

        le = LabelEncoder()
        le.fit(tracks['genre.top'].unique())
        datagen = data_processor.batch_generator(
            filepaths=data_processor.train_files,
            get_label_function=lambda track_num: le.transform(
                tracks.loc[track_num, 'genre_top']
            ),
            batch_size=self.config.batch_size
        )

        return datagen

    def run_epoch(self, sess, inputs, labels):
        #TODO FINISH
        n_minibatches, total_loss = 0, 0
        datagen = get_datagen()
        for input_batch, labels_batch in datagen:
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)

        return total_loss / n_minibatches

    def fit(self, sess, inputs, labels):
        #TODO FINISH
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            duration = time.time() - start_time
            print('Epoch {0}: loss = {1:.2f} ({2:.3f} sec)'.format(
                epoch, average_loss, duration
            ))
            losses.append(average_loss)

        return losses

    def __init__(self, config):
        '''
            Initializes the model.

            Args:
                self.config: A model self.configuration object of type
                             self.config.
        '''

        self.config = config
        self.build()


if __name__ == '__main__':
    config = Config()

    with tf.Graph().as_default():

        # Build the model and add the variable initializer Op
        model = GenresClassifier(config)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            # Run the Op to initialize the variables.
            sess.run(init)

            # Fit the model
            losses = model.fit(sess, inputs, labels)
