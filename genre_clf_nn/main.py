#!/usr/bin/python

'''
    Branden Carrier
    07/25/2017

    This purpose of this script is to build a neural network that classifies a
    song's genre based on its audio.
'''

import keras
import utils

from keras import (
    Dense,
    GRU,
    Sequential
)

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
