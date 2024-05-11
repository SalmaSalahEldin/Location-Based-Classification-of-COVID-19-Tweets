
import argparse
import math
import re
import time
from keras.preprocessing.text import Tokenizer
import gc
import numpy as np
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from os import path, environ
import os
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-d", "--dataset_size", type=str, help="Dataset size.", default='small',
                        choices=['micro', 'small', 'mid', 'large'])
    parser.add_argument("-c", "--classifier", type=str, help="Train a US State or US Census Region classifier.",
                        default='state',
                        choices=['region', 'state'])
    parser.add_argument("--max_words", type=int, help="Max number of words to analyze per user.", default=50)
    parser.add_argument("-v", "--vocab_size", type=int, help="Use the top N most frequent words.", default=86023)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size.", default=512)
    parser.add_argument("--hidden_size", type=int, help="Number of neurons in the hidden layers.", default=128)
    parser.add_argument("--tensorboard", action="store_true", help="Track training progress using Tensorboard.",
                        default=True)
    args = parser.parse_args()

    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    

    checkpoint_path = r"/content/drive/MyDrive/saved_trained_MODEL/"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  save_freq ="epoch")
                                                  #,save_freq=6095,monitor='val_accuracy',mode='max')

    callbacks=[cp_callback]    
        



    from twgeo.data import twus_dataset, constants
    from twgeo.models.geomodel import Model
    from twgeo.data.twus_dataset import  _load_data
    #from twgeo.data.twus_dataset import preprocess

    
    if args.classifier == 'state':
        num_of_classes = 13
        x_train, y_train, x_dev, y_dev, x_test, y_test = twus_dataset.load_state_data()
        print("train.py: Training a US State classifier.")


    geoModel = Model(batch_size=args.batch_size, use_tensorboard=args.tensorboard)

    geomodel_state_model_file = path.join(r'/content/drive/MyDrive/saved_trained_MODEL/', 'geomodel_' + args.classifier)
    if path.exists(geomodel_state_model_file + ".h5"):
        print("Loading existing model at {0}".format(geomodel_state_model_file))
        geoModel.load_saved_model(geomodel_state_model_file)
    else:     
         print("can't find a saved model")
         l=["full_paper_model_glove"]
         for i in l:
            mycode='''geoModel.{}(empadding_wor_len=50,dropout_par=0.4,num_outputs=num_of_classes, time_steps=args.max_words, vocab_size=args.vocab_size,hidden_layer_size=args.hidden_size) '''
            exec(mycode.format(i))
            
    geoModel.build_model(num_outputs=num_of_classes, time_steps=args.max_words, vocab_size=args.vocab_size,
                             hidden_layer_size=args.hidden_size)
    geoModel.train(x_train, y_train, x_dev, y_dev, epochs=args.epochs)# , callbacks=[cp_callback]  )    
    geoModel.save_model(path.join('/content/drive/MyDrive/saved_trained_MODEL/', 'geomodel_' + args.classifier + i))

    #workaround for bug https://github.com/tensorflow/tensorflow/issues/3388
    import gc
    gc.collect()
