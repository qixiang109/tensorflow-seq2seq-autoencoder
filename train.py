#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
import numpy as np
import tensorflow as tf
import data_utils
from tensorflow.python.platform import gfile
import seq2seq_autoencoder_model
import os

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data-path', help='path to read data from')
    arg('--model-path',help='path to save model to')
    arg('--max-read-num',type=int,default=100000,help='read this much of lines from data-path')
    arg('--validation-ratio',type=float,default=0.01, help='split out this much ratio of data as validation set')
    arg('--max-validation-num',type=int,default=1000, help = 'max validation set size')
    arg('--max-seq-length', type=int, default=30,help='the longest sequence length supported by the model, trainning data longer than this will be omitted, testing data longer than this will cause error')
    arg('--vocab-size', type=int, default=4000,help='how many most frequent words to keep, words beyond the vocabulary will be labeled as UNK')
    arg('--embedding-size', type=int, default=128,help='size of word embedding')
    arg('--state-size', type=int, default=128,help = 'size of hidden states')
    arg('--num-layers', type=int, default=1, help='number of hidden layers')
    arg('--cell', default='gru', help='cell type: lstm, gru')
    arg('--num-samples', type=int, default=256, help = 'number of sampled softmax')
    arg('--max-gradient-norm', type=float, default=5.0, help='gradient norm is commonly set as 5.0 or 15.0')
    arg('--optimizer',default='adam', help='Optimizer: adam, adadelta')
    arg('--learning-rate',type=float, default=0.01)
    arg('--batch-size', type=int, default=64)
    arg('--checkpoint-step', type=int, default=100, help='do validation and save after each this many of steps.')
    args = parser.parse_args()

    '''prepare data'''
    ids_path,vocab_path = data_utils.prepare_data(args.data_path, args.vocab_size)
    data_set = []
    with gfile.GFile(ids_path, mode="r") as ids_file:
        ids = ids_file.readline()
        counter = 0
        while ids and (not args.max_read_num or counter < args.max_read_num):
            counter += 1
            if counter % 100000 == 0:
                print("reading data line %d" % counter)
                sys.stdout.flush()
            ids = map(int, ids.split())
            data_set.append(ids)
            ids = ids_file.readline()
    data_set = [one for one in data_set if len(one)<=args.max_seq_length]
    train_validation_split_point = len(data_set) - min(int(len(data_set)*args.validation_ratio),args.max_validation_num)
    train_set,validation_set = data_set[0:train_validation_split_point],data_set[train_validation_split_point:len(data_set)]
    dictionary = {i:w for i,w in enumerate(open(vocab_path).read().decode('utf-8').strip().split('\n'))}
    reverse_dictionary ={w:i for i,w in dictionary.iteritems()}

    '''create model'''
    model = seq2seq_autoencoder_model.Model(args.vocab_size,args.embedding_size, args.state_size, args.num_layers, args.num_samples, args.max_seq_length, args.max_gradient_norm, args.cell, args.optimizer, args.learning_rate)

    '''fit model'''
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    model.fit(train_set,validation_set,args.batch_size,args.checkpoint_step,args.model_path)

if __name__ == '__main__':
    main()
