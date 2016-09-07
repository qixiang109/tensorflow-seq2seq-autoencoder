#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
import os
import numpy as np
import re
from tensorflow.python.platform import gfile

PAD, GO, EOS, UNK = ('_PAD','_GO','_EOS','_UNK')
PAD_ID, GO_ID, EOS_ID, UNK_ID = range(4)


def basic_tokenizer(sentence):
    return sentence.strip().split()

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" %
              (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = basic_tokenizer(line)
                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = [PAD, GO, EOS, UNK] + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary):
    words = sentence
    if not isinstance(sentence,list):
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]

def data_to_token_ids(data_path, ids_path, vocabulary_path):
    if not gfile.Exists(ids_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(ids_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab)
                    tokens_file.write(
                        " ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_data(data_path,vocab_size):
    # Get wmt data to the specified directory.
    data_dir = os.path.realpath(data_path).split('/')[-2]
    data_name = os.path.realpath(data_path).split('/')[-1]

    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "%s.vocab%d.txt" % (data_name,vocab_size))
    create_vocabulary(vocab_path, data_path, vocab_size)

    # Create token ids for the data.
    ids_path = os.path.join(data_dir, "%s.ids%d.txt" % (data_name,vocab_size))
    data_to_token_ids(data_path, ids_path, vocab_path)

    return (ids_path,vocab_path)
