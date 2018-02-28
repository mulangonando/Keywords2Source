import numpy as np
import re
import itertools
from collections import Counter
import csv

# sentence polarity dataset v1.0 from http://www.cs.cornell.edu/people/pabo/movie-review-data/

# Processing tokens
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 257 #prev 6
GO_ID = -1 # prev 1
EOS_ID = -2 # prev 2
UNK_ID = 256 # prev 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

#Load Extended ASCII codes
def get_ASCII():
    ascii_ref_dict = {}
    with open('data/ascii/ascii-table.csv', 'rb') as csvfile:
        asciicodes = csv.reader(csvfile)
        for row in asciicodes:
            ascii_ref_dict[row[1]] = row[0]
    return ascii_ref_dict

#Get Character Tockens
def get_char_tokens(sent):

    ascii_ref_dict = get_ASCII()
    coded_char_tokens = []
    chars = []
    i = iter(sent)
    #while i.hasnext():
    '''for curr_char in i :
        curr_code = 0
        #curr_char = i.next()
        if curr_char in ascii_ref_dict:
            curr_code = ascii_ref_dict[curr_char]
            coded_char_tokens.append(curr_code)
        else :
            coded_char_tokens.append(256)

            #print coded_char_tokens'''
    coded_char_tokens = [c for c in i]

    return coded_char_tokens

#Get Wod Tockens
def get_tokens(sent):

    words = []
    for space_separated_fragment in sent.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# 1. Load Data for Character Models [TO DO]
def load_char_data_and_labels():

    # Load the Class labels
    classes = list(open("data/classes.txt", "r").readlines())
    classes = [s.strip() for s in classes]
    classes_dict = {}

    for i in range(1,len(classes)+1):
        classes_dict[classes[i-1].strip()] = i
    #print classes,"\n\n"
    #print classes_dict

    #print 'Classes : ',classes
    # Load the data
    import csv
    X = []
    Y = []

    for label in classes :

        print "Starting : ", label
        with open('data/'+ label +'.csv', 'rb') as csvfile:
            data = csv.reader(csvfile)
            for row in data:
                if row[-1].strip() == label:
                    #print row[-1].strip()
                    Y.append(classes_dict[row[-1].strip()])
                    X.append(''.join(row[:-1]))
                #else:
                #    print row

    X = [get_char_tokens(sent) for sent in X ]

    '''    for sentence in X:
            i = iter(sentence)
            while i.hasnext():
                curr_code = 0
                curr_char = i.next()
                if curr_char in ascii_ref_dict:
                    curr_code = ascii_ref_dict[curr_char]
                    vocab[curr_code] += 1

    person_examples = list(open("data/person.csv", "r").readlines())
    positive_examples = list(open("data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    positive_examples = [get_tokens(clean_str(sent)) for sent in positive_examples]
    negative_examples = list(open("data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    negative_examples = [get_tokens(clean_str(sent)) for sent in negative_examples]
    X = positive_examples + negative_examples

    # Labels
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    print "Total: %i, NEG: %i, POS: %i" % (len(y), np.sum(y[:, 0]), np.sum(y[:, 1]))'''

    #create_char_vocabularyprint X
    '''y=[]

    '''
    j = 0
    for l in Y:
        if l== 2: j+=1

    print 'Total ones : ',j

    return X, Y

# 2. Load data for word models
def load_data_and_labels():

    # Load the data
    positive_examples = list(open("data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    positive_examples = [get_tokens(clean_str(sent)) for sent in positive_examples]
    negative_examples = list(open("data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    negative_examples = [get_tokens(clean_str(sent)) for sent in negative_examples]
    X = positive_examples + negative_examples

    # Labels
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    print "Total: %i, NEG: %i, POS: %i" % (len(y), np.sum(y[:, 0]), np.sum(y[:, 1]))

    return X, y

#Creates Vocabulary for Characters using Extended ASCII codes

def create_char_vocabulary(X, max_vocabulary_size=256):
    import csv
    vocab = {}
    for i in range(0,257):
        vocab[i] = 0

    ascii_ref_dict = get_ASCII()

    for sentence in X:
        for curr_char in sentence:
            curr_code = 0
            if curr_char in ascii_ref_dict:
                vocab[int(curr_code)] += 1


    # Get list of all vocab words starting with [_PAD, _GO, _EOS, _UNK]
    # and then words sorted by count
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    #vocab_list = vocab_list[:max_vocabulary_size]

    vocab_dict =  ascii_ref_dict #dict((x,y) for (y,x) in enumerate(vocab_list))
    rev_vocab_dict = {v: k for k, v in vocab_dict.items()}

    #print "Total of %i unique tokens" % len(vocab_list)

    return vocab_list, vocab_dict, rev_vocab_dict

#Creates Vocabulary for Words
def create_vocabulary(X, max_vocabulary_size=5000):

    vocab = {}
    for sentence in X:
        for word in sentence:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

    # Get list of all vocab words starting with [_PAD, _GO, _EOS, _UNK]
    # and then words sorted by count
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_list = vocab_list[:max_vocabulary_size]

    vocab_dict = dict((x,y) for (y,x) in enumerate(vocab_list))
    rev_vocab_dict = {v: k for k, v in vocab_dict.items()}

    print "Total of %i unique tokens" % len(vocab_list)
    return vocab_list, vocab_dict, rev_vocab_dict

def sentence_to_token_ids(sentence, vocab_dict):

    # get value for w if it is in vocab dict else return UNK_ID = 3
    return [vocab_dict.get(word, UNK_ID) for word in sentence]

# 1. For Character in a Sentence [TO DO]
def data_to_character_ids(X, vocab_dict):

    max_len = max(len(sentence) for sentence in X)
    seq_lens = []

    data_as_tokens = []
    for line in X:
        token_ids = sentence_to_token_ids(line, vocab_dict)
        # Padding
        data_as_tokens.append(token_ids + [PAD_ID]*(max_len - len(token_ids)))
        # Maintain original seq lengths for dynamic RNN
        seq_lens.append(len(token_ids))

    return data_as_tokens, seq_lens

# For words in a sentence
def data_to_token_ids(X, vocab_dict):

    max_len = max(len(sentence) for sentence in X)
    seq_lens = []

    data_as_tokens = []
    for line in X:
        token_ids = sentence_to_token_ids(line, vocab_dict)
        # Padding
        data_as_tokens.append(token_ids + [PAD_ID]*(max_len - len(token_ids)))
        # Maintain original seq lengths for dynamic RNN
        seq_lens.append(len(token_ids))

    return data_as_tokens, seq_lens

def split_data(X, y, seq_lens, train_ratio=0.8):

    X = np.array(X)
    y = np.array(y)
    seq_lens = np.array(seq_lens)

    #print seq_lens

    data_size = len(X)

    # Shuffle the data
    shuffle_indices = np.random.permutation(np.arange(data_size))

    X = X[shuffle_indices]
    y = y[shuffle_indices]
    seq_lens = seq_lens[shuffle_indices]

    #X, y, seq_lens = X[shuffle_indices], y[shuffle_indices], \
    #                 seq_lens[shuffle_indices]'''

    # Split into train and validation set
    train_end_index = int(train_ratio*data_size)
    train_X = X[:train_end_index]
    train_y = y[:train_end_index]
    train_seq_lens = seq_lens[:train_end_index]
    valid_X = X[train_end_index:]
    valid_y = y[train_end_index:]
    valid_seq_lens = seq_lens[train_end_index:]

    return train_X, train_y, train_seq_lens, valid_X, valid_y, valid_seq_lens

def generate_epoch(X, y, seq_lens, num_epochs, batch_size):

    for epoch_num in range(num_epochs):
        yield generate_batch(X, y, seq_lens, batch_size)

def generate_batch(X, y, seq_lens, batch_size):
    print 'Batch Size : ', batch_size

    data_size = len(X)

    print 'Data Size : ', data_size

    num_batches = (data_size // batch_size)
    print "Num Batches : ", num_batches

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield X[start_index:end_index], y[start_index:end_index], \
              seq_lens[start_index:end_index]