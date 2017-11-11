# coding: utf-8

# # Data PreProcessing
# 
# Tasks To Cover:
# 1. Convert Ids To Line
# 2. Get All Conversations
# 3. Gather Dataset
# 4. Filter Data
# 5. Indexing
# 6. Filter out sentences with too many unknowns
# 7. Save necessary Dictionaries
# 8. Print a few Datapoints in Order and Proceed To Training

# In[1]:


import itertools
import pickle

# Necessary imports
import nltk
import numpy as np

# In[2]:


# important variables
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
limit = {
    'maxq': 25,
    'minq': 2,
    'maxa': 25,
    'mina': 2
}
UNK = 'unk'
VOCAB_SIZE = 10000

# In[3]:


'''
    1. Read Data from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''


def get_id2line():
    lines = open('cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    # id2line = pd.DataFrame.from_dict(id2line,orient='index')
    return id2line


# print(get_id2line().head())


# In[4]:


'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''


def get_conversations():
    conv_lines = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8',
                      errors='ignore').read().split('\n')
    convs = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(_line.split(','))
    # convs = pd.DataFrame(np.array(convs))
    return convs


# print(get_conversations().head())


# In[5]:


'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''


def gather_dataset(convs, id2line):
    questions = [];
    answers = []

    for conv in convs:
        if len(conv) % 2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i % 2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])
                #     questions = pd.DataFrame(np.array(questions))
                #     answers = pd.DataFrame(np.array(answers))
    return questions, answers


# q,a = gather_dataset(get_conversations(),get_id2line())
# print(q.head())
# print(a.head())


# In[6]:


'''
 remove anything that isn't in the vocabulary
    return str pure english

'''


def filter_line(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''


def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


# In[7]:


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''


def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + ['unk'] + [x[0] for x in vocab]
    # word2index
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist


# In[8]:


'''
 filter based on number of unknowns (words not in vocabulary)
  filter out the worst sentences

'''


def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([w for w in qline if w not in w2idx])
        unk_count_a = len([w for w in aline if w not in w2idx])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q / len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len) * 100 / data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


# In[9]:


'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''


def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        # print(len(idx_q[i]), len(q_indices))
        # print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''


def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0] * (maxlen - len(seq))


# In[10]:


def process_data():
    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')
    convs = get_conversations()
    print(convs[121:125])
    print('>> gathered conversations.\n')
    questions, answers = gather_dataset(convs, id2line)

    # change to lower case (just for en)
    questions = [line.lower() for line in questions]
    answers = [line.lower() for line in answers]

    # filter out unnecessary characters
    print('\n>> Filter lines')
    questions = [filter_line(line, EN_WHITELIST) for line in questions]
    answers = [filter_line(line, EN_WHITELIST) for line in answers]

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(questions, answers)

    for q, a in zip(qlines[141:145], alines[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q, a))

    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in qlines]
    atokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in alines]
    print('\n:: Sample from segmented list of words')

    for q, a in zip(qtokenized[141:145], atokenized[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q, a))

    # indexing -> idx2w, w2idx 
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    # filter out sentences with too many unknowns
    print('\n >> Filter Unknowns')
    qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)
    print('\n Final dataset len : ' + str(len(qtokenized)))

    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # count of unknowns
    unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
    # count of words
    word_count = (idx_q > 1).sum() + (idx_a > 1).sum()

    print('% unknown : {0}'.format(100 * (unk_count / word_count)))
    print('Dataset count : ' + str(idx_q.shape[0]))


    # print '>> gathered questions and answers.\n'
    # prepare_seq2seq_files(questions,answers)


if __name__ == '__main__':
    process_data()


# In[11]:


def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a
