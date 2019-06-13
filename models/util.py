import os
import sys
from itertools import islice

import math
import numpy as np
import ujson as json
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.util import get_line_count
from collections import defaultdict
from torch.autograd import Variable
import torch
from common.util import array_from_sparse_dict


# pad per cross sentence paragraphs in pair


def pad_paragraph(cross_sents):
    num_paragraph = len(cross_sents)
    max_sent_num = int(np.mean([len(x) for x in cross_sents]))
    max_token_num = int(np.mean([len(val) for sublist in cross_sents for val in sublist]))
    main_matrix = np.zeros((num_paragraph, max_sent_num, max_token_num), dtype=np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i, j, k] = cross_sents[i][j][k]
                except IndexError:
                    pass
    return main_matrix.swapaxes(0, 1)


def eval_metric(true_y, pred_y, pred_p):
    '''
    calculate the precision and recall for p-r curve
    reglect the NA relation
    '''
    assert len(true_y) == len(pred_y)
    positive_num = len([i for i in true_y if i > 0])
    index = np.argsort(pred_p)[::-1]

    tp = 0
    fp = 0
    fn = 0
    all_pre = [0]
    all_rec = [0]
    fp_res = []

    # for idx in range(len(true_y)):
    # whole PR-Curve
    for idx in range(len(index)):
        i = true_y[index[idx]]
        j = pred_y[index[idx]]

        if i == 0:  # NA relation
            if j > 0:
                fp_res.append((index[idx], j, pred_p[index[idx]]))
                fp += 1
        else:
            if j == 0:
                fn += 1
            else:
                if i == j:
                    tp += 1

        if fp + tp == 0:
            precision = 1.0
        else:
            precision = tp * 1.0 / (tp + fp)
        recall = tp * 1.0 / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)

    print(("tp={}; fp={}; fn={}; positive_num={}".format(tp, fp, fn, positive_num)))
    return all_pre[1:], all_rec[1:], fp_res


def save_pr(out_dir, name, epoch, pre, rec):
    out = open('{}/{}_{}_PR.txt'.format(out_dir, name, epoch + 1), 'w')
    for p, r in zip(pre, rec):
        out.write('{} {}\n'.format(p, r))
    out.close()


def iterate_minibatches(data, batchsize, shuffle=False):
    if shuffle:
        # shuffle not supported now
        indices = list(range(len(data)))
        np.random.shuffle(indices)
    for start_idx in range(0, len(data) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield data[excerpt]


def gen_minibatch(data, batch_size, shuffle=False):
    for mini_batch, targets in iterate_minibatches(data, batch_size, shuffle=shuffle):
        batch_labels = targets
        batch_data = []
        for bag in mini_batch:
            cross_sents = pad_paragraph(bag[-1])
            d = []
            for x in bag[:-1]:
                if isinstance(x, int):
                    d.append([x])
                else:
                    d.append(x)
            d.append(cross_sents)
            batch_data.append(d)
        yield batch_data, batch_labels


class Data(Dataset):
    def __init__(self, x, labels):
        self.labels = labels
        self.x = x

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return len(self.x)


def weight_init(m):
    '''
    Usage:
        models = Model()
        models.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0.0)
        # nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)


# def load_w2v(wordvec_file, use_norm_emb=False):
#     model = KeyedVectors.load_word2vec_format(wordvec_file, binary=False)
#
#     vocab = model.index2word
#     pad = '<PAD>'
#     unk = '<UNK>'
#     vocab.insert(0, pad)
#     vocab.append(unk)
#     word2ind = {w: i for i, w in enumerate(vocab)}
#     dim = model.vector_size
#     vec = np.vstack([np.zeros(dim, dtype=np.float32), model.vectors, np.random.normal(size=dim).astype(np.float32)])
#     if use_norm_emb:
#         vec = normalize(vec, norm='l2')
#
#     return vec, word2ind, vocab

def load_w2v(wordvec_file, use_norm_emb=False):
    with open(wordvec_file) as fin:
        vec = json.load(fin)
        vec = np.array(vec)
    if use_norm_emb:
        vec = normalize(vec, norm='l2')
    return vec


# def load_vocab(fin):
#     '''
#     :param fin: input a opened file object
#     :return: word2ind, vocab
#     '''
#     vocab = [v[:-1] for v in fin]
#
#     pad = '<PAD>'
#     unk = '<UNK>'
#     vocab.insert(0, pad)
#     vocab.append(unk)
#
#     word2ind = {w: i for i, w in enumerate(vocab)}
#
#     return word2ind, vocab

def load_vocab(word2ind_file):
    with open(word2ind_file) as fin:
        word2ind = json.load(fin)
    vocab = [v for v in word2ind]
    return word2ind, vocab


def load_data_pcnn(feature_rootdir, limit=None):
    '''
        batch_data[0] : entity_ids,
        batch_data[1] : num_instances,
        batch_data[2] : paragraph_ids,
        batch_data[3] : position_features,
        batch_data[4] : entity_positions,
        batch_data[5] : num_abbreviations,
        batch_data[6] : abbbreviation_ids,
        batch_data[7] : abbreviation_match,
        batch_data[8] : entity_char_ids,
        batch_data[9] : paragraph_char_ids,
        batch_data[10] : y,
    '''
    print('load data...')
    data = defaultdict(list)
    for data_type in ['train', 'valid', 'test', 'manual']:
        data_file = feature_rootdir / data_type / 'bags_feature.json'
        total = get_line_count(str(data_file))
        if limit is not None:
            total = limit
        with data_file.open() as fin:
            for line in tqdm(islice(fin, 0, limit), total=total):
                data[data_type].append(json.loads(line))

        num_data = len(data[data_type])
        data[data_type] = list(zip(*[d.values() for d in data[data_type]]))

        none_list = [None] * num_data
        data[data_type][5] = none_list
        data[data_type][6] = none_list
        data[data_type][7] = none_list
        data[data_type][8] = none_list
        data[data_type][9] = none_list
        data[data_type] = [d for d in zip(*data[data_type])]

    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']
    manual_data = data['manual']

    return train_data, valid_data, test_data, manual_data


def load_data(feature_rootdir, data_type, limit=None):
    print('load {} data...'.format(data_type))
    data = []
    data_file = feature_rootdir / data_type / 'bags_feature.json'
    total = get_line_count(str(data_file))
    if limit is not None:
        total = limit
    with data_file.open() as fin:
        for line in tqdm(islice(fin, 0, limit), total=total):
            data.append(json.loads(line))
    # data[data_type] = list(zip(*[d.values() for d in data[data_type]]))

    return data


def load_abbvid_2_typeid(abbv_type_file):
    abbvid_2_typeid = json.load(open(abbv_type_file))
    int_abbvid_2_typeid = {}
    for abbvid, typeid in abbvid_2_typeid.items():
        int_abbvid_2_typeid[int(abbvid)] = int(typeid)
    return int_abbvid_2_typeid


# def load_data(feature_rootdir, limit=None):
#     print('load data...')
#     data = defaultdict(list)
#     for data_type in ['train', 'valid', 'test', 'manual']:
#         data_file = feature_rootdir / data_type / 'bags_feature.json'
#         total = get_line_count(str(data_file))
#         if limit is not None:
#             total = limit
#         with data_file.open() as fin:
#             for line in tqdm(islice(fin, 0, limit), total=total):
#                 data[data_type].append(json.loads(line))
#         # data[data_type] = list(zip(*[d.values() for d in data[data_type]]))
#
#     train_data = data['train']
#     valid_data = data['valid']assert use_gpu
#     test_data = data['test']
#     manual_data = data['manual']
#
#     return train_data, valid_data, test_data, manual_data


def sample_data(data, negative_ratio=5, saved_feature_file=None):
    if saved_feature_file is not None:
        assert data is None
        sampled_data = []
        with open(saved_feature_file) as fin:
            for line in fin:
                sampled_data.append(json.loads(line))
    else:
        labels = np.array([d['y'] for d in data])
        # labels = np.array([d[10] for d in data])
        if negative_ratio is not None:
            pos_inds = np.where(labels == 1)[0]
            neg_inds = np.where(labels == 0)[0]
            sample_neg_inds = np.random.choice(neg_inds, size=len(pos_inds) * negative_ratio)
            sample_inds = np.hstack([pos_inds, sample_neg_inds])
        else:
            sample_inds = np.arange(0, len(data))

        sampled_data = []
        for i in sample_inds:
            sampled_data.append(data[i])
    return sampled_data


# def sample_data(data, negative_ratio=5):
#     # labels = [d['y'] for d in data]
#     labels = np.array([d[10] for d in data])
#     if negative_ratio is not None:
#         pos_inds = np.where(labels == 1)[0]
#         neg_inds = np.where(labels == 0)[0]
#         sample_neg_inds = np.random.choice(neg_inds, size=len(pos_inds) * negative_ratio)
#         sample_inds = np.hstack([pos_inds, sample_neg_inds])
#         # np.random.shuffle(sample_inds)
#     else:
#         sample_inds = np.arange(0, len(data))
#
#     sampled_data = []
#     for i in sample_inds:
#         sampled_data.append(data[i])
#
#     return sampled_data


def prepare_data(batch_data, use_sparse=True, use_gpu=True):
    """
    batch_data[0] : entity_ids,
    batch_data[1] : num_instances,
    batch_data[2] : paragraph_ids,
    batch_data[3] : position_features,
    batch_data[4] : entity_positions,
    batch_data[5] : num_abbreviations,
    batch_data[6] : abbbreviation_ids,
    batch_data[7] : abbreviation_match,
    batch_data[8] : entity_char_ids,
    batch_data[9] : paragraph_char_ids,
    batch_data[10] : y,
    """
    assert use_gpu
    # batch_data = list(zip(*[d.values() for d in batch_data]))
    entity_ids = Variable(torch.LongTensor(batch_data[0]))
    num_instances = Variable(torch.LongTensor(batch_data[1]))

    if not use_sparse:
        paragraph_ids = Variable(torch.LongTensor(batch_data[2]))
        position_features = Variable(torch.LongTensor(batch_data[3]))
        abbreviation_match = Variable(torch.LongTensor(batch_data[7]))
        paragraph_char_ids = Variable(torch.LongTensor(batch_data[9]))
    else:
        # paragraph_ids = [array_from_sparse_dict(d) for d in batch_data[2]]
        # position_features = [array_from_sparse_dict(d) for d in batch_data[3]]
        # abbreviation_match = [array_from_sparse_dict(d) for d in batch_data[7]]
        # paragraph_char_ids = [array_from_sparse_dict(d) for d in batch_data[9]]
        paragraph_ids = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[2]]
        position_features = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[3]]
        abbreviation_match = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[7]]
        paragraph_char_ids = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[9]]

    # abbreviation_ids = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[6]]
    entity_positions = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[4]]
    entity_char_ids = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[8]]
    num_abbreviations = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[5]]

    y = Variable(torch.LongTensor(batch_data[10]))

    entity_ids = entity_ids.cuda()
    num_instances = num_instances.cuda()
    # paragraph_ids = paragraph_ids.cuda()
    # position_features = position_features.cuda()
    # abbreviation_match = abbreviation_match.cuda()
    # paragraph_char_ids = paragraph_char_ids.cuda()
    y = y.cuda()

    batch_data = {
        'entity_ids': entity_ids,
        'num_instances': num_instances,
        'paragraph_ids': paragraph_ids,
        'position_features': position_features,
        'entity_positions': entity_positions,
        'num_abbreviations': num_abbreviations,
        # 'abbreviation_ids': abbreviation_ids,
        'abbreviation_match': abbreviation_match,
        'entity_char_ids': entity_char_ids,
        'paragraph_char_ids': paragraph_char_ids,
        'y': y
    }

    return batch_data


def get_batches(data, batch_size, evaluation=False):
    if not evaluation:
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        data = [data[i] for i in indices]

    for i in range(0, len(data), batch_size):
        batch_size = len(data[i:i + batch_size])
        yield data[i:i + batch_size]


###################
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TFCNN(nn.Module):
    """
        CNN in TF
    """

    def __init__(self, hidden, num_filters, dropout, window=5):
        super(TFCNN, self).__init__()
        self.conv = nn.Conv2d(1, num_filters, (window, hidden), padding=(int(window / 2), 0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.conv(self.dropout(x.unsqueeze(1))).squeeze(3).transpose(1, 2)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
