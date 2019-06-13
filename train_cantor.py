# coding: utf-8
from functools import partial

import argparse
import torch

from config import Config
from models.cantor import Model
from models.util import load_w2v, weight_init
from train import train

print = partial(print, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("modelname", help="model name")
    parser.add_argument("feature_type", help="feature_type")
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true')
    parser.add_argument('--no-debug_mode', dest='debug_mode', action='store_false')
    parser.set_defaults(debug_mode=False)
    args = parser.parse_args()

    opt = Config(args.dataset, args.modelname, args.feature_type)
    opt.debug_mode = args.debug_mode

    print('user config:')
    for k, v in opt.__dict__.items():
        if not k.startswith('_'):
            print(k, getattr(opt, k))

    model = Model(opt)
    model.apply(weight_init)

    if opt.use_pretrained:
        print('Loading pretrained embedding...')
        pretrained_wordvec = load_w2v(str(opt.wordvec_file), use_norm_emb=opt.use_norm_emb)
        model.cross_model.word_lookup.weight.data.copy_(torch.from_numpy(pretrained_wordvec))

    model.cross_model.char_lookup.weight.data[0].fill_(0)  # character <PAD> embedding

    if opt.use_gpu:
        model.cuda()

    train(model, opt)
