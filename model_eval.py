# coding: utf-8

import importlib
import re

import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score
from torch import nn

# sys.path.append("..")
from config import Config
from models.util import sample_data, load_data, load_abbvid_2_typeid
from train import eval
import json


def find_best_with_root_dir(root_dir, max_epoch_used=16):
    best_path = None
    max_epoch = -1
    for l in root_dir.iterdir():
        epoch = int(re.findall('\d+', l.name.rsplit('_', 1)[1])[0])
        if epoch > max_epoch and epoch <= max_epoch_used:
            best_path = l
            max_epoch = epoch
    return best_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("model_name", help="model name")
    parser.add_argument("feature_type", help="feature type")
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true')
    parser.add_argument('--no-debug_mode', dest='debug_mode', action='store_false')
    parser.set_defaults(debug_mode=False)
    args = parser.parse_args()

    model_f = importlib.import_module('models.' + args.model_name)
    opt = Config(args.dataset, args.model_name, args.feature_type)
    data_limit = 10000 if args.debug_mode else None

    best_model_file = find_best_with_root_dir(opt.checkpoints_dir)
    print(best_model_file)
    model = model_f.Model(opt)
    model.load(best_model_file)
    if opt.use_gpu:
        model.cuda()
    opt.result_dir.mkdir(parents=True, exist_ok=True)

    abbvid_2_typeid = None
    if opt.use_abbv_type:
        abbvid_2_typeid = load_abbvid_2_typeid(opt.abbv_type_file)
    criterion = nn.CrossEntropyLoss()

    word2ind = json.load(opt.word2ind_file.open())
    vocab = [w for w in word2ind]


    def ids2word(ids):
        return [vocab[i] for i in ids]


    for type in ['valid', 'test', 'train']:
        print('evaluating {} ...'.format(type))
        test_data = load_data(opt.feature_dir, type, limit=data_limit)

        np.random.seed(0)

        # all data
        negative_ratio = None
        sample_test_data = sample_data(test_data, negative_ratio=negative_ratio)
        if negative_ratio is None:
            type += '_all'

        batch_size = 128
        all_true_y, all_pred_y, all_pred_p, loss, type_loss, type_true_y, type_pred_y = eval(model, sample_test_data,
                                                                                             criterion,
                                                                                             batch_size, opt.use_gpu,
                                                                                             abbvid_2_typeid)

        metric_dict = classification_report(all_true_y, all_pred_y, output_dict=True)
        ap = average_precision_score(all_true_y, all_pred_p)

        all_pre, all_rec, thresholds = precision_recall_curve(all_true_y, all_pred_p)

        with (opt.result_dir / '{}_{}_report.txt'.format(type, opt.model_name)).open(mode='w') as fout:
            fout.write(classification_report(all_true_y, all_pred_y, output_dict=False))
            fout.write('\n')
            fout.write('Average Precision:{}'.format(ap))
            fout.write('\n')
            if opt.use_abbv_type:
                fout.write('typing loss: {}\n'.format(type_loss))
                fout.write(classification_report(type_true_y, type_pred_y, output_dict=False))

        with (opt.result_dir / '{}_{}_predictions.txt'.format(type, opt.model_name)).open(mode='w') as fout:
            for i, r in enumerate(zip(all_true_y, all_pred_y, all_pred_p)):
                y, pred_y, pred_p = r
                eids = sample_test_data[i]['entity_ids']
                e1, e2 = ids2word(eids)
                fout.write('{} {} {} {} {}'.format(e1, e2, y, pred_y, pred_p))
                fout.write('\n')

        with (opt.result_dir / '{}_{}_pr.txt'.format(type, opt.model_name)).open(mode='w') as fout:
            for p, r in zip(all_pre, all_rec):
                fout.write('{} {}\n'.format(p, r))
