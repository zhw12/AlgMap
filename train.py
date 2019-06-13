# coding: utf-8
from functools import partial

import fire
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score
from torch import nn, optim
from tqdm import tqdm

from common.util import now
from models.util import save_pr, sample_data, get_batches, load_data, load_abbvid_2_typeid, load_vocab

print = partial(print, flush=True)


def print_and_save_metric(model, sampled_data, criterion, epoch, max_num_epochs, data_type, batch_size,
                          use_gpu, abbvid_2_typeid):
    all_true_y, all_pred_y, all_pred_p, loss, type_loss, type_true_y, type_pred_y = eval(model, sampled_data, criterion,
                                                                                         batch_size, use_gpu,
                                                                                         abbvid_2_typeid)

    metric_dict = classification_report(all_true_y, all_pred_y, output_dict=True)
    ap = average_precision_score(all_true_y, all_pred_p)

    print(('{} Epoch {}/{}: {} loss: {};'.format(
        now(), epoch + 1, max_num_epochs, data_type, loss)))
    print(classification_report(all_true_y, all_pred_y, output_dict=False))
    print('{}, Average Precision:{}'.format(data_type, ap))
    if abbvid_2_typeid is not None:
        print('typing loss', type_loss)
        print(classification_report(type_true_y, type_pred_y, output_dict=False))
    res = {'metric_dict': metric_dict,
           'ap': ap,
           'loss': loss,
           'all_true_y': all_true_y,
           'all_pred_y': all_pred_y,
           'all_pred_p': all_pred_p}

    return res


def train(model, opt):
    print(model)
    data_limit = 1000 if opt.debug_mode else None
    # use_tensor_board = False

    opt.result_dir.mkdir(parents=True, exist_ok=True)
    opt.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)

    test_data = load_data(opt.feature_dir, 'test', limit=data_limit)
    # manual_data = load_data(opt.feature_dir, 'manual', limit=data_limit)

    abbvid_2_typeid = None
    if opt.use_abbv_type:
        abbvid_2_typeid = load_abbvid_2_typeid(opt.abbv_type_file)

    # if opt.use_pre_sample:
    #     sample_valid_data = sample_data(None, negative_ratio=opt.negative_ratio,
    #                                     saved_feature_file=(opt.feature_dir / 'valid/sample1_bags_feature.json'))
    # else:
    train_data = load_data(opt.feature_dir, 'train', limit=data_limit)
    valid_data = load_data(opt.feature_dir, 'valid', limit=data_limit)
    sample_valid_data = sample_data(valid_data, negative_ratio=opt.negative_ratio)

    if not opt.debug_mode:
        valid_data = load_data(opt.feature_dir, 'valid', limit=data_limit)
        # sample_valid_data = sample_data(valid_data, negative_ratio=opt.negative_ratio)
        sample_all_valid_data = sample_data(valid_data, negative_ratio=opt.test_negative_ratio)
    else:
        sample_all_valid_data = sample_valid_data

    sample_test_data = sample_data(test_data, negative_ratio=opt.test_negative_ratio)
    # sample_manual_data = sample_data(manual_data, negative_ratio=None)

    criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss()
    all_parameters = model.parameters()
    optimizer = optim.Adam(all_parameters, lr=opt.learning_rate)
    optimizer.zero_grad()

    # train
    eval_batch_size = 128
    previous_valid_f1 = -1
    best_epoch = 1
    max_num_epochs = opt.max_num_epochs
    global_step = 0

    for epoch in range(max_num_epochs):
        total_losses = []

        # if opt.use_pre_sample:
        #     sample_train_data = sample_data(None, negative_ratio=opt.negative_ratio, saved_feature_file=(
        #             opt.feature_dir / 'train/sample{}_bags_feature.json'.format(epoch + 1)))
        # else:
        sample_train_data = sample_data(train_data, negative_ratio=opt.negative_ratio)

        g = get_batches(sample_train_data, opt.batch_size)
        for batch_data in tqdm(g, total=int(len(sample_train_data) / opt.batch_size)):
            batch_data = model.prepare_data(batch_data, use_gpu=opt.use_gpu, abbvid_2_typeid=abbvid_2_typeid)
            optimizer.zero_grad()
            out, type_out = model.get_predictions(batch_data)

            batch_labels = batch_data['y']
            batch_type_labels = batch_data['type_labels']

            if batch_type_labels is not None:
                loss_type = criterion(type_out[:, 0, :], batch_type_labels[:, 0]) + criterion(type_out[:, 1, :],
                                                                                              batch_type_labels[:, 1])
                match_mask = batch_labels
                match_mask = match_mask.unsqueeze(1).expand_as(type_out[:, 0, :]).float()
                t1 = type_out[:, 0, :] * match_mask
                t2 = type_out[:, 1, :] * match_mask
                t1 = torch.log_softmax(t1, 1)
                t2 = torch.softmax(t2, 1)

                loss_type_match = kl_criterion(t1, t2)
            else:
                loss_type = 0
                loss_type_match = 0

            loss_re = criterion(out, batch_labels)  # avg loss of the batch

            loss = opt.lambda_re * loss_re + opt.lambda_type * loss_type + opt.lambda_type_match * loss_type_match
            loss.backward()
            optimizer.step()
            total_losses.append(loss.item())
            global_step += 1

            if global_step % opt.step_print_train_loss == 0:
                avg_loss = torch.mean(torch.Tensor(total_losses))
                print(('{} Epoch {}/{}: train loss: {};'.format(
                    now(), epoch + 1, max_num_epochs, avg_loss)))

        avg_loss = torch.mean(torch.Tensor(total_losses))
        print(('{} Epoch {}/{}: train loss: {};'.format(
            now(), epoch + 1, max_num_epochs, avg_loss)))

        # res = print_and_save_metric(model, sample_valid_data, criterion, epoch, max_num_epochs, 'valid')
        res = print_and_save_metric(model, sample_all_valid_data, criterion, epoch, max_num_epochs, 'valid_all',
                                    eval_batch_size, opt.use_gpu, abbvid_2_typeid)

        if previous_valid_f1 < res['metric_dict']['1']['f1-score']:
            previous_valid_f1 = res['metric_dict']['1']['f1-score']
            res = print_and_save_metric(model, sample_test_data, criterion, epoch, max_num_epochs, 'test',
                                        eval_batch_size, opt.use_gpu, abbvid_2_typeid)
            all_pre, all_rec, thresholds = precision_recall_curve(res['all_true_y'], res['all_pred_p'])

            best_epoch = epoch + 1
            print("save models, best epoch: epoch {};".format(best_epoch))
            if not opt.debug_mode:
                save_pr(str(opt.result_dir), model.model_name, epoch, all_pre, all_rec)
                model.save(opt.checkpoints_dir / '{}_{}.pth'.format(opt.print_opt, str(epoch + 1)))

            # _ = print_and_save_metric(model, sample_manual_data, criterion, epoch, max_num_epochs, 'manual',
            #                           eval_batch_size, abbvid_2_typeid)

    print('best epoch is epoch {}.'.format(best_epoch))


# test
def eval(model, data, criterion, batch_size=1, use_gpu=True, abbvid_2_typeid=None):
    model.eval()
    all_pred_p = []
    all_pred_y = []
    all_true_y = []

    type_true_y = []
    type_pred_y = []

    g = get_batches(data, batch_size, evaluation=True)
    total_losses = []
    total_type_losses = []
    for batch_data in tqdm(g, total=int(len(data) / batch_size)):
        batch_data = model.prepare_data(batch_data, use_gpu=use_gpu, abbvid_2_typeid=abbvid_2_typeid)
        batch_labels = batch_data['y']
        batch_type_labels = batch_data['type_labels']
        unnormalized_out, unnormalized_type_out = model.get_predictions(batch_data)
        out = F.softmax(unnormalized_out, 1)

        if abbvid_2_typeid is not None:
            type_out = F.softmax(unnormalized_type_out, 2)
            loss_type = criterion(unnormalized_type_out[:, 0, :], batch_type_labels[:, 0]) + criterion(
                unnormalized_type_out[:, 1, :],
                batch_type_labels[:, 1])
            total_type_losses.append(loss_type.item())

            t_true_y = batch_type_labels.view(-1).cpu().detach().numpy()
            t_pred_y = torch.max(type_out.view(-1, type_out.shape[-1]), 1)[1].cpu().detach().numpy()

            type_true_y.extend(t_true_y.tolist())
            type_pred_y.extend(t_pred_y.tolist())
        else:
            loss_type = 0
            total_type_losses.append(loss_type)
            type_true_y = None
            type_pred_y = None

        loss_re = criterion(unnormalized_out, batch_labels)  # avg loss of the batch
        total_losses.append(loss_re.item())

        pred_p = out[:, 1].cpu().detach().numpy().tolist() # probability of label 1 in binary classification
        pred_y = (out[:, 1] > out[:, 0]).cpu().numpy().tolist()
        all_pred_p.extend(pred_p)
        all_pred_y.extend(pred_y)

        all_true_y.extend(batch_labels.cpu().numpy().tolist())

    avg_loss = torch.mean(torch.Tensor(total_losses))
    avg_type_loss = torch.mean(torch.Tensor(total_type_losses))
    model.train()
    return all_true_y, all_pred_y, all_pred_p, avg_loss, avg_type_loss, type_true_y, type_pred_y


if __name__ == '__main__':
    fire.Fire()
