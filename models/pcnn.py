# coding: utf-8

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

sys.path.append(str(Path(__file__).absolute().parent.parent))
from models.basic_module import BasicModule
from common.util import array_from_sparse_dict


class PCNN(nn.Module):
    def __init__(self, opt):
        super(PCNN, self).__init__()
        self.model_name = 'PCNN'
        feature_dim = opt.word_emb_size + 2 * opt.pos_emb_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, opt.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in opt.filter_kernel_sizes])

        self.word_lookup = nn.Embedding(opt.word_size, opt.word_emb_size)
        self.pos1_lookup = nn.Embedding(opt.pos_size, opt.pos_emb_size)
        self.pos2_lookup = nn.Embedding(opt.pos_size, opt.pos_emb_size)
        # self.word_lookup.requires_grad = False

        self.dropout = nn.Dropout(opt.dropout_rate)
        self.linear = nn.Linear(opt.total_filters_num, opt.rel_num)

    def init_model_weight(self):
        # init conv layer
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def piecewise_max_pooling(self, x, epos):
        batch_res = []
        for i in range(len(x)):
            ins = x[i]  # total_filters_num * max_len
            pool = epos[i] + 1
            seg1 = torch.max(ins[:, :pool[0]], 1)[0].unsqueeze(1)
            seg2 = torch.max(ins[:, pool[0]:pool[1]], 1)[0].unsqueeze(1)
            seg3 = torch.max(ins[:, pool[1]:], 1)[0].unsqueeze(1)
            piecewise_max_pool = torch.cat([seg1, seg2, seg3], 1).view(1, -1)
            batch_res.append(piecewise_max_pool)
        out = torch.cat(batch_res, 0)
        return out

    def forward(self, x):
        eids, _, sent, pf, epos = x
        word_emb = self.word_lookup(sent)

        pf1 = pf[:, 0]  # batch * max_len
        pf2 = pf[:, 1]

        pf1_emb = self.pos1_lookup(pf1)  # batch * max_len * pos_emb_size
        pf2_emb = self.pos2_lookup(pf2)

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)  # batch * max_len * total_emb_dim
        x = x.unsqueeze(1)
        x = self.dropout(x)

        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs]  # conv(x) batch * total_filters_num * max_len * 1
        x = [self.piecewise_max_pooling(i, epos) for i in x]  # x[0] batch * (filters_num*3)
        # x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # batch * (filters_num*3)*len(kernel_sizes)
        x = self.dropout(x)
        x = self.linear(x)  # batch * rel_num
        return x

    # select instance for pcnn
    def select_instance_for_inner(self, batch_data):
        pcnn_model = self
        pcnn_model.eval()
        select_eids = []
        select_num = []
        select_sent = []
        select_pf = []
        select_epos = []
        # for bag, label in zip(batch_data, batch_labels):
        #     num_inner_sents = bag[1][0]
        #     max_ins_id = 0
        #     if num_inner_sents > 1:
        #         data = [Variable(torch.LongTensor(x)).cuda() for x in bag[:-2]]
        #         pcnn_out = pcnn_model(data)
        #         max_ins_id = torch.max(pcnn_out[:, label], 0)[1]
        #         max_ins_id = max_ins_id.data.cpu().numpy()
        #     select_eids.append(bag[0])
        #     select_num.append(bag[1][0])
        #     select_sent.append(bag[2][max_ins_id])
        #     select_pf.append(bag[3][max_ins_id])
        #     select_epos.append(bag[4][max_ins_id])

        batch_size = batch_data['entity_ids'].shape[0]
        for i in range(batch_size):
            num_instances = batch_data['num_instances'][i]
            max_ins_id = 0
            if num_instances > 1:
                # eids, _, sent, pf, epos = x
                label = batch_data['y'][i]
                data = [batch_data['entity_ids'][i], num_instances, batch_data['paragraph_ids'][i],
                        batch_data['position_features'][i],
                        batch_data['entity_positions'][i]]
                # data = [Variable(torch.LongTensor(x)).cuda() for x in bag[:-2]]

                pcnn_out = pcnn_model(data)

                max_ins_id = torch.max(pcnn_out[:, label], 0)[1]
                max_ins_id = max_ins_id.data.cpu().numpy()
            select_eids.append(batch_data['entity_ids'][i])
            select_num.append(num_instances)
            select_sent.append(batch_data['paragraph_ids'][i][max_ins_id])
            select_pf.append(batch_data['position_features'][i][max_ins_id])
            select_epos.append(batch_data['entity_positions'][i][max_ins_id])

        select_bags = [select_eids, select_num, select_sent, select_pf, select_epos]
        # data = [torch.stack(x, dim=0) for x in select_bags]

        pcnn_model.train()
        return select_bags


class Model(BasicModule):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.pcnn_model = PCNN(opt)

    def forward(self, x):
        return x

    def get_predictions(self, batch_data):
        pcnn_model = self.pcnn_model
        select_bags = pcnn_model.select_instance_for_inner(batch_data)
        data = [np.stack(x, axis=0) for x in select_bags]
        data = [Variable(torch.LongTensor(x)).cuda() for x in data]
        pcnn_out = pcnn_model(data)

        del data  # avoid memory leak

        final_out = pcnn_out

        return final_out

    def prepare_data(self, batch_data, use_sparse=True, use_gpu=True):
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
        batch_data = list(zip(*[d.values() for d in batch_data]))
        # batch_data = list(zip(*batch_data))
        entity_ids = Variable(torch.LongTensor(batch_data[0]))
        num_instances = Variable(torch.LongTensor(batch_data[1]))

        if not use_sparse:
            paragraph_ids = Variable(torch.LongTensor(batch_data[2]))
            position_features = Variable(torch.LongTensor(batch_data[3]))
            # abbreviation_match = Variable(torch.LongTensor(batch_data[7]))
            # paragraph_char_ids = Variable(torch.LongTensor(batch_data[9]))
        else:
            # paragraph_ids = [array_from_sparse_dict(d) for d in batch_data[2]]
            # position_features = [array_from_sparse_dict(d) for d in batch_data[3]]
            # abbreviation_match = [array_from_sparse_dict(d) for d in batch_data[7]]
            # paragraph_char_ids = [array_from_sparse_dict(d) for d in batch_data[9]]
            paragraph_ids = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[2]]
            position_features = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[3]]
            # abbreviation_match = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[7]]
            # paragraph_char_ids = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[9]]

        # abbreviation_ids = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[6]]
        entity_positions = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[4]]
        # entity_char_ids = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[8]]
        # num_abbreviations = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[5]]

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
            # 'num_abbreviations': num_abbreviations,
            # 'abbreviation_ids': abbreviation_ids,
            # 'abbreviation_match': abbreviation_match,
            # 'entity_char_ids': entity_char_ids,
            # 'paragraph_char_ids': paragraph_char_ids,
            'y': y
        }

        return batch_data

#
# # test
# def eval(model, data, criterion):
#     model.eval()
#
#     all_pred_p = []
#     all_pred_y = []
#     all_true_y = []
#
#     g = gen_minibatch(data, 1, shuffle=False)
#     total_losses = []
#     for batch_data, batch_labels in tqdm(g, total=int(len(data))):
#         unnormalized_out = get_predictions(model, batch_data, batch_labels)
#         out = F.softmax(unnormalized_out, 1)
#
#         loss = criterion(unnormalized_out, Variable(torch.LongTensor(batch_labels)).cuda())
#         total_losses.append(loss.item())
#         pred = out[0]
#         pred_p = pred[1].item()
#         pred_y = int(pred[1] > pred[0])
#
#         all_pred_p.append(pred_p)
#         all_pred_y.append(pred_y)
#         # all_true_y.extend(batch_labels.data.cpu().tolist())
#         all_true_y.extend(batch_labels)
#
#     avg_loss = torch.mean(torch.Tensor(total_losses))
#
#     model.train()
#     return all_true_y, all_pred_y, all_pred_p, avg_loss
