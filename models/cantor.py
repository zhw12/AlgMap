# coding: utf-8
"""
    CANTOR relation extraction model
"""

import sys
from pathlib import Path

from torch.autograd import Variable

sys.path.append(str(Path(__file__).absolute().parent.parent))
from models.basic_module import BasicModule
from models.transformer import TransformerBlock
from common.util import array_from_sparse_dict

import torch.nn as nn

import torch


class PCNN(nn.Module):
    """
        single sentence
    """

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

        pf1 = pf[:, 0]  # batch x max_len
        pf2 = pf[:, 1]

        pf1_emb = self.pos1_lookup(pf1)  # batch x max_len x pos_emb_size
        pf2_emb = self.pos2_lookup(pf2)

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)  # batch x max_len x total_emb_dim
        x = x.unsqueeze(1)
        x = self.dropout(x)

        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs]  # conv(x) batch x total_filters_num x max_len x 1
        x = [self.piecewise_max_pooling(i, epos) for i in x]  # x[0] batch x (filters_num*3)

        x = torch.cat(x, 1)  # batch x (filters_num*3)*len(kernel_sizes)
        x = self.dropout(x)
        x = self.linear(x)  # batch x rel_num
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

        batch_size = batch_data['entity_ids'].shape[0]
        for i in range(batch_size):
            num_instances = batch_data['num_instances'][i]
            max_ins_id = 0
            if num_instances > 1:
                label = batch_data['y'][i]
                data = [batch_data['entity_ids'][i], num_instances, batch_data['paragraph_ids'][i],
                        batch_data['position_features'][i],
                        batch_data['entity_positions'][i]]

                pcnn_out = pcnn_model(data)

                max_ins_id = torch.max(pcnn_out[:, label], 0)[1]
                max_ins_id = max_ins_id.data.cpu().numpy()
            select_eids.append(batch_data['entity_ids'][i])
            select_num.append(num_instances)
            select_sent.append(batch_data['paragraph_ids'][i][max_ins_id])
            select_pf.append(
                batch_data['position_features'][i][max_ins_id].long())  # .long handle padding 0 which is float
            select_epos.append(batch_data['entity_positions'][i][max_ins_id].long())

        select_bags = [select_eids, select_num, select_sent, select_pf, select_epos]

        pcnn_model.train()
        return select_bags


class CROSS(nn.Module):
    """
        cross sentence
    """

    def __init__(self, opt):
        super(CROSS, self).__init__()
        self.model_name = 'CROSS'
        feature_dim = opt.word_emb_size + 2 * opt.pos_emb_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, opt.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in opt.filter_kernel_sizes])

        self.word_lookup = nn.Embedding(opt.word_size, opt.word_emb_size)
        self.pos1_lookup = nn.Embedding(opt.pos_size, opt.pos_emb_size)
        self.pos2_lookup = nn.Embedding(opt.pos_size, opt.pos_emb_size)
        self.char_lookup = nn.Embedding(opt.char_size, opt.char_emb_size)
        # self.word_lookup.requires_grad = False

        self.dropout = nn.Dropout(opt.dropout_rate)
        self.linear = nn.Linear(opt.total_filters_num, opt.rel_num)

        hidden = opt.word_emb_size + opt.pos_emb_size * 2
        abbv_hidden = opt.word_emb_size + opt.pos_emb_size * 2 + opt.char_emb_size
        n_layers = opt.tf_layers
        attn_heads = 4

        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.use_abbv_type = opt.use_abbv_type

        self.feed_forward_hidden = hidden * 4

        # transformer blocks for self-attention
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, opt.dropout_rate) for _ in range(n_layers)])

        self.transformer_blocks_abbv = nn.ModuleList(
            [TransformerBlock(abbv_hidden, 2, abbv_hidden * 4, opt.dropout_rate, 1) for _ in range(n_layers)])

        self.fusion_layer = nn.Linear(hidden * 1 + abbv_hidden * 2 + 2 * opt.word_emb_size + 2 * opt.char_emb_size,
                                      opt.rel_num)
        if opt.use_abbv_type:
            self.typing_linear = nn.Linear(abbv_hidden, opt.num_abbv_types)

        self.instance_batch_size = 64  # prevent a very big bag causing GPU memory overflow

        self.char_emb_size = opt.char_emb_size
        self.char_cnn_filters = 50
        self.char_cnn = nn.Conv2d(1, self.char_cnn_filters, (7, opt.char_emb_size), padding=(int(7 / 2), 0))

    def forward(self, x):
        eids, _, sent, pf, epos, e_charids, abbv_match, p_char_ids = x
        word_emb = self.word_lookup(sent)
        num_instances = sent.shape[0]
        max_sent_len = sent.shape[1]
        max_char_len = p_char_ids.shape[-1]

        e_charids = e_charids.expand(num_instances, 2, e_charids.shape[-1])
        e_charids1 = e_charids[:, 0, :]
        e_charids2 = e_charids[:, 1, :]

        entity_char_embeded1 = self.char_lookup(e_charids1).unsqueeze(
            1)  # num_instance x 1 x max_char_len x char_emb_size
        entity_char_embeded2 = self.char_lookup(e_charids2).unsqueeze(1)

        entity_char_embeded1 = self.char_cnn(entity_char_embeded1).squeeze(3)  # num_instance x 1 x max_char_len
        entity_char_embeded2 = self.char_cnn(entity_char_embeded2).squeeze(3)

        char_embeded = torch.cat([entity_char_embeded1, entity_char_embeded2], 1)  # num_instance x 2 x max_char_len
        char_embeded = torch.max(char_embeded, 2)[0]  # num_instance x 2

        p_char_embeded = self.char_lookup(p_char_ids)  # num_instance x max_sent_len x max_char_len x char_emb_size
        p_char_embeded = p_char_embeded.view(num_instances * max_sent_len, 1, max_char_len, -1)
        p_char_embeded = self.char_cnn(p_char_embeded).squeeze(3)
        p_char_embeded = torch.max(p_char_embeded, 2)[0]
        p_char_embeded = p_char_embeded.view(num_instances, max_sent_len, -1)

        entity_embeded = self.word_lookup(eids)  # bag[0] : 1 x 2;  2 x word_emb_size
        entity_embeded = entity_embeded.expand(num_instances, 2, word_emb.shape[-1])
        entity_embeded = entity_embeded.view(num_instances, -1)

        pf1 = pf[:, 0]  # batch * max_len
        pf2 = pf[:, 1]

        pf1_emb = self.pos1_lookup(pf1)  # batch * max_len * pos_emb_size
        pf2_emb = self.pos2_lookup(pf2)

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)  # batch * max_len * total_emb_dim
        mask = (sent > 0).unsqueeze(1).repeat(1, sent.size(1), 1).unsqueeze(1)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        abbv_x = torch.cat([word_emb, pf1_emb, pf2_emb, p_char_embeded], 2)
        abbv_mask = (abbv_match > 0).unsqueeze(1).repeat(1, abbv_match.size(1), 1).unsqueeze(1)
        for transformer in self.transformer_blocks_abbv:
            abbv_x = transformer.forward(abbv_x, abbv_mask)

        abbv_e1 = torch.cat([abbv_x[i, j, :].unsqueeze(0) for i, j in zip(torch.arange(num_instances), epos[:, 0])])
        abbv_e2 = torch.cat([abbv_x[i, j, :].unsqueeze(0) for i, j in zip(torch.arange(num_instances), epos[:, 1])])
        x = torch.cat([x[:, 0, :], entity_embeded, char_embeded, abbv_e1, abbv_e2], 1)

        x = self.fusion_layer(x)

        if self.use_abbv_type:
            type_pred1 = self.typing_linear(abbv_e1).unsqueeze(1)
            type_pred2 = self.typing_linear(abbv_e2).unsqueeze(1)
            type_pred = torch.cat([type_pred1, type_pred2], 1)
        else:
            type_pred = None

        return x, type_pred

    # select instance for pcnn
    def select_instance_for_inner(self, batch_data):
        cross_model = self
        cross_model.eval()
        select_eids = []
        select_num = []
        select_sent = []
        select_pf = []
        select_epos = []
        select_e_char_ids = []
        select_abbv_match = []
        select_p_char_ids = []

        batch_size = batch_data['entity_ids'].shape[0]
        for i in range(batch_size):
            num_instances = batch_data['num_instances'][i]
            max_ins_id = 0
            if num_instances > 1:
                # eids, _, sent, pf, epos = x
                label = batch_data['y'][i]
                # data = [Variable(torch.LongTensor(x)).cuda() for x in bag[:-2]]

                cross_out = None
                for j in range(0, num_instances, self.instance_batch_size):
                    actual_instance_batch_size = batch_data['paragraph_ids'][i][j:j + self.instance_batch_size].shape[0]
                    data = [batch_data['entity_ids'][i], actual_instance_batch_size,
                            batch_data['paragraph_ids'][i][j: j + actual_instance_batch_size],
                            batch_data['position_features'][i][j:j + actual_instance_batch_size],
                            batch_data['entity_positions'][i][j:j + actual_instance_batch_size],
                            batch_data['entity_char_ids'][i],
                            batch_data['abbreviation_match'][i][j:j + actual_instance_batch_size],
                            batch_data['paragraph_char_ids'][i][j:j + actual_instance_batch_size]]
                    if cross_out is None:
                        cross_out, _ = cross_model(data)
                        cross_out = cross_out.detach()
                    else:
                        b_out, _ = cross_model(data)
                        b_out = b_out.detach()
                        cross_out = torch.cat([cross_out, b_out], 0)

                max_ins_id = torch.max(cross_out[:, label], 0)[1]
                max_ins_id = max_ins_id.data.cpu().numpy()
            select_eids.append(batch_data['entity_ids'][i])
            select_num.append(num_instances)
            select_sent.append(batch_data['paragraph_ids'][i][max_ins_id])
            select_pf.append(batch_data['position_features'][i][max_ins_id].long())
            select_epos.append(batch_data['entity_positions'][i][max_ins_id].long())
            select_e_char_ids.append(batch_data['entity_char_ids'][i])
            select_abbv_match.append(batch_data['abbreviation_match'][i][max_ins_id])
            select_p_char_ids.append(batch_data['paragraph_char_ids'][i][max_ins_id])

        cross_model.train()
        return select_eids, select_num, select_sent, select_pf, select_epos, select_e_char_ids, select_abbv_match, select_p_char_ids


class Model(BasicModule):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.cross_model = CROSS(opt)
        self.pcnn_model = PCNN(opt)
        self.weight = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return x

    def get_predictions(self, batch_data):
        pcnn_model = self.pcnn_model
        select_bags = pcnn_model.select_instance_for_inner(batch_data)
        data = [torch.stack(x, dim=0) for x in select_bags]
        # data = [np.stack(x, axis=0) for x in select_bags]
        # data = [Variable(torch.LongTensor(x)).cuda() for x in data]
        pcnn_out = pcnn_model(data)
        del data

        cross_model = self.cross_model
        select_eids, select_num, select_sent, select_pf, select_epos, select_e_char_ids, select_abbv_match, select_p_char_ids = cross_model.select_instance_for_inner(
            batch_data)

        # data = [np.stack(x, axis=0) for x in
        #         [select_eids, select_num, select_sent, select_pf, select_epos, select_e_char_ids, select_abbv_match,
        #          select_p_char_ids]]
        # data = [Variable(torch.LongTensor(x)).cuda() for x in data]

        data = [torch.stack(x, dim=0) for x in
                [select_eids, select_num, select_sent, select_pf, select_epos, select_e_char_ids, select_abbv_match,
                 select_p_char_ids]]

        cross_out, type_pred = cross_model(data)

        del data  # avoid memory leak
        del select_eids, select_num, select_sent, select_pf, select_epos

        weight_clip = torch.clamp(self.weight, 0, 1)
        final_out = weight_clip * pcnn_out + (1 - weight_clip) * cross_out

        final_type_pred = type_pred

        return final_out, final_type_pred

    def prepare_data(self, batch_data, use_gpu=True, abbvid_2_typeid=None):
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

        batch_data = list(zip(*[d.values() for d in batch_data]))
        type_labels = None
        if abbvid_2_typeid is not None:
            for eids in batch_data[0]:
                type_labels.append([abbvid_2_typeid.get(eid, 0) for eid in eids])
            type_labels = Variable(torch.LongTensor(type_labels))
            if use_gpu:
                type_labels = type_labels.cuda()

        # batch_data = list(zip(*batch_data))


        # if not use_sparse:
        #     paragraph_ids = Variable(torch.LongTensor(batch_data[2]))
        #     position_features = Variable(torch.LongTensor(batch_data[3]))
        #     abbreviation_match = Variable(torch.LongTensor(batch_data[7]))
        #     paragraph_char_ids = Variable(torch.LongTensor(batch_data[9]))
        # elif use_gpu:

        if use_gpu:
            entity_ids = Variable(torch.LongTensor(batch_data[0])).cuda()
            num_instances = Variable(torch.LongTensor(batch_data[1])).cuda()
            paragraph_ids = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[2]]
            position_features = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[3]]
            abbreviation_match = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[7]]
            paragraph_char_ids = [Variable(torch.from_numpy(array_from_sparse_dict(d))).cuda() for d in batch_data[9]]
            # abbreviation_ids = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[6]]
            entity_positions = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[4]]
            entity_char_ids = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[8]]
            # num_abbreviations = [Variable(torch.LongTensor(d)).cuda() for d in batch_data[5]]
            y = Variable(torch.LongTensor(batch_data[10])).cuda()
        else:
            entity_ids = Variable(torch.LongTensor(batch_data[0]))
            num_instances = Variable(torch.LongTensor(batch_data[1]))
            entity_ids = Variable(torch.LongTensor(batch_data[0]))
            num_instances = Variable(torch.LongTensor(batch_data[1]))
            paragraph_ids = [Variable(torch.from_numpy(array_from_sparse_dict(d))) for d in batch_data[2]]
            position_features = [Variable(torch.from_numpy(array_from_sparse_dict(d))) for d in batch_data[3]]
            abbreviation_match = [Variable(torch.from_numpy(array_from_sparse_dict(d))) for d in batch_data[7]]
            paragraph_char_ids = [Variable(torch.from_numpy(array_from_sparse_dict(d))) for d in batch_data[9]]

            entity_positions = [Variable(torch.LongTensor(d)) for d in batch_data[4]]
            entity_char_ids = [Variable(torch.LongTensor(d)) for d in batch_data[8]]
            y = Variable(torch.LongTensor(batch_data[10]))



        batch_data = {
            'entity_ids': entity_ids,
            'num_instances': num_instances,
            'paragraph_ids': paragraph_ids,
            'position_features': position_features,
            'entity_positions': entity_positions,
            # 'num_abbreviations': num_abbreviations,
            # 'abbreviation_ids': abbreviation_ids,
            'abbreviation_match': abbreviation_match,
            'entity_char_ids': entity_char_ids,
            'paragraph_char_ids': paragraph_char_ids,
            'y': y,
            'type_labels': type_labels
        }

        return batch_data
