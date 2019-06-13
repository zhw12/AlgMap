import ujson as json
import warnings
from pathlib import Path

from models.util import load_abbvid_2_typeid


class Config:
    def __init__(self, dataset, model_name, feature_type):
        self.dataset = dataset
        self.model_name = model_name
        self.feature_type = feature_type  # paragraph/standard

        self.limit = 80  # position feature limit

        self.word_emb_size = 100
        self.pos_emb_size = 10
        self.char_emb_size = 50

        self.filter_kernel_sizes = [7]
        self.filters_num = 200

        self.rel_num = 2
        self.tf_layers = 1

        self.lambda_re = 1
        self.lambda_type = 1
        self.lambda_type_match = 0.05

        self.use_pretrained = True
        self.use_norm_emb = False
        self.use_gpu = True
        self.negative_ratio = 5
        self.test_negative_ratio = None

        self.learning_rate = 0.001
        self.dropout_rate = 0.3
        self.batch_size = 32
        self.max_num_epochs = 16

        self.step_print_train_loss = 500

        self.root_dir = Path('/scratch/home/hanwen/algmap/')
        self.dataset_dir = self.root_dir / 'data' / self.dataset
        self.result_dir = self.root_dir / 'out' / self.dataset / self.model_name
        self.checkpoints_dir = self.root_dir / 'checkpoints' / self.dataset / self.model_name

        self.feature_dir = self.dataset_dir / self.feature_type
        # self.vocab_file = self.dataset_dir / 'vocab.txt'
        self.word2ind_file = self.dataset_dir / 'word2id.json'
        self.wordvec_file = self.dataset_dir / 'word_emb.json'
        self.char2ind_file = self.dataset_dir / 'char2id.json'
        self.word_size = len(json.load(self.word2ind_file.open()))
        self.char_size = len(json.load(self.char2ind_file.open()))
        self.pos_size = 2 * self.limit + 2
        self.total_filters_num = self.filters_num * len(self.filter_kernel_sizes) * 3

        self.use_abbv_type = False
        self.abbv_type_file = self.dataset_dir / 'abbvid_2_typeid.json'
        self.num_abbv_types = 0
        if self.abbv_type_file.exists():
            self.num_abbv_types = len(set(load_abbvid_2_typeid(self.abbv_type_file).values())) + 1

        self.print_opt = 'Model_DEF_{}'.format(self.model_name)

        # self.use_pre_sample = False


    def parse(self, kwargs):
        '''
            update config by kwargs
        '''

        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)