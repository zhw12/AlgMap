"""
    Construct the algorithm roadmap given a query
"""

import json
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from models.util import load_vocab

if __name__ == '__main__':

    dataset = 'NIPS'
    model_name = 'cantor'
    feature_type = 'paragraph'
    root_dir = Path('/scratch/home/hanwen/algmap/')
    dataset_dir = root_dir / 'data' / dataset
    result_dir = root_dir / 'out' / dataset / model_name
    word2ind_file = dataset_dir / 'word2id.json'

    word2ind, vocab = load_vocab(word2ind_file)
    exact_pattern = r'^(?:[a-z]*[A-Z][a-z\d+-]*){2,}$'

    co_occur = defaultdict(float)
    for eval_type in ['train', 'valid', 'test']:
        with open(result_dir / '{}_all_{}_predictions.txt'.format(eval_type, model_name)) as fin:
            for line in tqdm(fin):
                w1, w2, y, pred_y, pred_p = line.strip().split()
                y, pred_y, pred_p = int(y), int(pred_y), float(pred_p)
                co_occur[(w1, w2)] = pred_p
                co_occur[(w2, w1)] = pred_p

    time_cnter_dict = json.load(open(dataset_dir / 'time_cnter_dict.json'))
    time_dict = Counter({w: int(min(time_cnter_dict[w])) for w in time_cnter_dict})

    with open(dataset_dir / 'word_cnter.json') as fin:
        word_hist = json.load(fin)
        word_hist = {w: word_hist[w] for w in word_hist if re.match(exact_pattern, w)}
        word_hist = Counter(word_hist)


    def roadmap_graph_by_time(word_hist, co_occur, cutoff=0, min_time=1988):
        """Convert a word histogram with co-occurrences to a weighted graph.
        Edges are only added if the count is above cutoff.
        """
        g = nx.DiGraph()
        for (w1, w2), count in co_occur.items():
            if count <= cutoff:
                continue
            y1, y2 = time_dict[w1], time_dict[w2]
            if y1 < min_time or y2 < min_time:
                continue
            if y1 < y2 or (y1 == y2 and word_hist[w1] >= word_hist[w2]):
                g.add_edge(w1, w2, weight=count)
            else:
                g.add_edge(w2, w1, weight=count)
        return g


    G = roadmap_graph_by_time(word_hist, co_occur, cutoff=0.5)

    raw_query = 'GAN'

    subgraph_nodes = nx.single_source_shortest_path_length(G, raw_query, cutoff=2)
    subgraph_nodes = list(subgraph_nodes)
    subG = G.subgraph(subgraph_nodes)
