import argparse
import logging
import os
import json
from tqdm import tqdm
import numpy as np
import sys
import torch
import ipdb
import random
import argparse

from utils_data import DictionaryDataset, QueryDataset
from utils import Model_Wrapper, evaluate_cluster
from utils_bel import DL_metric, load_queries_dlner
from utils_dict import *

def parse_args():
    parser = argparse.ArgumentParser(description='sapbert evaluation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser.add_argument('--fpath_dev', type=str, default='../testdata/BC5CDR/test_mask.json')
    parser.add_argument('--target_root', type=str, default='../dictionary')
    parser.add_argument('--target_file', type=str, default='BC5CDR/dict4BEL_dev.txt')
    parser.add_argument('--ava_entity', type=str, default='Chemical*Disease')

    parser.add_argument('--iter', type=int, default=20)
    parser.add_argument('--thres', type=int, default=2)
    parser.add_argument('--iter_batch_size', type=int, default=4096)
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument('--select_rate', type=float, default=0)

    parser.add_argument('--shuffle_seed', type=int, default=1)
    parser.add_argument('--sample_rate', type=float, default=1.0)
    
    parser.add_argument('--dname', type=str, default='BC5CDR')
    parser.add_argument('--ner_type', type=str, default='iob')
    parser.add_argument('--init_type', type=str, default='gold')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    ava_entity = args.ava_entity.split('*')
    # ipdb.set_trace()
    dm = DL_metric()
    eval_queries, belong, dm = load_queries_dlner(args.fpath_dev, dm)

    model_wrapper = Model_Wrapper().load_model(
            path = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
            max_length = 25, use_cuda = True,
    )

    eval_queries = eval_queries
    num_query = len(eval_queries)
    query_names = [ele[0] for ele in eval_queries]
    query_dense_embeds = model_wrapper.embed_dense(names=query_names, show_progress=True, batch_size=args.iter_batch_size, agg_mode="cls")

    dict_init_set = load_dict2('init', args.shuffle_seed, args.sample_rate, args.dname)
    candidate_set_1 = load_dict2('CDT', args.shuffle_seed, 1, args.dname, 50000)
    candidate_set = load_dict2('UMLS', args.shuffle_seed, 1, args.dname, 100000)
    candidate_set = candidate_set_1 + candidate_set
    random.shuffle(candidate_set)

    # ipdb.set_trace()
    if args.init_type == 'noinit':
        if args.dname == 'NCBI':
            dict_init_set = candidate_set[:1707]
        if args.dname == 'BC5CDR':
            dict_init_set = candidate_set[:2482]
        if args.dname == 'BC5CDR-UNI':
            dict_init_set = candidate_set[:4600]
        if args.dname == 'BIONLP13':
            dict_init_set = candidate_set[:9100]
        random.shuffle(candidate_set)

    if args.ner_type == 'nest':
        candidate_set = build_neg(candidate_set, 1)

    dictInit_dense_embeds = model_wrapper.embed_dense(names=dict_init_set, show_progress=True, batch_size=args.iter_batch_size, agg_mode='cls')
    bestId_per_query, bestScore_per_query = [], []

    dsm = None
    # Get result from init dict
    bs = args.iter_batch_size
    for i in tqdm(np.arange(0,len(eval_queries),bs), total=len(eval_queries)//bs+1):
        mention_dense_embeds = query_dense_embeds[i: i+bs]
        dense_score_matrix = model_wrapper.get_score_matrix(
                query_embeds=mention_dense_embeds, 
                dict_embeds=dictInit_dense_embeds
        )
        for j in range(dense_score_matrix.shape[0]):
            idx = np.argmax(dense_score_matrix[j])
            bestId_per_query.append(idx)
            bestScore_per_query.append(dense_score_matrix[j][idx])
        
        if dsm is None: dsm = dense_score_matrix
        else: dsm = np.vstack((dsm, dense_score_matrix))

    print(dsm.shape)

    # Update dict
    iteration = args.iter
    itbs = args.iter_batch_size
    select_rate = args.select_rate
    drop_rate = args.drop_rate
    dict_set = dict_init_set
    dict_activate_list = [1 for i in range(len(dict_set))]

    print("Total entities in dict is {}".format(sum(dict_activate_list)))
    select_per_iter, acc_per_iter = [], []

    _, __, acc_ = check_acc(eval_queries, dict_init_set, bestId_per_query, ava_entity)
    acc_per_iter.append(acc_)


    for iters in range(iteration):
        # select
        select_list = [0 for i in range(itbs)]
        mention = candidate_set[iters * itbs: iters * itbs + itbs]
        dictIter_dense_embeds = model_wrapper.embed_dense(names=mention, batch_size=args.iter_batch_size, agg_mode='cls')
        dense_score_matrix = model_wrapper.get_score_matrix(
                query_embeds=query_dense_embeds, 
                dict_embeds=dictIter_dense_embeds
        )

        select_count = 0
        for i in tqdm(range(itbs)):
            correct_count, wrong_count, update_list = 0, 0, []
            for j in range(num_query):
                try:
                    if dense_score_matrix[j][i] > bestScore_per_query[j]:
                        update_list.append(j)
                        if equal_type(eval_queries[j][1], mention[i][0], ava_entity):
                            correct_count += 1
                        else: wrong_count += 1
                except: ipdb.set_trace()
            if correct_count > wrong_count:
                select_list[i] = 1
            if correct_count == wrong_count and random.random() < select_rate:
                select_list[i] = 1

            if select_list[i] == 1:
                cidx = len(dict_set)
                dict_set.append(mention[i])
                dict_activate_list.append(1)
                select_count += 1
                for ele in update_list:
                    bestId_per_query[ele] = cidx
                    bestScore_per_query[ele] = dense_score_matrix[ele][i]
                # ipdb.set_trace()
                dsm = np.hstack((dsm, dense_score_matrix[:, i].reshape(num_query, 1)))
        print("Add {} Entities.".format(select_count))
        select_per_iter.append(select_count)

        # drop        
        bad_dict_idx, wrong_query_idx, acc_ = check_acc(eval_queries, dict_set, bestId_per_query, ava_entity)
        acc_per_iter.append(acc_)
        # select bad idx
        bad_dict_idx = list(set( bad_dict_idx ))
        bad_idxs, bad_dis = [], []
        for bidx in bad_dict_idx:
            wcount, rcount = 0, 0
            for i in range(num_query):
                if bestId_per_query[i] != bidx: continue
                if equal_type(eval_queries[i][1], dict_set[bidx][0], ava_entity):
                    rcount += 1
                else: wcount += 1
            if wcount > rcount:
                bad_idxs.append(bidx)
                bad_dis.append(wcount - rcount)

        # drop_idxs = random.sample(bad_dict_idx, int(len(bad_dict_idx)*drop_rate))
        # bad_rank_idx = np.array(bad_dis).argsort()
        # drop_num = max(1, int(len(bad_idxs)*drop_rate))
        # if len(bad_idxs) == 0: drop_num = 0
        # drop_idxs = [bad_dict_idx[i_] for i_ in bad_rank_idx[:drop_num]]

        drop_idxs = []
        for idx_, dis_ in zip(bad_idxs, bad_dis):
            if dis_ > args.thres: drop_idxs.append(idx_)
        print("Drop {} Entities.".format(len(drop_idxs)))

        # Update after dropping
        for idx in drop_idxs:
            dict_activate_list[idx] = 0
            for i in range(num_query):
                if bestId_per_query[i] == idx:
                    bestScore_per_query[i] = -100000
                    for j in range(len(dict_set)):
                        if dict_activate_list[j] > 0 and dsm[i][j] > bestScore_per_query[i]:
                            bestId_per_query[i] = j
                            bestScore_per_query[i] = dsm[i][j]

    print("Total entities in dict is {}".format(sum(dict_activate_list)))

    new_path = os.path.join(args.target_root, args.target_file)
    assert len(dict_set) == len(dict_activate_list)
    nlines = []
    for ele, ava in zip(dict_set, dict_activate_list):
        if ava > 0: nlines.append(ele[0] + '\t' + ele[1] + '\n')
    with open(new_path, 'w') as of:
        of.writelines(nlines)
    
    print(select_per_iter)
    print(acc_per_iter)
