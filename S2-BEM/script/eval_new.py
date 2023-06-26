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

from utils_data import DictionaryDataset, QueryDataset
from utils import Model_Wrapper, evaluate
from utils_bel import DL_metric, load_queries_dlner

LOGGER = logging.getLogger()

sys.path.append("../") 

match_num, pred_num, gold_num = 0, 0, 0

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert evaluation')

    # Required
    parser.add_argument('--model_dir', required=True, help='Directory for model')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')
    parser.add_argument('--entity_types', type=str, required=True)

    # Run settings
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--output_dir', type=str, default='../output/', help='Directory for output')
    parser.add_argument('--save_predictions', action="store_true", help="whether to save predictions")
    parser.add_argument('--debug_mode', default=False, type=bool)

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    args = parser.parse_args()
    return args


def load_dictionary(dictionary_path): 
    dictionary = DictionaryDataset(dictionary_path = dictionary_path)
    return dictionary.data

def load_dictionary_bc5cdr(dpath):
    data = []
    with open(dpath, 'r') as of:
        lines = of.readlines()
        for line in lines:
            data.append( (line.split('\t')[1].replace('\n', ''), line.split('\t')[0])  )
    return data


def par(belong, idx):
    while belong[idx] != idx:
        idx = belong[idx]
    return idx

def merge(idx1, idx2, belong):
    par1 = par(belong, idx1)
    par2 = par(belong, idx2)
    if par1 != par2:
        belong[par1] = par2
    return belong


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


if __name__ == '__main__':
    args = parse_args()
    init_logging()
    print(args)

    dm = DL_metric()

    # load dictionary and data
    eval_dictionary = load_dictionary_bc5cdr(args.dictionary_path)
    random.shuffle(eval_dictionary)
    print ("[reference dictionary loaded]")
    
    eval_queries, belong, dm = load_queries_dlner(args.data_dir, dm, args.debug_mode)
    entity_types = args.entity_types.split('*')

    model_wrapper = Model_Wrapper().load_model(
            path=args.model_dir,
            max_length=args.max_length,
            use_cuda=args.use_cuda,
    )

    result_evalset = evaluate(
            model_wrapper=model_wrapper,
            eval_dictionary=eval_dictionary,
            eval_queries=eval_queries,
            topk=1, # sort only the topk to save time
    )

    res_collect = {}

    for i in range(len(result_evalset)):
        pid = str(par(belong, i))
        if pid not in res_collect:
            res_collect[pid] = [result_evalset[i]]
        else: 
            res_collect[pid].append(result_evalset[i])


    for key in res_collect.keys():
        if len(res_collect[key]) == 1:
            gold_type = res_collect[key][0]['gold']
            preds = res_collect[key][0]['preds']
            ename = res_collect[key][0]['name']
        elif len(res_collect[key]) > 1:
            idx = 0
            gold_type = res_collect[key][idx]['gold']
            preds = res_collect[key][idx]['preds']
            ename = res_collect[key][idx]['name']
        else: 
            continue

        pred_type = preds[0]
        if pred_type not in entity_types:
            pred_type = 'Other'

        if args.debug_mode:
            print(res_collect[key][0]['pred_names'])
            print(ename)
            print(preds)
            print(gold_type)
            print()
            # if preds[0] == 'Other' and gold_type == 'Other':
            #     ipdb.set_trace()
            # if ename == 'CIN':
            #     ipdb.set_trace()
            # if ename == 'histamine':
            #     ipdb.set_trace()
            ipdb.set_trace()

        if gold_type == 'Other':
            if pred_type != 'Other':
                dm.update(2, 'pred')
        else:
            dm.update(2, 'gold')
            if pred_type == gold_type:
                dm.update(2, 'pred')
                dm.update(2, 'match')
    
    # ipdb.set_trace
    dm.compute()

    # ipdb.set_trace()
    data_name = args.data_dir.split('/')[-2]
    dict_name = args.dictionary_path.split('/')[-1].replace('.txt', '')
    tar_file_output = os.path.join(args.output_dir, data_name, dict_name+'.json')
    
    with open(tar_file_output, 'w') as of:
        json.dump(res_collect, of)
