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


parser = argparse.ArgumentParser(description='assemble evaluation')
parser.add_argument('--dname', type=str, default='JNLPBA')
args = parser.parse_args()

dname = args.dname
output_root = '../output/###'.replace('###', dname)

if dname == 'BC5CDR': data_dir = '../testdata/###/test_mask.json'.replace('###', dname)
elif dname == 'BC5CDR-UNI': data_dir = '../testdata/###/test_uni.json'.replace('###', dname)
else: data_dir = '../testdata/###/test_iob.json'.replace('###', dname)
output_files = []

if dname == 'BC4CHEMD':
    output_files = []
    output_files.append('dict-dev-iob-1.json')
    output_files.append('dict-dev-iob-2.json')
    output_files.append('dict-dev-iob-3.json')
    ava_entity = ['Chemical']
if dname == 'NCBI':
    output_files = []
    output_files.append('dict-dev-iob-1.json')
    output_files.append('dict-dev-iob-2.json')
    output_files.append('dict-dev-iob-3.json')
    ava_entity = ['Disease']
if dname == 'BC5CDR-disease':
    output_files = []
    output_files.append('dict-dev-iob-1.json')
    output_files.append('dict-dev-iob-2.json')
    output_files.append('dict-dev-iob-3.json')
    ava_entity = ['Disease']
if dname == 'BC2GM':
    output_files = []
    output_files.append('dict-dev-iob-1.json')
    output_files.append('dict-dev-iob-2.json')
    output_files.append('dict-dev-iob-3.json')
    ava_entity = ['Gene']

if dname == 'JNLPBA':
    output_files = []
    output_files.append('dict-dev-iob-1.json')
    output_files.append('dict-dev-iob-2.json')
    output_files.append('dict-dev-iob-3.json')
    ava_entity = ['Gene']

if dname == 'BC5CDR-UNI':
    output_files = []
    output_files.append('dict-dev-uni-1.json')
    output_files.append('dict-dev-uni-2.json')
    output_files.append('dict-dev-uni-3.json')
    # output_files.append('dict-kb1.json')
    # output_files.append('dict-kb2.json')
    # output_files.append('dict-kb3.json')
    ava_entity = ['Chemical', 'Disease']

if dname == 'BC5CDR':
    output_files = []
    output_files.append('dict-dev-mask-1.json')
    # output_files.append('dict-aba-sr-noinit.json')
    output_files.append('dict-dev-mask-3.json')
    output_files.append('dict-dev-1.json')
    output_files.append('dict-dev-5.json')
    # output_files.append('dict-kb3.json')
    # output_files.append('dict-kb.json')
    output_files.append('dict4BEL_pos.json')
    ava_entity = ['Chemical', 'Disease']

if dname == 'BC5CDR-chem':
    output_files = []
    output_files.append('dict-dev-iob-1.json')
    output_files.append('dict-dev-iob-2.json')
    output_files.append('dict-dev-iob-3.json')
    ava_entity = ['Chemical']

if dname == 'linnaeus':
    output_files = []
    output_files.append('dict-dev-iob-1.json')
    output_files.append('dict-dev-iob-2.json')
    output_files.append('dict-dev-iob-3.json')
    ava_entity = ['Species']

if dname == 's800':
    output_files = []
    output_files.append('dict-dev-iob-1.json')
    output_files.append('dict-dev-iob-2.json')
    output_files.append('dict-dev-iob-3.json')
    ava_entity = ['Species']





def vote(preds, edicts=['Chemical', 'Disease']):
    d2i = {}
    edicts.append('Other')
    npreds = []
    for i in range(len(preds)):
        if preds[i] not in edicts:
            npreds.append('Other')
        else: npreds.append(preds[i])
    for i in range(len(edicts)):
        d2i[edicts[i]] = i
    cnt = [0 for i in range(len(edicts))]
    for v_ in npreds: cnt[d2i[v_]] += 1
    # if cnt[0] > 1 and cnt[1] > 1: return 'Other'
    return edicts[np.argmax(cnt)]


def chaos_rate(votes, dicts):
    # dicts = dicts + ['Other']
    d2i = {}
    for i in range(len(dicts)):
        d2i[dicts[i]] = i
    cnt = [0 for i in range(len(dicts))]
    for v_ in votes:
        if v_ not in dicts:
            v_ = 'Other'
        cnt[d2i[v_]] += 1
    return np.std(cnt)


def overlap(men1, men2):
    ws1 = men1.split(' ')
    ws2 = men1.split(' ')
    for w1 in ws1:
        if w1 in ws2: 
            return True
    return False



def compute_assemble(output_files_, data_dir_):

    dm = DL_metric()
    eval_queries, belong, dm = load_queries_dlner(data_dir_, dm)

    outputs = {}
    for ofname in output_files_:
        fpath = os.path.join(output_root, ofname)
        output = json.load(open(fpath))
        outputs[ofname] = output

    num = len(output_files_) 
    voted_res = {}

    for key in outputs[output_files_[0]].keys():
        preds = []
        for ofname in output_files_:
            preds.append(outputs[ofname][key])

        num_pred = len(preds[0])
        res, cres, name, gold = [], [], [], []
        for i in range(num_pred):
            output = []
            for j in range(num):
                pred_type_here = preds[j][i]['preds'][0]
                output.append(pred_type_here)

            res.append(vote(output, ava_entity))
            cres.append(-chaos_rate(output, ava_entity))
            name.append(preds[j][i]['name'])
            gold.append(preds[j][i]['gold'])
        
        # ipdb.set_trace()
        # voted_res[]
        select_list = [0 for i in range(num_pred)]
        indexs = np.argsort(cres)

        for idx in indexs:
            select = 1

            select_list[idx] = select
            if select == 1:
                gold_type = gold[idx]
                pred_type = res[idx]
                if gold_type == 'Other':
                    if pred_type != 'Other':
                        dm.update(2, 'pred')
                else:
                    dm.update(2, 'gold')
                    if pred_type == gold_type:
                        dm.update(2, 'pred')
                        dm.update(2, 'match')

    return dm.compute()

compute_assemble(output_files, data_dir)

# rp, rr, rf = [], [], []
# for ele in output_files:
#     p_, r_, f_ = compute_assemble([ele], data_dir)
#     rp.append(p_)
#     rr.append(r_)
#     rf.append(f_)

# print(np.average(rp))
# print(np.average(rr))
# print(np.average(rf))
