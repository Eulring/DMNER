import os
import json
import ipdb
import argparse

from utils_dict import *

parser = argparse.ArgumentParser()
parser.add_argument("--dname", default = 'linnaeus')
parser.add_argument("--droot", default = '')
parser.add_argument("--etype", default = 'Species')
args = parser.parse_args()

train_file = args.droot + '/DATA/Datasets/####/train.json'
target_file = args.droot + '/S2-BEM/dictionary/####/dict-init.txt'

train_file = train_file.replace('####', args.dname)
target_file = target_file.replace('####', args.dname)

data_glod = json.load(open(train_file))
etype = args.etype
nlines = []

lines_set = set()

for key in data_glod.keys():
    words = data_glod[key]['words']
    spans = data_glod[key]['spans']
    try:
        types = data_glod[key]['spans_type']
    except:
        types = [etype for i in range(len(spans))]

    if len(types) != len(spans):
        types = [etype for i in range(len(spans))]
    for span, tp in zip(spans, types):
        if tp == 'GP': tp = 'Gene'
        mention = ' '.join(words[span[0]:span[1] + 1])
        lines_set.add(tp + '\t' + mention + '\n')

nlines = list(lines_set)
# for mention in mention_list:
#     nlines.append(etype + '\t' + mention + '\n')

with open(target_file, 'w') as of:
    of.writelines(nlines)
