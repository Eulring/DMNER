import os
import random
import ipdb


def equal_type(e1, e2, ava_entity):
    if e1 not in ava_entity: e1 = 'Other'
    if e2 not in ava_entity: e2 = 'Other'
    return e1 == e2

def check_acc(queries, dict_set, id_per_query, ava_entity):
    match_count = 0
    dismatch_dictidx = []
    dismatch_queryidx = []
    count = 0
    # ipdb.set_trace()
    for query, idx in zip(queries, id_per_query):
        gold_type = query[1]
        pred_type = dict_set[idx][0]
        if pred_type not in ava_entity:
            pred_type = 'Other'
        if gold_type not in ava_entity:
            gold_type = 'Other'

        if gold_type == pred_type:
            match_count += 1
        else:
            dismatch_dictidx.append(idx)
            dismatch_queryidx.append(count)
        count += 1
    # ipdb.set_trace()
    acc = match_count / len(id_per_query)
    print("Accuracy: {}".format(acc))
    return dismatch_dictidx, dismatch_queryidx, acc


def f1_score(queries, dict_set, id_per_query, ava_entity, mode = '1'):
    match_count, pred_count, gold_count = 0, 0, 0
    dismatch_dictidx = []
    dismatch_queryidx = []
    count = 0
    # ipdb.set_trace()
    for query, idx in zip(queries, id_per_query):
        gold_type = query[1]
        pred_type = dict_set[idx][0]
        if pred_type not in ava_entity:
            pred_type = 'Other'
        else: pred_count += 1

        if gold_type not in ava_entity:
            gold_type = 'Other'
        else: gold_count += 1

        if gold_type == pred_type:
            if gold_type in ava_entity:
                match_count += 1
        else:
            dismatch_dictidx.append(idx)
            dismatch_queryidx.append(count)
        count += 1


    if mode == '1':
        pre = match_count / pred_count
        rec = match_count / gold_count
        f1 = 2 * pre * rec / (pre + rec + 0.000000001)
        print("F1: {}".format(f1))
        return pred_count, gold_count, match_count
    # else:
    return pred_count, gold_count, match_count, dismatch_dictidx, dismatch_queryidx

def f1_moment(tp, tg, tm, mp, mg, mm):
    old_pre = tm / tp
    old_rec = tm / tg
    old_f1 = 2 * old_pre * old_rec / (old_pre + old_rec + 0.0000001)
    new_pre = (tm + mm) / (tp + mp)
    new_rec = (tm + mm) / (tg + mg)
    new_f1 = 2 * new_pre * new_rec / (new_pre + new_rec + 0.0000001)
    return new_f1 - old_f1


def load_lines(fpath, mode = 1):
    if mode == 1:
        with open(fpath, 'r') as of:
            lines = of.readlines()
            lines = [line.replace('\n', '') for line in lines]
    if mode == 2:
        with open(fpath, 'r') as of:
            lines = of.readlines()
            lines = [line.replace('\n', '').split('\t') for line in lines]
    return lines




def load_dict2(load_type = 'CDT', shuffle_seed = 1, sample_rate = 1.0, dname = 'BC5CDR', sample_num = 10000, droot = '###'):
    if load_type == 'CDT':
        dict_type = []
        dict_type.append('Anatomy')
        dict_type.append('Chemical')
        dict_type.append('Disease')
        dict_type.append('Gene')
        dict_type.append('Pathway')

        pairs = []
        dict_root = droot + '/DATA/DSDict/'
        for etype in dict_type:
            fpath_dict = os.path.join(dict_root, 'C_' + etype + '.txt')
            mentions = load_lines(fpath_dict)
            for men in mentions:
                pairs.append((etype, men))
        for i in range(shuffle_seed):
            random.shuffle(pairs)
        # ipdb.set_trace()
        return pairs[:sample_num]

    if load_type == 'UMLS':
        pairs = []
        dict_umls_root = droot + '/DATA/DSDict/'
        fpath_dict = os.path.join(dict_umls_root, 'dict_neg_UMLS_Semantic_Network_2023.txt')
        fp = open(fpath_dict)
        for men in fp:
            etype=men.split("||")[0]
            mentext=men.split("||")[1]
            pairs.append((etype, mentext))
        for i in range(shuffle_seed):
            random.shuffle(pairs)
        new_pair = []
        for i in range(sample_num):
            ele = pairs[i]
            new_pair.append((ele[0], ele[1].replace('\n', '')))
        return new_pair
    
    if load_type == 'init':
        fpath = '../dictionary/####/dict-init.txt'.replace('####', dname)
        if dname == 'BC5CDR': fpath = '../dictionary/BC5CDR/dict4BEL_pos.txt'
        if dname == 'BC5CDR-UNI': 
            idx = str(shuffle_seed % 3 + 1)
            fpath = '../dictionary/BC5CDR-UNI/dict-kb##.txt'.replace('##', idx)
        pairs = load_lines(fpath, 2)
        print(fpath)
        for i in range(shuffle_seed):
            random.shuffle(pairs)
        pairs = pairs[:int(len(pairs) * sample_rate)]
        return pairs




def entity_split(ss):
    sep_char = [' ', '-', '_']
    for sc in sep_char:
        if len(ss.split(sc)) > 0: return ss.split(sc)
    return [ss]

def segment_entity(ss):
    mid = random.randint(1, len(ss) - 1)
    if random.random() < 0.5: 
        return ' '.join(ss[:mid])
    else: 
        return ' '.join(ss[mid:])

def build_neg(mentions, build_rate):
    num = len(mentions)
    for i in range(num):
        ss = entity_split(mentions[i][1])
        if len(ss) > 1 and random.random() < build_rate:
            mentions.append( ('Other', segment_entity(ss)) )
    random.shuffle(mentions)
    return mentions
