import ipdb
import os
import json


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

def span_conflict(sp1, sp2):
    if sp2[0] <= sp1[1] and sp2[0] >= sp1[0]: return True
    if sp2[1] <= sp1[1] and sp2[1] >= sp1[0]: return True
    return False

def load_queries_dlner(data_path, dm, debug_mode = False):
    with open(data_path, 'r') as of:
        data = json.load(of)

    query_pairs, spans = [], []
    belong = []

    count = 0
    for key in data.keys():
        ele = data[key]

        if 'iob' not in data_path:
            sid = []
            for i in range(len(ele['preds'])):
                u = ele['preds'][i]
                if u[1] - u[0] < 7: sid.append(i)
            ele['preds'] = [ ele['preds'][id_] for id_ in sid]

        nump = len(ele['preds'])
        dm.update(1, 'pred', len(ele['preds']))
        dm.update(1, 'gold', len(ele['golds']))

        for i in range(nump):
            span = ele['preds'][i]
            ename = ' '.join(ele['words'][span[0]: span[1] + 1])
            etype = 'Other'
            for j in range(len(ele['golds'])):
                if span == ele['golds'][j]:
                    dm.update(1, 'match')
                    etype = ele['gold_types'][j]
                    break
            query_pairs.append((ename, etype))
        
        # init combine set
        ele['idxs'] = [i + count for i in range(len(ele['preds']))]
        belong.extend(ele['idxs'])

        for i in range(nump):
            for j in range(nump):
                if i != j:
                    if span_conflict(ele['preds'][i], ele['preds'][j]):
                        belong = merge(ele['idxs'][i], ele['idxs'][j], belong)
        count += nump

    return query_pairs, belong, dm
            



class DL_metric(object):
    def __init__(self):
        self.pred_count_bel = 0
        self.gold_count_bel = 0
        self.match_count_bel = 0
        self.pred_count_ebd = 0
        self.gold_count_ebd = 0
        self.match_count_ebd = 0

    def update(self, step, idx, num=1):
        if step == 1:
            if idx == 'pred':
                self.pred_count_ebd += num
            if idx == 'gold':
                self.gold_count_ebd += num
            if idx == 'match':
                self.match_count_ebd += num
        if step == 2:
            if idx == 'pred':
                self.pred_count_bel += num
            if idx == 'gold':
                self.gold_count_bel += num
            if idx == 'match':
                self.match_count_bel += num

    def vis(self):
        print(self.pred_count_bel)
        print(self.gold_count_bel)
        print(self.match_count_bel)

    def compute(self):
        ebd_precision = self.match_count_ebd / self.pred_count_ebd
        ebd_recall = self.match_count_ebd / self.gold_count_ebd
        ebd_f1 = 2 * ebd_precision * ebd_recall / (ebd_precision + ebd_recall)

        bel_precision = self.match_count_bel / self.pred_count_bel
        bel_recall = self.match_count_bel / self.gold_count_bel
        bel_f1 = 2 * bel_precision * bel_recall / (bel_precision + bel_recall)

        precision_all = self.match_count_bel / self.pred_count_bel
        recall_all = self.match_count_bel / self.gold_count_ebd
        f1_all = 2 * precision_all * recall_all / (precision_all + recall_all)

        print("EBD pre = {}".format(ebd_precision))
        print("EBD rec = {}".format(ebd_recall))
        print("EBD f1 = {}".format(ebd_f1))
        print()
        print("BEL pre = {}".format(bel_precision))
        print("BEL rec = {}".format(bel_recall))
        print("BEL f1 = {}".format(bel_f1))
        print()
        print("Pipeline pre = {}".format(precision_all))
        print("Pipeline rec = {}".format(recall_all))
        print("Pipeline f1 = {}".format(f1_all))
        print()

        return precision_all, recall_all, f1_all

