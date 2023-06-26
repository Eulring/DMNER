#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import ipdb
# import parser
import argparse

def equal(s1, s2): return s1.lower() == s2.lower()

def update_types(types):
    ntypes = []
    for ele in types:
        if ele == 'GP': ele = 'Gene'
        ntypes.append(ele)
    return ntypes

def save_json(path, d):
    with open(path, 'w') as f:
        #json.dump(d, f, ensure_ascii=False, indent=2)
        json.dump(d, f)

def word_level_align(words, mention):
    mwords = mention.split(' ')
    n = len(words)
    m = len(mwords)
    spans = []
    for i in range(n - m + 1):
        for j in range(m):
            if not equal(words[i + j], mwords[j]):
                break
            if j == m - 1:
                spans.append([i, i + m - 1])
    return spans

def char_level_align(words, mention):
    # ipdb.set_trace()
    special_token = ['/', '-', ' ']
    raw_text = ''.join(words)
    raw_mention = mention.replace(' ', '')
    n = len(raw_text)
    m = len(raw_mention)
    start = [-1 for i in range(n)]
    end = [-1 for i in range(n)]
    count = 0
    spans = []
    for i, word in enumerate(words):
        start[count] = i
        end[count + len(word) - 1] = i
        wlen = len(word)
        left, right = 0, 0
        while (word[left] in special_token) and left < wlen - 1:
            start[count + left + 1] = i
            left += 1
        while (word[wlen - 1 - right] in special_token) and right < wlen - 1:
            end[count + len(word) - 1 - right - 1] = i
            right += 1

        count += len(word)

    for i in range(n - m + 1):
        for j in range(m):
            if raw_text[i+j] != raw_mention[j]:
                break
            if j == m - 1:
                if start[i] > -1 and end[i + m - 1] > -1:
                    l = start[i]
                    r = end[i + m - 1]
                    spans.append([l, r])
    return spans


def convert_file(input_gold, input_predict, output_path, dname, mode = '2'):
    
    # words = ['histopathological', 'analysis', 'revealed', 'progressive', 'cardiomyocyte', 'degeneration', ',', 'hypertrophy', '/cytomegaly', ',', 'and', 'extensive', 'vacuolation', 'after', 'two', 'doses', '.']
    # men = 'cytomegaly'
    # char_level_align(words, men)
    
    data_raw = json.load(open(input_gold))

    for k_ in data_raw.keys():
        data_raw[k_]['golds'] = data_raw[k_]['spans']
        data_raw[k_]['gold_types'] = update_types(data_raw[k_]['spans_type'])
        data_raw[k_]['spans_type'] = update_types(data_raw[k_]['spans_type'])

    data_predict = json.load(open(input_predict))
    count = 0

    for ele in data_predict:
        entity = ele["output_entity"]
        raw = data_raw[str(count)]["words"]
        res = list(map(lambda x:x.lower(), raw))
        pos = []

        if mode == '2':
            for ent in entity:
                spans = word_level_align(res, ent[2])
                if len(spans) == 0:
                    spans = char_level_align(res, ent[2])
                
                if len(spans) == 0:
                    print(count)
                    # ipdb.set_trace()
                else: 
                    for span in spans:
                        if span not in pos:
                            pos.append(span)
                            break


        if mode == '1':

            for idx, item in enumerate(res):
                if '-' in item:
                    if item.index('-') == 0 or item.index('-') == len(item)-1:
                        res[idx] = item.replace('-','')

            for j in entity:
                try:
                    try:
                        start = res.index(j[2].split(' ')[0])
                    except:
                        start = res.index(j[2].replace(' - ','-').split(' ')[0])
                    try:
                        end = res.index(j[2].split(' ')[-1])
                    except:
                        end = res.index(j[2].replace(' - ','-').split(' ')[-1])
                    pos.append([start, end])
                except:
                    try:
                        ele_name = j[2]
                        ele_name = ele_name.replace(' / ', '/')
                        ele_name = ele_name.replace(' , ', ',')
                        ele_name = ele_name.replace(' - ', '-')

                        start = res.index(ele_name.split(' ')[0])
                        end = res.index(ele_name.split(' ')[-1])

                        pos.append([start, end])
                    except:
                        continue

        data_raw[str(count)]['preds'] = pos
        count += 1

    save_json(output_path, data_raw)
    print(f"Convert {count} samples, save to {output_path}")



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-input_gold", default = '/home/test2/GPUNER-v2/S2-BEL/testdata/BC5CDR/test_biobert-iob.json')
    parser.add_argument("-input_predict", default = '/home/test2/GPUNER-v2/S1-EBD/mrc/dataset/tmp/predict.json')
    parser.add_argument("-output_path", default = '/home/test2/GPUNER-v2/S2-BEL/testdata/BC5CDR/test_mrc.json')
    parser.add_argument("-dname", default = 'bc5cdr')
    args = parser.parse_args()

    print(args)
    convert_file(args.input_gold, args.input_predict, args.output_path, args.dname)

if __name__ == '__main__':
    main()