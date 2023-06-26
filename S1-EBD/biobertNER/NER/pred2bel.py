import os
import json
import ipdb
import argparse


def end_of_chunk(ptag, tag, ntag):
    chunk_end = False
    if tag == 'B' and ntag == 'O': chunk_end = True
    if tag == 'I' and ntag == 'O': chunk_end = True
    if tag == 'I' and ntag == 'B': chunk_end = True
    if tag == 'B' and ntag == 'B': chunk_end = True
    return chunk_end

def start_of_chunk(ptag, tag, ntag):
    chunk_start = False
    if tag == 'B': chunk_start = True
    if tag == 'I' and ptag == 'O': chunk_start = True
    return chunk_start


def load_data(path_pred, path_gold, args):
    with open(path_pred, 'r') as of:
        lines1 = of.readlines()
    with open(path_gold, 'r') as of:
        lines2 = of.readlines()

    assert len(lines1) == len(lines2)
    res = {}

    words, ptags, gtags = [], [], []
    countp = 0
    countg = 0
    num_gold = 0
    num_pred = 0
    for l1, l2 in zip(lines1, lines2):
        if l1 == '\n':
            # count = strlen(res)
            ptags = ['O'] + ptags + ['O']
            gtags = ['O'] + gtags + ['O']
            n = len(words)
            l_p = [i for i in range(n) if start_of_chunk(ptags[i], ptags[i+1], ptags[i+2])]
            l_g = [i for i in range(n) if start_of_chunk(gtags[i], gtags[i+1], gtags[i+2])]
            # l_p = [i for i in range(n) if ptags[i] == 'B']
            # l_g = [i for i in range(n) if gtags[i] == 'B']
            r_p = [i for i in range(n) if end_of_chunk(ptags[i], ptags[i+1], ptags[i+2])]
            # r_p = [i for i in range(n) if ptags[i + 1] != 'I' and ptags[i]!= 'O']
            # r_g = [i for i in range(n) if gtags[i + 1] != 'I' and gtags[i]!= 'O']
            r_g = [i for i in range(n) if end_of_chunk(gtags[i], gtags[i+1], gtags[i+2])]

            # ipdb.set_trace()
            if len(l_p) != len(r_p):
                ipdb.set_trace()
            preds = [(e1,e2) for e1,e2 in zip(l_p, r_p)]
            golds = [(e1,e2) for e1,e2 in zip(l_g, r_g)]
            
            unit = {
                'words': words,
                'preds': preds,
                'golds': golds,
                'gold_types': [args.etype for i in range(len(golds))]
            }
            num_gold += len(golds)
            num_pred += len(preds)
            res[str(len(res))] = unit
            words, ptags, gtags = [], [], []
        else:
            e1 = l1.replace('\n', '').split(' ')
            e2 = l2.replace('\n', '').split(' ')
            assert e1[0] == e2[0]
            words.append(e1[0])
            ptags.append(e1[1])
            gtags.append(e2[1])

            if e2[1] == 'B': countg += 1
            if e1[1] == 'B': countp += 1
        
    print(countp)
    print(num_pred)

    print(countg)
    print(num_gold)
    
    return res




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dname", default = 'linnaeus')
    parser.add_argument("--droot", default = '')
    parser.add_argument("--etype", default = 'Species')
    args = parser.parse_args()

    pred_file_root = os.path.join('./output', args.dname)
    output_root = os.path.join(args.droot + '/S2-BEM/testdata', args.dname)
    gold_file_root = os.path.join('../datasets', args.dname)

    if not os.path.exists(output_root): os.mkdir(output_root)

    pred_test = os.path.join(pred_file_root, 'test_predictions.txt')
    pred_dev = os.path.join(pred_file_root, 'dev_predictions.txt')

    gold_test = os.path.join(gold_file_root, 'test.txt')
    gold_dev = os.path.join(gold_file_root, 'devel.txt')

    output_test = os.path.join(output_root, 'test_iob.json')
    output_dev = os.path.join(output_root, 'dev_iob.json')

    res_test = load_data(pred_test, gold_test, args)
    res_dev = load_data(pred_dev, gold_dev, args)

    with open(output_test, 'w') as of:
        json.dump(res_test, of)
    with open(output_dev, 'w') as of:
        json.dump(res_dev, of)


if __name__ == '__main__':
    main()