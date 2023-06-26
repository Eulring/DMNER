#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_dataset.py

import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
import ipdb


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, json_path, tokenizer: BertWordPieceTokenizer, max_length: int = 512, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False, data_use='train'):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen
        self.data_use = data_use

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id
        """
        data = self.all_data[item]
        tokenizer = self.tokenizer

        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]
        
        # if self.data_use == 'train': ipdb.set_trace()
        if self.data_use == 'train':
            start_positions, end_positions, start_positions_ct, end_positions_ct = self.combine_labels(data, mode='5')
        

        # add space offsets
        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        if self.data_use == 'train':
            start_positions_ct = [x + sum([len(w) for w in words[:x]]) for x in start_positions_ct]
            end_positions_ct = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions_ct]

        # if item 
        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids
        offsets = query_context_tokens.offsets

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx


        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

        if self.data_use == 'train':
            new_start_positions_ct = [origin_offset2token_idx_start[start] for start in start_positions_ct]
            new_end_positions_ct = [origin_offset2token_idx_end[end] for end in end_positions_ct]

        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # the start/end position must be whole word
        if not self.is_chinese:
            for token_idx in range(len(tokens)):
                current_word_idx = query_context_tokens.words[token_idx]
                next_word_idx = query_context_tokens.words[token_idx+1] if token_idx+1 < len(tokens) else None
                prev_word_idx = query_context_tokens.words[token_idx-1] if token_idx-1 > 0 else None
                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # if self.data_use == 'train':
        #     start_labels_ct = [(1 if idx in new_start_positions_ct else 0) for idx in range(len(tokens))]
        #     end_labels_ct = [(1 if idx in new_end_positions_ct else 0) for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]
        # if self.data_use == 'train':
        #     start_labels_ct = start_labels_ct[: self.max_length]
        #     end_labels_ct = end_labels_ct[: self.max_length]


        # make sure last token is [SEP]
        sep_token = tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0


        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)
            # if self.data_use == 'train':
            #     start_labels_ct = self.pad(start_labels_ct)
            #     end_labels_ct = self.pad(end_labels_ct)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1
        
        
        match_mask_ct = torch.ones([seq_len, seq_len]).float()
        if self.data_use == 'train':
            for start, end in zip(new_start_positions_ct, new_end_positions_ct):
                if start >= seq_len or end >= seq_len:
                    continue
                match_mask_ct[start, end] = 0
                # ipdb.set_trace()
                start_label_mask[start] = 0
                end_label_mask[end] = 0

                # ipdb.set_trace()
            

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            match_mask_ct,
            sample_idx,
            label_idx
        ]

    def combine_labels(self, data, mode='1'):
        sp0 = data['span_position']
        sp1 = data['span_position_gpt']
        sp2 = data['span_position_ap']
        sp3 = data['span_position_umls']

        if mode == '1':
            # keep all sp0, and no mask
            spos = [int(e_.split(';')[0]) for e_ in sp0]
            epos = [int(e_.split(';')[1]) for e_ in sp0]
            return spos, epos, [], []

        if mode == '2':
            # keep all sp0 + sp1, and no mask
            sps = sp0 + sp1
            spos = [int(e_.split(';')[0]) for e_ in sps]
            epos = [int(e_.split(';')[1]) for e_ in sps]
            return spos, epos, [], []

        if mode == '3':
            # keep sp0, mask others
            sp_pos = sp0
            sp_mask = sp1 + sp2 + sp3
            sp_mask = [ele for ele in sp_mask if ele not in sp_pos]

            spos = [int(e_.split(';')[0]) for e_ in sp_pos]
            epos = [int(e_.split(';')[1]) for e_ in sp_pos]
            smask = [int(e_.split(';')[0]) for e_ in sp_mask]
            emask = [int(e_.split(';')[1]) for e_ in sp_mask]

            return spos, epos, smask, emask

        if mode == '4':
            # keep sp0, mask others (without gpt label)
            sp_pos = sp0
            sp_mask = sp2 + sp3
            sp_mask = [ele for ele in sp_mask if ele not in sp_pos]

            spos = [int(e_.split(';')[0]) for e_ in sp_pos]
            epos = [int(e_.split(';')[1]) for e_ in sp_pos]
            smask = [int(e_.split(';')[0]) for e_ in sp_mask]
            emask = [int(e_.split(';')[1]) for e_ in sp_mask]

            return spos, epos, smask, emask

        if mode == '5':
            # keep sp0, mask only autophrae
            sp_pos = sp0
            sp_mask = sp2
            sp_mask = [ele for ele in sp_mask if ele not in sp_pos]

            spos = [int(e_.split(';')[0]) for e_ in sp_pos]
            epos = [int(e_.split(';')[1]) for e_ in sp_pos]
            smask = [int(e_.split(';')[0]) for e_ in sp_mask]
            emask = [int(e_.split(';')[1]) for e_ in sp_mask]

            return spos, epos, smask, emask

        if mode == '6':
            # keep sp0, mask only gpt
            sp_pos = sp0
            sp_mask = sp1
            sp_mask = [ele for ele in sp_mask if ele not in sp_pos]

            spos = [int(e_.split(';')[0]) for e_ in sp_pos]
            epos = [int(e_.split(';')[1]) for e_ in sp_pos]
            smask = [int(e_.split(';')[0]) for e_ in sp_mask]
            emask = [int(e_.split(';')[1]) for e_ in sp_mask]

            return spos, epos, smask, emask

        if mode == '7':
            # keep sp0, mask only umls
            sp_pos = sp0
            sp_mask = sp3
            sp_mask = [ele for ele in sp_mask if ele not in sp_pos]

            spos = [int(e_.split(';')[0]) for e_ in sp_pos]
            epos = [int(e_.split(';')[1]) for e_ in sp_pos]
            smask = [int(e_.split(';')[0]) for e_ in sp_mask]
            emask = [int(e_.split(';')[1]) for e_ in sp_mask]

            return spos, epos, smask, emask
            
        return [], [], [], []


    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def run_dataset():
    """test dataset"""
    import os
    from datasets.collate_functions import collate_to_max_length
    from torch.utils.data import DataLoader
    # zh datasets
    bert_path = "/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12"
    vocab_file = os.path.join(bert_path, "vocab.txt")
    # json_path = "/mnt/mrc/zh_msra/mrc-ner.test"
    json_path = "/data/xiaoya/datasets/mrc_ner/zh_msra/mrc-ner.train"
    is_chinese = True

    # en datasets
    # bert_path = "/mnt/mrc/bert-base-uncased"
    # json_path = "/mnt/mrc/ace2004/mrc-ner.train"
    # json_path = "/mnt/mrc/genia/mrc-ner.train"
    # is_chinese = False

    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer,
                            is_chinese=is_chinese)

    dataloader = DataLoader(dataset, batch_size=1,
                            collate_fn=collate_to_max_length)

    for batch in dataloader:
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx in zip(*batch):
            tokens = tokens.tolist()
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            print(start_labels.numpy().tolist())

            tmp_start_position = []
            for tmp_idx, tmp_label in enumerate(start_labels.numpy().tolist()):
                if tmp_label != 0:
                    tmp_start_position.append(tmp_idx)

            tmp_end_position = []
            for tmp_idx, tmp_label in enumerate(end_labels.numpy().tolist()):
                if tmp_label != 0:
                    tmp_end_position.append(tmp_idx)

            if not start_positions:
                continue
            print("="*20)
            print(f"len: {len(tokens)}", tokenizer.decode(tokens, skip_special_tokens=False))
            for start, end in zip(start_positions, end_positions):
                print(str(sample_idx.item()), str(label_idx.item()) + "\t" + tokenizer.decode(tokens[start: end+1]))

            print("!!!"*20)
            for start, end in zip(tmp_start_position, tmp_end_position):
                print(str(sample_idx.item()), str(label_idx.item()) + "\t" + tokenizer.decode(tokens[start: end + 1]))


if __name__ == '__main__':
    run_dataset()
