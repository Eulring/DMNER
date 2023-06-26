# CUDA_VISIBLE_DEVICES=3 python 4-12_dict_umls_filter.py
import os
import pickle
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from torch.cuda.amp import autocast
from pytorch_metric_learning import miners, losses, distances

from transformers import (
    AutoTokenizer, 
    AutoModel, 
)
import ipdb
import random

LOGGER = logging.getLogger()


def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)

def predict_topk(model_wrapper, eval_dictionary, eval_queries, topk, cometa=False, agg_mode="cls"):

    encoder = model_wrapper.get_dense_encoder()
    tokenizer = model_wrapper.get_dense_tokenizer()
    
    # embed dictionary
    dict_names = [row[0] for row in eval_dictionary]
    
    print ("[start embedding dictionary...]")
    # ipdb.set_trace()
    dict_dense_embeds = model_wrapper.embed_dense(names=dict_names, show_progress=True, agg_mode=agg_mode)
    
    mean_centering = False
    if mean_centering:
        tgt_space_mean_vec = dict_dense_embeds.mean(0)
        dict_dense_embeds -= tgt_space_mean_vec

    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        golden_cui = eval_query[1].replace("+","|")
        
        dict_mentions = []

        
        for mention in mentions:
            mention_dense_embeds = model_wrapper.embed_dense(names=[mention], agg_mode=agg_mode)
            
            if mean_centering:
                mention_dense_embeds -= tgt_space_mean_vec

            # get score matrix
            dense_score_matrix = model_wrapper.get_score_matrix(
                    query_embeds=mention_dense_embeds, 
                    dict_embeds=dict_dense_embeds,
            )
            score_matrix = dense_score_matrix

            candidate_idxs = model_wrapper.retrieve_candidate_cuda(
                    score_matrix = score_matrix, 
                    topk = topk,
                    batch_size=16,
                    show_progress=False
            )

            #print(candidate_idxs.shape)
            np_candidates = [eval_dictionary[ind] for ind in candidate_idxs[0].tolist()]#.squeeze()
            dict_candidates = []
            for i, np_candidate in enumerate(np_candidates):
                dict_candidates.append({
                        'name':np_candidate[0],
                        'labelcui':np_candidate[1],
                        'label':check_label(np_candidate[1],golden_cui),
                        'score':score_matrix[0][candidate_idxs[0][i]]
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result


    

def predict_topk_fast(model_wrapper, eval_dictionary, eval_queries, topk, agg_mode="cls"):
    """
    for MedMentions only
    """

    encoder = model_wrapper.get_dense_encoder()
    tokenizer = model_wrapper.get_dense_tokenizer()
    
    # embed dictionary
    dict_names = [row[0] for row in eval_dictionary]
    print ("[start embedding dictionary...]")
    dict_dense_embeds = model_wrapper.embed_dense(names=eval_dictionary, show_progress=True, batch_size=4096, agg_mode=agg_mode)
    print ("dict_dense_embeds.shape:", dict_dense_embeds.shape)
    
    bs = 32
    candidate_idxs = None
    candidate_scores = None
    print ("[computing rankings...]")


    for i in tqdm(np.arange(0,len(eval_queries),bs), total=len(eval_queries)//bs+1):
        mentions = eval_queries[i:i+bs]
        mentions = [ele[0] for ele in mentions]
        mention_dense_embeds = model_wrapper.embed_dense(names=mentions, agg_mode=agg_mode)

        # get score matrix
        dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=mention_dense_embeds, 
            dict_embeds=dict_dense_embeds
        )
        score_matrix = dense_score_matrix
        candidate_idxs_batch = model_wrapper.retrieve_candidate_cuda(
            score_matrix = score_matrix, 
            topk = topk,
            batch_size=bs,
            show_progress=False
        )
        if candidate_idxs is None:
            candidate_idxs = candidate_idxs_batch
            # ipdb.set_trace()
        else:
            candidate_idxs = np.vstack([candidate_idxs, candidate_idxs_batch])
            
    golden_cuis = [ele[1] for ele in eval_queries]
    mentions = [ele[0] for ele in eval_queries]
    results = []
    
    print ("[writing json...]")
    for i in tqdm(range(len(eval_queries))):
        np_candidates = [eval_dictionary[ind] for ind in candidate_idxs[i].tolist()]#.squeeze()
        
        unit = {
            'name': mentions[i],
            'gold': golden_cuis[i],
            'preds': [ele[1] for ele in np_candidates],
            'pred_names': [ele[0] for ele in np_candidates]
        }
        results.append(unit)

    return results


def predict_cluster(model_wrapper, eval_dictionary, eval_queries, ava_etype, agg_mode="cls"):
    encoder = model_wrapper.get_dense_encoder()
    tokenizer = model_wrapper.get_dense_tokenizer()
    
    # embed dictionary
    dict_names = [row[0] for row in eval_dictionary]
    dict_types = [row[1] for row in eval_dictionary]
    type_index = {}
    for key in ava_etype:
        type_index[key] = [i for i, dtype in enumerate(dict_types) if dtype == key]
    type_index['Other'] = [i for i, dtype in enumerate(dict_types) if dtype not in ava_etype]

    print ("[start embedding dictionary...]")
    dict_dense_embeds = model_wrapper.embed_dense(names=eval_dictionary, show_progress=True, batch_size=4096, agg_mode=agg_mode)
    print ("dict_dense_embeds.shape:", dict_dense_embeds.shape)
    
    bs = 32
    scores_matrix = None
    cluster_sims = []

    print ("[computing rankings...]")

    for i in tqdm(np.arange(0,len(eval_queries),bs), total=len(eval_queries)//bs+1):
        mentions = eval_queries[i:i+bs]
        mentions = [ele[0] for ele in mentions]
        mention_dense_embeds = model_wrapper.embed_dense(names=mentions, agg_mode=agg_mode)
        # get score matrix
        dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=mention_dense_embeds, 
            dict_embeds=dict_dense_embeds
        )
        score_matrix = dense_score_matrix

        for j in range(len(mentions)):
            unit = {}
            for key in type_index.keys():
                scores = score_matrix[j][type_index[key]]
                unit[key] = np.average(scores)
            cluster_sims.append(unit)

    queries = []
    golden_cuis = [ele[1] for ele in eval_queries]
    mentions = [ele[0] for ele in eval_queries]
    
    print ("[writing json...]")
    results = []
    for i in tqdm(range(len(eval_queries))):
        unit = {
            'name': eval_queries[i][0],
            'gold_type': eval_queries[i][1],
            'cluster': cluster_sims[i]
        }
        results.append(unit)

    return results


def sample_k_fast(model_wrapper, eval_dictionary, eval_queries, topk, lastk, agg_mode="cls"):
    """
    for MedMentions only
    """

    encoder = model_wrapper.get_dense_encoder()
    tokenizer = model_wrapper.get_dense_tokenizer()
    
    # embed dictionary
    dict_names = [row[0] for row in eval_dictionary]
    print ("[start embedding dictionary...]")
    dict_dense_embeds = model_wrapper.embed_dense(names=dict_names, show_progress=True, batch_size=4096, agg_mode=agg_mode)
    print ("dict_dense_embeds.shape:", dict_dense_embeds.shape)
    
    bs = 32
    candidate_idxs = None
    candidate_idxs_neg = None
    print ("[computing rankings...]")

    for i in tqdm(np.arange(0,len(eval_queries),bs), total=len(eval_queries)//bs+1):
        # ipdb.set_trace()
        mentions = eval_queries[i:i+bs]
        mention_dense_embeds = model_wrapper.embed_dense(names=mentions, agg_mode=agg_mode)

        # get score matrix
        dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=mention_dense_embeds, 
            dict_embeds=dict_dense_embeds
        )
        score_matrix = dense_score_matrix

        candidate_idxs_batch_neg = model_wrapper.retrieve_negative_cuda(
            score_matrix = score_matrix, 
            topk = lastk,
            batch_size=bs,
            show_progress=False
        )

        candidate_idxs_batch = model_wrapper.retrieve_candidate_cuda(
            score_matrix = score_matrix, 
            topk = topk,
            batch_size=bs,
            show_progress=False
        )
        # candidate_idxs_batch_neg = candidate_idxs_batch

        if candidate_idxs is None: candidate_idxs = candidate_idxs_batch
        else: candidate_idxs = np.vstack([candidate_idxs, candidate_idxs_batch])
        
        if candidate_idxs_neg is None: candidate_idxs_neg = candidate_idxs_batch_neg
        else: candidate_idxs_neg = np.vstack([candidate_idxs_neg, candidate_idxs_batch_neg])

    queries = []
    golden_cuis = [ele[1] for ele in eval_queries]
    mentions = [ele[0] for ele in eval_queries]
    print ("[writing json...]")
    for i in tqdm(range(len(eval_queries))):
        np_candidates = [eval_dictionary[ind] for ind in candidate_idxs[i].tolist()]#.squeeze()
        np_candidates_neg = [eval_dictionary[ind] for ind in candidate_idxs_neg[i].tolist()]#.squeeze()
        
        dict_candidates = []
        dict_candidates_neg = []
        dict_mentions = []
        
        for np_candidate in np_candidates:
            dict_candidates.append({
                'name':np_candidate[0],
                'labelcui':np_candidate[1]
            })

        for np_candidate in np_candidates_neg:
            dict_candidates_neg.append({
                'name':np_candidate[0],
                'labelcui':np_candidate[1]
            })

        dict_mentions.append({
            'mention':mentions[i],
            'golden_cui':golden_cuis[i], # golden_cui can be composite cui
            'candidates':dict_candidates,
            'candidates_neg': dict_candidates_neg
        })
        queries.append({ 'mentions':dict_mentions })
    
    result = {'queries':queries}

    return result


def evaluate(model_wrapper, eval_dictionary, eval_queries, topk, cometa=False, medmentions=False, agg_mode="cls"):
    result = predict_topk_fast(model_wrapper, eval_dictionary, eval_queries, topk, agg_mode=agg_mode)
    return result

def evaluate_cluster(model_wrapper, eval_dictionary, eval_queries, ava_etype, cometa=False, medmentions=False, agg_mode="cls"):
    result = predict_cluster(model_wrapper, eval_dictionary, eval_queries, ava_etype, agg_mode=agg_mode)
    return result

def sample2(model_wrapper, eval_dictionary, eval_queries, topk, lastk, cometa=False, medmentions=False, agg_mode="cls"):
    result = sample_k_fast(model_wrapper, eval_dictionary, eval_queries, topk, lastk, agg_mode=agg_mode)
    return result



class Model_Wrapper(object):
    """
    Wrapper class for BERT encoder
    """

    def __init__(self):
        self.tokenizer = None
        self.encoder = None

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def save_model(self, path, context=False):
        # save bert model, bert config
        self.encoder.save_pretrained(path)

        # save bert vocab
        self.tokenizer.save_pretrained(path)
        

    def load_model(self, path, max_length=25, use_cuda=True, lowercase=True):
        self.load_bert(path, max_length, use_cuda)
        
        return self

    def load_bert(self, path, max_length, use_cuda, lowercase=True):
        # ipdb.set_trace()
        self.tokenizer = AutoTokenizer.from_pretrained(path, 
                use_fast=True, do_lower_case=lowercase)
        self.encoder = AutoModel.from_pretrained(path)
        if use_cuda:
            self.encoder = self.encoder.cuda()

        return self.encoder, self.tokenizer
    

    def get_score_matrix(self, query_embeds, dict_embeds, cosine=False, normalise=False):
        """
        Return score matrix

        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings

        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        if cosine:
            score_matrix = cosine_similarity(query_embeds, dict_embeds)
        else:
            score_matrix = np.matmul(query_embeds, dict_embeds.T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min() ) / (score_matrix.max() - score_matrix.min())
        
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """
        
        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix) 
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def retrieve_candidate_cuda(self, score_matrix, topk, batch_size=128, show_progress=False):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        res = None
        for i in tqdm(np.arange(0, score_matrix.shape[0], batch_size), disable=not show_progress):
            score_matrix_tmp = torch.tensor(score_matrix[i:i+batch_size]).cuda()
            matrix_sorted = torch.argsort(score_matrix_tmp, dim=1, descending=True)[:, :topk].cpu()
            if res is None: 
                res = matrix_sorted
            else:
                res = torch.cat([res, matrix_sorted], axis=0)

        return res.numpy()


    def retrieve_negative_cuda(self, score_matrix, topk, batch_size=128, show_progress=False, thres=110):
        res = None
        # negrank = int(score_matrix.shape[1] * 0.7)
        bs, n = score_matrix.shape[0], score_matrix.shape[1]

        for i in tqdm(np.arange(0, score_matrix.shape[0], batch_size), disable=not show_progress):
            score_matrix_tmp = torch.tensor(score_matrix[i:i+batch_size]).cuda()
            # matrix_sorted = torch.argsort(score_matrix_tmp, dim=1, descending=True)[:, :topk].cpu()     
            for j in range(score_matrix.shape[0]):
                neg_idxs = []
                for j_ in range(n):
                    if score_matrix[j][j_] < thres:
                        neg_idxs.append(j_)
                    if len(neg_idxs) >= 2 * topk:
                        break
                # ipdb.set_trace()
                try:
                    neg_sample = np.array(random.sample(neg_idxs, topk))
                except:
                    neg_sample = np.array(neg_idxs)
                # neg_sample = np.array(neg_idxs)
                if res is None: res = neg_sample
                else: 
                    try:
                        res = np.vstack((res, neg_sample))
                    except:
                        ipdb.set_trace()

        return res
    

    def embed_dense(self, names, show_progress=False, batch_size=2048, agg_mode="cls"):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval() # prevent dropout
        
        batch_size=batch_size
        dense_embeds = []

        #print ("converting names to list...")
        #names = names.tolist()

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, len(names), batch_size))
            else:
                iterations = range(0, len(names), batch_size)
                
            for start in iterations:
                end = min(start + batch_size, len(names))
                batch = names[start:end]
                # ipdb.set_trace()
                batch_tokenized_names = self.tokenizer.batch_encode_plus(
                        batch, add_special_tokens=True, 
                        truncation=True, max_length=25, 
                        padding="max_length", return_tensors='pt')
                batch_tokenized_names_cuda = {}
                for k,v in batch_tokenized_names.items(): 
                    batch_tokenized_names_cuda[k] = v.cuda()
                
                last_hidden_state = self.encoder(**batch_tokenized_names_cuda)[0]
                if agg_mode == "cls":
                    batch_dense_embeds = last_hidden_state[:,0,:] # [CLS]
                elif agg_mode == "mean_all_tok":
                    batch_dense_embeds = last_hidden_state.mean(1) # pooling
                elif agg_mode == "mean":
                    batch_dense_embeds = (last_hidden_state * batch_tokenized_names_cuda['attention_mask'].unsqueeze(-1)).sum(1) / batch_tokenized_names_cuda['attention_mask'].sum(-1).unsqueeze(-1)
                else:
                    print ("no such agg_mode:", agg_mode)

                batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds)
        # ipdb.set/
        dense_embeds = np.concatenate(dense_embeds, axis=0)
        
        return dense_embeds




class Sap_Metric_Learning(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda, pairwise, 
            loss, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls"):

        LOGGER.info("Sap_Metric_Learning! learning_rate={} weight_decay={} use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
            learning_rate,weight_decay,use_cuda,loss,use_miner,miner_margin,type_of_triplets,agg_mode
        ))
        super(Sap_Metric_Learning, self).__init__()
        self.encoder = encoder
        self.pairwise = pairwise
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07) # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()


        print ("miner:", self.miner)
        print ("loss:", self.loss)
    
    @autocast() 
    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        
        last_hidden_state1 = self.encoder(**query_toks1, return_dict=True).last_hidden_state
        last_hidden_state2 = self.encoder(**query_toks2, return_dict=True).last_hidden_state
        if self.agg_mode=="cls":
            query_embed1 = last_hidden_state1[:,0]  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2[:,0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            query_embed1 = last_hidden_state1.mean(1)  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            query_embed1 = (last_hidden_state1 * query_toks1['attention_mask'].unsqueeze(-1)).sum(1) / query_toks1['attention_mask'].sum(-1).unsqueeze(-1)
            query_embed2 = (last_hidden_state2 * query_toks2['attention_mask'].unsqueeze(-1)).sum(1) / query_toks2['attention_mask'].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        
        labels = torch.cat([labels, labels], dim=0)
        
        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs) 
        else:
            return self.loss(query_embed, labels) 


    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table



