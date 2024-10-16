# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import logging
import argparse
import math
import os
import sys
import random
import numpy
import pandas as pd

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
import model_config as info

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_{1}_tokenizer.dat'.format(opt.dataset,opt.dataset_mode))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_{2}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset, opt.dataset_mode))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            if opt.only_embedding:
                raise SystemExit("Exiting program, Embedding and tokenizer files are generated")

        self.tokenizer = tokenizer

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer, mode = opt.dataset_mode)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer, mode = opt.dataset_mode)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, df_metrics = self._evaluate_acc_f1(val_data_loader,aspect="all")

            pd.options.display.float_format = '{:.4f}'.format
            logger.info("val metrics:")
            logger.info(df_metrics)
            logger.info("--------------------------------")
            logger.info('val_acc: {:.4f}'.format(val_acc))
            val_f1 = df_metrics["Average"][0]

            # logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc))

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                path = 'state_dict/{0}_{1}_{2}/'.format(self.opt.dataset , self.opt.model_name, self.opt.dataset_mode)
                if not os.path.exists(path):
                    os.mkdir(path)
                path = '{0}val_acc_{1}'.format(path,round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _evaluate_acc_f1(self, data_loader,aspect):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        process = True
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                if aspect != "all":
                    # evalute model for one aspect
                    aspect_index = torch.tensor(self.tokenizer.text_to_sequence(aspect))
                    aspect_index_broadcast = aspect_index.unsqueeze(0).expand_as(t_batch['aspect_indices'])
                    mask = torch.all(t_batch['aspect_indices'] == aspect_index_broadcast, dim=1)
                    # Only proceed if there is at least one match
                    process = mask.any()

                if process:
                    if aspect != "all":
                        t_batch = {k: v[mask] for k, v in t_batch.items()}
                    t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                    t_targets = t_batch['polarity'].to(self.opt.device)
                    t_outputs = self.model(t_inputs)

                    n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                    n_total += len(t_outputs)

                    if t_targets_all is None:
                        t_targets_all = t_targets
                        t_outputs_all = t_outputs
                    else:
                        t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                        t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        y_true = t_targets_all.cpu()
        y_pred = torch.argmax(t_outputs_all, -1).cpu()

        f1 = {}
        f1["f1_avg"] = metrics.f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')
        f1["f1_neg"] = metrics.f1_score(y_true,y_pred, labels=[0], average='macro')
        f1["f1_neutral"] = metrics.f1_score(y_true, y_pred, labels=[1], average='macro')
        f1["f1_pos"] = metrics.f1_score(y_true, y_pred, labels=[2], average='macro')

        precision = {}
        precision["precision_avg"] = metrics.precision_score(y_true, y_pred, labels=[0, 1, 2], average='macro')
        precision["precision_neg"] = metrics.precision_score(y_true, y_pred, labels=[0], average='macro')
        precision["precision_neutral"] = metrics.precision_score(y_true, y_pred, labels=[1], average='macro')
        precision["precision_pos"] = metrics.precision_score(y_true, y_pred, labels=[2], average='macro')

        recall = {}
        recall["recall_avg"] = metrics.recall_score(y_true, y_pred, labels=[0, 1, 2], average='macro')
        recall["recall_neg"] = metrics.recall_score(y_true, y_pred, labels=[0], average='macro')
        recall["recall_neutral"] = metrics.recall_score(y_true, y_pred, labels=[1], average='macro')
        recall["recall_pos"] = metrics.recall_score(y_true, y_pred, labels=[2], average='macro')

        data = {
            "Negative": [f1["f1_neg"], precision["precision_neg"], recall["recall_neg"]],
            "Neutral": [f1["f1_neutral"], precision["precision_neutral"], recall["recall_neutral"]],
            "Positive": [f1["f1_pos"], precision["precision_pos"], recall["recall_pos"]],
            "Average":[f1["f1_avg"], precision["precision_avg"], recall["recall_avg"]]
        }
        
        index = ["F1", "Precision", "Recall"]
        metrics_classification = pd.DataFrame(data, index=index)

        return acc,metrics_classification
    

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        # best_model_path = 'state_dict/aoa_coursera_val_acc_0.837'
        self.model.load_state_dict(torch.load(best_model_path))
        aspects = ["all" , "the course" , "the teacher"]
        aspect = "all"
        for aspect in aspects:
            test_acc, metrics  = self._evaluate_acc_f1(test_data_loader, aspect = aspect)
            pd.options.display.float_format = '{:.4f}'.format
            logger.info("ASPECT : {}".format(aspect))
            logger.info("test metrics:")
            logger.info(metrics)
            logger.info("--------------------------------")
            logger.info('test_acc: {:.4f}'.format(test_acc))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--model_name', default='aoa', type=str)
    parser.add_argument('--dataset', default='preprocessed_coursera', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--only_embedding', default=False, type=bool, help='only generate embedding embdding and tokenizer matrices using glove the break the code ')
    # parser.add_argument('--only_evaluate', default=False, type=str, help='only evaluate the model, without training ')
    # dataset_mode for coursera datasets
    # 'neg_false_keep_all'
    # 'neg_flase_del_all'
    # 'neg_flase_del_except_neg'
    # 'neg_true_keep_all'
    # 'neg_true_del_all'
    # 'neg_true_del_except_neg'
    parser.add_argument('--dataset_mode', default="", type=str, help='dataset to train the model on')
    # parser.add_argument('--evaluate_aspect', default="all", type=str, help='evaluate test dataset on this aspect ')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = info.model_classes
    dataset_files = info.dataset_files
    input_colses = info.input_colses

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = 'log/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
