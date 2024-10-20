
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT , OURS
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC

model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        'ours' : OURS,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }

dataset_files = {
    'twitter': {
        'train': './datasets/acl-14-short-data/train.raw',
        'test': './datasets/acl-14-short-data/test.raw'
    },
    'restaurant': {
        'train': './datasets/semeval14/Restaurants_Train.xml.seg',
        'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    },
    'laptop': {
        'train': './datasets/semeval14/Laptops_Train.xml.seg',
        'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
    },
    'coursera': {
        'train': './datasets/coursera/train_Coursera_dataset.seg',
        'test': './datasets/coursera/test_Coursera_dataset.seg'
    },
    'preprocessed_coursera': {
        'train': './datasets/coursera/train_preprocessed_coursera.seg',
        'test': './datasets/coursera/test_preprocessed_coursera.seg'
    }
}

input_colses = {
    'lstm': ['text_indices'],
    'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
    'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
    'atae_lstm': ['text_indices', 'aspect_indices'],
    'ian': ['text_indices', 'aspect_indices'],
    'memnet': ['context_indices', 'aspect_indices'],
    'ram': ['text_indices', 'aspect_indices', 'left_indices'],
    'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
    'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
    'aoa': ['text_indices', 'aspect_indices'],
    'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
    'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
    'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
    'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
    'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    # 'ours' : ['text_indices', 'aspect_indices','text_bert_indices','aspect_bert_indices']
    'ours' : ['text_indices', 'aspect_indices']
}