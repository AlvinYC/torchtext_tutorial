from torchtext.data import Field,Example,Pipeline
from torchtext.data import TabularDataset,Dataset, BucketIterator
import re
import jieba
import codecs
import numpy as np
import subprocess
from datetime import datetime
import random
def custom_batch(self, batch):
    r = random.randint(4,233)
    # overwrite index 2,3,4 as random word 'r=46'
    # such as [2, 46, 46, 46, 30, 305, 17, 82, 6, 12, 52, 0, 0, 3, 1, 1, 1, 1, 1]

    return list(map(lambda example: [r if i>0 and i<4 else x for i,x in enumerate(example)] , batch))
#def custom_batch_pipe(ba)

def load_dataset_txt(batch_size,macbook=False):
    DE = Field(include_lengths=True,
                init_token='<sos>', eos_token='<eos>', postprocessing=custom_batch)
    EN = Field(include_lengths=True,
                init_token='<sos>', eos_token='<eos>', postprocessing=custom_batch)
    fields = {"DE":('src',DE),"EN":('trg',EN)}
    train, val, test = TabularDataset.splits(path='./multi30k_json/',format='json',fields=fields,
                                                train='train.json',validation='valid.json',test='test.json')
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/50)]
        val.examples = train.examples[0:int(len(val.examples)/50)]
        test.examples = train.examples[0:int(len(test.examples)/50)]

    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=True, shuffle=False)
    batch = next(iter(train_iter))
    return train_iter, val_iter, test_iter, DE, EN


if __name__ == "__main__":
    s1 = datetime.now() 
    (train_iter, val_iter, test_iter, DE, EN) = load_dataset_txt(batch_size=32,macbook=True)
    s2 = datetime.now()
    print('total execution time is '+ str((s2-s1).seconds))
    #print(len(examples))
    #print('example[0].src => ' + str(examples[0].src))
    #print('example[0].trg => ' + str(examples[0].trg))