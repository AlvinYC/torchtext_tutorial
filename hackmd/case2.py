import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
#----------------------
DE.build_vocab(train)
EN.build_vocab(train)
train_iter = data.Iterator(dataset=train,batch_size=32,shuffle=False,train=True)
# retrive frist batche
train_batch0 = next(iter(train_iter))
