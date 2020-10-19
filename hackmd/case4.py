import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}    # json 的 field 是dict 格式
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
train_examples = train.examples
#----------------------
fields = [('de',DE),('en',EN)]              # example 的 field 是 list 格式
train_dataset = data.Dataset(examples=train_examples,fields=fields)
DE.build_vocab(train_dataset)
EN.build_vocab(train_dataset)
train_iter = data.Iterator(dataset=train_dataset,batch_size=32,shuffle=False,train=True)
# retrive frist batche
train_batch = next(iter(train_iter))

