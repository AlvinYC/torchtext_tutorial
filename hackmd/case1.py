import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
#----------------------
DE.build_vocab(train)
EN.build_vocab(train)
train_bucketiter = data.BucketIterator(dataset=train,batch_size=5, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))

