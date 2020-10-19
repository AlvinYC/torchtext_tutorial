import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
print('len train.examples = ' + str(len(train.examples)))
tencross = train.split(split_ratio=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
print('train_s1 length = '+str(len(train_s1.examples)))
print('valid_s1 length = '+str(len(valid_s1.examples)))
print('test_s1 length = '+str(len(test_s1.examples)))
