import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}    # json 的 field 是dict 格式
train =  data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
train_examples = train.examples
#-----------------------------
fields = [('de',DE),('en',EN)]              # example 的 field 是 list 格式 
train_dataset = data.Dataset(examples=train_examples,fields=fields)
#如果要把 data.Dataset object 切成三份，用法與 data.TabularDataset 一樣，直接call method split即可
train_s1, valid_s1, test_s1 = train_dataset.split(split_ratio=[0.8,0.1,0.1])

print('len train.examples = ' + str(len(train_s1.examples)))
print('len valid.examples = ' + str(len(valid_s1.examples))) 
print('len test.examples  = ' + str(len(test_s1.examples)))

