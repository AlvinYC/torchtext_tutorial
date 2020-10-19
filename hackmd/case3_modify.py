import torchtext.data as data
import codecs
import subprocess
import ast 

path_json = './multi30k_json/train.json'
def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = f.read().splitlines()
    return content
contents = read_text(path_json)
dict_list = list(map(lambda x: ast.literal_eval(x),contents))

DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {'DE':('de',DE),'EN':('en',EN)}
train_examples = list(map(lambda x:data.Example.fromdict(x,fields),dict_list))

print('type of manual examples ==> ' + str(type(train_examples)))
print('__dict__.keys() in examples element ==> ' + str(train_examples[0].__dict__.keys()))
print('len of manual examples ==> ' + str(len(train_examples)))
print('examples[0].de ==> ' + str(train_examples[0].de))
print('examples[0].en ==> ' + str(train_examples[0].en))
fields = [('de',DE),('en',EN)]              # example 的 field 是 list 格式 
train_dataset = data.Dataset(examples=train_examples,fields=fields)
DE.build_vocab(train_dataset)
EN.build_vocab(train_dataset)
train_bucketiter = data.BucketIterator(dataset=train_dataset,batch_size=32, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))