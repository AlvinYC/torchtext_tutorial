import codecs
import subprocess
import torchtext.data as data

path_csv  = './multi30k_csv/train.csv'
path_json = './multi30k_json/train.json'

def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = f.read().splitlines()
    return content

list_from_csv = list(map(lambda x: x.split('\t'), read_text(path_csv)))

DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = [('de',DE),('en',EN)] 
examples = list(map(lambda x:data.Example.fromlist(x,fields),list_from_csv))

print('type of manual examples ==> ' + str(type(examples)))
print('__dict__.keys() in examples element ==> ' + str(examples[0].__dict__.keys()))
print('len of manual examples ==> ' + str(len(examples)))
print('examples[0].de ==> ' + str(examples[0].de))
print('examples[0].en ==> ' + str(examples[0].en))