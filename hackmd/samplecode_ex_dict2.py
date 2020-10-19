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
examples = list(map(lambda x:data.Example.fromdict(x,fields),dict_list))

print('type of manual examples ==> ' + str(type(examples)))
print('__dict__.keys() in examples element ==> ' + str(examples[0].__dict__.keys()))
print('len of manual examples ==> ' + str(len(examples)))
print('examples[0].de ==> ' + str(examples[0].de))
print('examples[0].en ==> ' + str(examples[0].en))
