import torchtext.data as data
import codecs
import json

path_json = './multi30k_json/train_array.json'
with codecs.open(path_json, encoding='utf-8') as fp:
    json_array = json.load(fp)

DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {'DE':('de',DE),'EN':('en',EN)}
examples = list(map(lambda x:data.Example.fromdict(x,fields),json_array))

print('type of manual examples ==> ' + str(type(examples)))
print('__dict__.keys() in examples element ==> ' + str(examples[0].__dict__.keys()))
print('len of manual examples ==> ' + str(len(examples)))
print('examples[0].de ==> ' + str(examples[0].de))
print('examples[0].en ==> ' + str(examples[0].en))

'''
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
'''