from torchtext.data import Field,Example,Pipeline
import re
import jieba
import codecs
import subprocess
from datetime import datetime
from opencc import OpenCC


def get_xmllcsts(filename,limit=None):
    # regular_expression is suitable for LCSTS PART_I.txt, PART_II.txt, PART_III.txt
    pattern = re.compile(r'''<doc id=(?:\d+)>(?:\n\s+<human_label>(?:\d+)</human_label>)?
    <summary>\n\s+(.+)\n\s+</summary>
    <short_text>\n\s+(.+)\n\s+</short_text>\n</doc>''', re.M)
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = ''.join(f.readlines())
    lcsts_list = re.findall(pattern, content)[:limit]

    return lcsts_list

def jieba_tokenizer(text): # create a tokenizer function
    text = OpenCC('s2t').convert(text)
    #return [tok.text for tok in spacy_en.tokenizer(text)]
    return [tok for tok in jieba.lcut(text)]

#simple2trad = lambda x: OpenCC('t2s').convert(x)
def simple2trad(x):
    return list(map(lambda y: OpenCC('s2t').convert(y),x))

simple2trad_pipe = Pipeline(lambda x: OpenCC('s2t').convert(x))

s1 = datetime.now()
filename = './lcsts_xml/PART_I_10000.txt' 
TRG = Field( tokenize=jieba_tokenizer,include_lengths=True,
                init_token='<sos>', eos_token='<eos>')
SRC = Field(tokenize=jieba_tokenizer, include_lengths=True,
                init_token='<sos>', eos_token='<eos>',
                preprocessing=simple2trad_pipe)
fields = [('trg',TRG),('src',SRC)] 
lcsts_list = get_xmllcsts(filename,limit=50)

examples = list(map(lambda x :Example.fromlist(x,fields),lcsts_list))
s2 = datetime.now()
print('total execution time is '+ str((s2-s1).seconds))
print(len(examples))
print('example[0].src => ' + str(examples[0].src))
print('example[0].trg => ' + str(examples[0].trg))