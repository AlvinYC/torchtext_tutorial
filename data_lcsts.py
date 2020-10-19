#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:24:23 2020

@author: c95yyw
"""

import re
#import pandas as pd
from random import shuffle


if __name__ == "__main__":
    pattern = re.compile(r'''<doc id=(\d+)>
    <human_label>\d+</human_label>
    <summary>
        (.+)
    </summary>
    <short_text>
        (.+)
    </short_text>
</doc>''', re.M)

    with open('./lcsts_xml/PART_III.txt', encoding='utf-8') as f:
        text = ''.join(f.readlines())
    #matches = re.findall(pattern, text)[:5000]
    matches = re.findall(pattern, text)[:500]
    #shuffle(matches)
    #matches = matches[:512]
    '''
    df = pd.DataFrame(matches)
    '''
    #df.to_csv('./data/big_valid_lcsts.csv', sep='\t', header=False, index=False)
    '''
    df.iloc[:,1].to_csv('./data/test.tgt',header=False, index=False)
    df.iloc[:,2].to_csv('./data/test.src',header=False, index=False)
    '''
    #<human_label>\d+</human_label>