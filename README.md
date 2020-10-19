# torchtext toturial - 如何秒殺NLP資料集

- [torchtext 資料結構](#torchtext-資料結構)
  - [torchtext.data 主結構](#torchtextdata-主結構)
    - 資料儲存單位 dataset, Batch, Example
    - 迭代器 iterators 
    - Fields
  - [Pipline](#Pipline)
  - [其他功能](#其他功能)
  - [torchext標準使用方式](#torchext標準使用方式)
- [使用情境](#使用情境)
  - [情境1: TabularDataset/BucketIterator(建議)](#情境1:-TabularDatasetBucketIterator(建議))
  - [情境2: TabularDataset/Iterator](#情境2:-TabularDatasetIterator)
  - [情境3: Dataset/BucketIterator](#情境3:-DatasetBucketIterator)
  - [情境4: Dataset/Iterator](#情境4:-DatasetIterator)
- [情境比較](#情境比較)
  - [情境1 vs 情境2](#情境1-vs-情境2)
  - [情境3 vs 情境4](#情境3-vs-情境4)
- [其他使用方式](#其他使用方式)
  - [手動建置 Example](#手動建置-Example)
    - [Example from list](#Example-from-list)
    - [Example from dict](#Example-from-dict)
        - JSON array 處理方式
        - dict list 處理方式
    - [fromlist/fromdict 比較](#fromlist/fromdict-比較)
    - [情境3 修改為人工建置 Example](#情境3-修改為人工建置-Example)
  - [自動拆分資料集 split](#自動拆分資料集-split)
    - 注意事項
  - [padding 策略](#padding-策略)
    - [每句話前後自動補\<sos\>\<eos\>](#每句話前後自動補\<sos\>\<eos\>)
- [Multi30 DE2EN 資料集](#Multi30-DE2EN-資料集)
  - [introduction](#introduction)
    - [Multi30k by github](#Multi30k-by-github) 
    - [sample](#sample)
    - [multi30k raw to JSON](#multi30k-raw-to-JSON )
  - [CSV format](#CSV-format)
    - [CSV sample](#CSV-sample)
    - [multi30k raw to csv format](#multi30k-raw-to-csv-format)
- [seq2seq 應用](#seq2seq-應用)
  - [KEON seq2seq encode](#KEON-seq2seq-encode)
  - [load_dataset by spacy](#load_dataset-by-spacy)
  - [load_dataset by json](#load_dataset-by-json)
  - [假如不用torchtext](#假如不用torchtext)
- [進階用法](#進階用法)
  - [中文處理/斷詞](#中文處理/斷詞)
  - [preprocessing 與 postprocessing](#preprocessing-與-postprocessing)
    - [preprocessing](#preprocessing)
    - [使用 data.Pipeline 加速 preprocessing](#使用-data.Pipeline-加速-preprocessing)
    - [postprocessing](#postprocessing)
    - [使用 data.Pipeline 加速 postprocessing](#使用-data.Pipeline-加速-postprocessing)
  - [使用 Word2Vector (W2V)](#使用-Word2Vector-(W2V))

[[top]](#torchtext-toturial---如何秒殺NLP資料集)

# torchtext 資料結構
## torchtext.data 主結構
### 資料儲存單位 dataset, Batch, Example

   - torchtext.data.__Dataset__(examples, fields, filter_pred=None)
   資料已經是example形式，並且已知field，將該二資訊轉換為torchtext.data.Dataset 格式 
   - torchtext.data.__TabularDataset__(path, format, fields, skip_header=False, csv_reader_params={}, **kwargs)
   - torchtext.data.__Batch__(data=None, dataset=None, device=None）
   - torchtext.data.__Example__

[[top]](#torchtext-toturial---如何秒殺NLP資料集)

### 迭代器 iterators 
   - torchtext.data.__Iterator__(dataset, batch_size, sort_key=None, device=None, batch_size_fn=None, train=True, repeat=False, shuffle=None, sort=None, sort_within_batch=None)
   - torchtext.data.__BucketIterator__(dataset, batch_size, sort_key=None, device=None, batch_size_fn=None, train=True, repeat=False, shuffle=None, sort=None, sort_within_batch=None)
   - torchtext.data.__BPTTIterator__(dataset, batch_size, bptt_len, **kwargs)
   
torchtext 支援三種迭代器，從參數可以看到都是以 batch 為單位。
[[top]](#torchtext-toturial---如何秒殺NLP資料集)

### Fields

   - torchtext.data.__RawField__(preprocessing=None, postprocessing=None, is_target=False) 
   - torchtext.data.__Field__(sequential=True, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False, tokenize=None, tokenizer_language=\'en\', include_lengths=False, batch_first=False, pad_token=\'<pad>\', unk_token=\'<unk>\', pad_first=False, truncate_first=False, stop_words=None, is_target=False)
   - torchtext.data.__ReversibleField__(**kwargs)
   - torchtext.data.__SubwordField__(**kwargs)
   - torchtext.data.__NestedField__(nesting_field, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.int64, preprocessing=None, postprocessing=None, tokenize=None, tokenizer_language='en', include_lengths=False, pad_token=\'<pad>\', pad_first=False, truncate_first=False)

[[top]](#torchtext-toturial---如何秒殺NLP資料集)

## Pipline
   - torchtext.data.__Pipeline__(convert_token=None)[[top]](#torchtext-toturial---如何秒殺NLP資料集)  

## 其他功能
   - torchtext.data.__batch__(data, batch_size, batch_size_fn=None)
   從dataset（如TabluraDataset object) 直接取出 example，並以list方式回傳，比較直觀
   - torchtext.data.__pool__(data, batch_size, key, batch_size_fn=<function <lambda>>, random_shuffler=None, shuffle=False, sort_within_batch=False)
   - torchtext.data.__get_tokenizer__(tokenizer, language='en')

[[top]](#torchtext-toturial---如何秒殺NLP資料集)   

## torchext標準使用方式
torchtext.data.__Dataset__(examples, fields, filter_pred=None)
   資料已經是example形式，並且已知field，將該二資訊轉換為torchtext.data.Dataset 格式 
   ** 請先參考 torchtext.data.examples 與 torchtext.data.field後再閱讀次章節
```python=
import torchtext.data as data
# step 1: 將檔案的文本資料匯入torchtext
# 這個部份是torchtext很大的貢獻之一，若是標準的文字檔格式 csv/tsv/json不需要額外寫任何paring程式就可以直接匯入
# 若資料集不是以上格式torchtext也可以透過輔助縮短paring的開發時程
# 後續會詳細說明資料集為非標準格式的撰寫方式，此處以json格式為範例說明
DE = data.Field()
EN = data.Field()
fields = {"DE":('de',DE),"EN":('en',EN)}    # 配合 .json 文字檔的格
train = data.TabularDataset(path='./multi30k/train.json',format='json',fields=fields)


# step 2: 建立辭典id對照表
DE.build_vocab(train)
EN.build_vocab(train)
# step 3: 製作以 batch 為單位的迭代器 iterator
train_bucketiter = data.BucketIterator(dataset=train,batch_size=32, sort_key=False, train=True, shuffle=False)
# 開始訓練
for batch in train_bcuketitr:
    model(batch)
```
參數訊息
```python
>>> type(train) # 看不出什麼，只知道整個 train 是一個 TabularDataset 物件    
<class 'torchtext.data.dataset.TabularDataset'> 

>>> train.__dick__.kesy() # 但是可以知道整個物件只有兩個欄位，
dict_keys(['examples', 'fields'])
# examplls 存放了資料集讀進來的所有資料
# fields 就是 code line 8 所寫的資料集欄位設定

>>> type(train.examples)  # 表示資料每一筆都是放到list裡面，可以直接觀察
<class 'list'>
>>> type(train.fields)    # example 裡面是以這個格式存放
<class 'dict'>

>>> train.field # 由於我們已經知道 train.field 是dict 所以直接看內容
{'de': <torchtext.data.field.Field object at 0x106f1e470>, 'en': <torchtext.data.field.Field object at 0x106f1e438>}

>>> type(train.examples[0]) # 這裏就發現，list中每一個元素都是 Example 物件
# 也就是說我們只要能夠知道 Example內容是什麼，
# 我們就可以參透 torchtext 資料的基本處理方法
class 'torchtext.data.example.Example'>


>>> vars(train.examples[0]) # list元素是物件，所以直接使用 vars 觀察物件內容
# 發現沒有那麼複雜，再把 field 放進來一起想
# 可以發現，Example 這個物件的內容就是 field 所設定的格式
{'de': ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche.'], 
 'en': ['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.']}

>>> train.examples[0].__dict__.keys() #再確認一次
dict_keys(['de', 'en'])
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
# 使用情境
## 情境1: TabularDataset/BucketIterator(建議)
```python
import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
#----------------------
#下面這行用法是錯誤的，因為json field是 dict format，Example需要的是list format
#詳細操作參考本頁情境3 Dataset+BucketIterator 
#train_dataset = data.Dataset(examples=train_examples,fields=fields) 
DE.build_vocab(train)
EN.build_vocab(train)
train_bucketiter = data.BucketIterator(dataset=train,batch_size=32, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))

#---------------------- 以下都是資料驗證
print(train_batch)
print('train_batch.en ==> col-basd data format \n' + str(train_batch.en))
# data is saved as col-based rather than row-based, we should use [:,0] to retrive first sentence in first batch 
print('trian_batch.en[:,0] ==> first column \n ' + str(train_batch.en[:,0]))
first_en = list(map(lambda x: EN.vocab.itos[x],train_batch.en[:,0]))
first_de = list(map(lambda x: DE.vocab.itos[x],train_batch.de[:,0]))
print(first_en)
print(first_de)
#verify value is corrected
print('train.examples[0].en ==> ' + str(train.examples[0].en))
print('train.examples[0].de ==> ' + str(train.examples[0].de))
```

```python
[torchtext.data.batch.Batch of size 32]
	[.de]:[torch.LongTensor of size 17x32]
	[.en]:[torch.LongTensor of size 20x32]

train_batch.en ==> col-basd data format 
tensor([[  15, 129,   3,   3,  15,   3,   3,   3,   3,1960,   3, 119,   3,   3, 129,   3,   3,   3, 289,   3,   3,   3,   3,  15,   3,   3,   3,  41,4055,   3,   3, 289],
        [1467,  34,  52,   9,  34,   9,   9,7820,  13, 249,2422, 380,  24,   9,  49, 105,  52,   9,  18,  54,  55,  55,   9,  72,   9,  63,   9,  68,4349,  20,   9,  18],
        [1313,   4,  29,   4,  12,   4,   7,  29,  11,   6, 864, 184,  33,   4, 660,   4,  29, 989,  12,  10,1075,  98, 226, 124,  17,  67,   4,  54,5244,   9,  30,  36],
        [ 866, 319, 214,   2,  16,  46, 147, 114,   2,1510,  10,  17,   8,  2,   64,   2,   7,   6,  27, 203, 111,  10,   2,   6,   2,   4,  46, 645,4063,   4,  16,  11],
        [  12, 316,  62,  26,   5, 118,  16,   6,  55,   4, 767, 316,   2,1291,   4,  24,  27,   5,   4,  49,1602,  18, 970,   2,1567,   2, 148,   6,  47,   2,   2,   2],
        [  64,  12,   2,  25,2373,   2,   2,  40, 905,   5,  93,  45,2057,  46,   2,  97,   4, 188,   2, 109,   8,  95,   6,  68, 358,  26,  36,   2, 371,  24,15093,814],
        [  75,1342, 219,   7, 357, 156, 967, 517,   7, 207,  87,1145,  33,   8, 323,  11,  38,  14,1277,  61,   7,  64,  21,14008,  8, 191,  37,  28,8836,   8,  11,1158],
        [ 309,   2,13308, 30, 425,  23,3318,  23,  36,  10,   4,  12,  12,  82,   1, 140,  10, 436,  11,10481,255,   4, 601,   4,   2,   7,   5,1647,8293,  57,   2,  4],
        [1677, 739,   1,   6,   1,   5,   1,3737,  42,   5,14024, 87, 766, 231,   1,   7,   2,   2, 848, 155,   4,  38,   1,   5, 319,  30, 233,   6,   1,  86, 633,   5],
        [   1,5523,   1,   2,   1,  77,   1,5641,   2, 376,   1,  16,   1,   7,   1,5674,  55,  19,   1,  51,   2,  10,   1, 428,  69,   4,   1,   2,   1,   7,1007,  99],
        [   1,7737,   1, 703,   1,   9,   1,  37,1785,   1,   1,   5,   1, 502,   1,7325, 542,  33,   1, 223,7428,   5,   1,   1, 118,   6,   1, 683,   1,2475,   1,   1],
        [   1,   1,   1, 550,   1,2174,   1,   5,   1,   1,   1,  97,   1,   6,   1,14568,13567, 7,   1,1514,   1,2292,   1,   1,   2,   2,   1,   1,   1,  16,   1,   1],
        [   1,   1,   1,   2,   1,  21,   1,  66,   1,   1,   1,  10,   1,   2,   1,   6,   1,1368,   1, 104,   1,  14,   1,   1, 604, 246,   1,   1,   1, 130,   1,   1],
        [   1,   1,   1, 432,   1, 383,   1,   1,   1,   1,   1,   2,   1,  46,   1,   2,   1,14814,  1,1301,   1,   2,   1,   1,   4,1303,   1,   1,   1,   8,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,1520,   1,2827,   1,10227,  1,   1,   1, 962,   1,8963,   1,   1,   5,2375,   1,   1,   1, 397,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,1241,   1,   1,   1,   1,   1, 745,   1,   1, 279, 339,   1,   1,   1,   1,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  10,   1,   1,   1,   1,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1,   1,   1,   1,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  59,   1,   1,   1,   1,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,1648,   1,   1,   1,   1,   1,   1]])
        
trian_batch.en[:,0] ==> first column 
 tensor([  15,1467,1313, 866,  12,  64,  75, 309,1677,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1])        
        
['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche.', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']

train.examples[0].en ==> ['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.']
train.examples[0].de ==> ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche.']
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
## 情境2: TabularDataset/Iterator
```python
import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
#----------------------
DE.build_vocab(train)
EN.build_vocab(train)
train_iter = data.Iterator(dataset=train,batch_size=32,shuffle=False,train=True)
# 結果與 data.BucketIterator 一致
print(train_iter.__dict__.keys())
# content from next(iter(train_iter)) is embed-format such as train_batch.en[:,0] = tensor([15,1467,1313,...])
train_batch0 = next(iter(train_iter))

#---------------------- 以下都是資料驗證
# 去資料的方式也與 data.BucketIterator 一樣
print(train_batch0.en)
print(train_batch0.en[:,0])
print(list(map(lambda x: EN.vocab.itos[x], train_batch0.en[:,0])))
# content from next(train_iter.batch) is text-format such as train_batch[0].en = ['A,'man','in]
# 跟data.bucketIterator一樣，在next(iter(train_iter))後，train_iter會多一個batches變數，
# 內容是example，而且是尚未轉換成數字的example，可以直接看raw data 文字內容
train_batch1 = next(train_iter.batches)
```

```python
dict_keys(['_restored_from_state', 'random_shuffler', '_random_state_this_epoch', 'train', 'device', 'dataset', 'sort_key', 'repeat', 'sort_within_batch', '_iterations_this_epoch', 'batch_size_fn', 'iterations', 'sort', 'shuffle', 'batch_size'])
tensor([[  15, 129,   3,   3,  15,   3,   3,   3,   3,1960,   3, 119,   3,   3, 129,   3,   3,   3, 289,   3,   3,   3,   3,  15,   3,   3,   3,  41,4055,   3,   3, 289],
        [1467,  34,  52,   9,  34,   9,   9,7820,  13, 249,2422, 380,  24,   9,  49, 105,  52,   9,  18,  54,  55,  55,   9,  72,   9,  63,   9,  68,4349,  20,   9,  18],
        [1313,   4,  29,   4,  12,   4,   7,  29,  11,   6, 864, 184,  33,   4, 660,   4,  29, 989,  12,  10,1075,  98, 226, 124,  17,  67,   4,  54,5244,   9,  30,  36],
        [ 866, 319, 214,   2,  16,  46, 147, 114,   2,1510,  10,  17,   8,  2,   64,   2,   7,   6,  27, 203, 111,  10,   2,   6,   2,   4,  46, 645,4063,   4,  16,  11],
        [  12, 316,  62,  26,   5, 118,  16,   6,  55,   4, 767, 316,   2,1291,   4,  24,  27,   5,   4,  49,1602,  18, 970,   2,1567,   2, 148,   6,  47,   2,   2,   2],
        [  64,  12,   2,  25,2373,   2,   2,  40, 905,   5,  93,  45,2057,  46,   2,  97,   4, 188,   2, 109,   8,  95,   6,  68, 358,  26,  36,   2, 371,  24,15093,814],
        [  75,1342, 219,   7, 357, 156, 967, 517,   7, 207,  87,1145,  33,   8, 323,  11,  38,  14,1277,  61,   7,  64,  21,14008,  8, 191,  37,  28,8836,   8,  11,1158],
        [ 309,   2,13308, 30, 425,  23,3318,  23,  36,  10,   4,  12,  12,  82,   1, 140,  10, 436,  11,10481,255,   4, 601,   4,   2,   7,   5,1647,8293,  57,   2,  4],
        [1677, 739,   1,   6,   1,   5,   1,3737,  42,   5,14024, 87, 766, 231,   1,   7,   2,   2, 848, 155,   4,  38,   1,   5, 319,  30, 233,   6,   1,  86, 633,   5],
        [   1,5523,   1,   2,   1,  77,   1,5641,   2, 376,   1,  16,   1,   7,   1,5674,  55,  19,   1,  51,   2,  10,   1, 428,  69,   4,   1,   2,   1,   7,1007,  99],
        [   1,7737,   1, 703,   1,   9,   1,  37,1785,   1,   1,   5,   1, 502,   1,7325, 542,  33,   1, 223,7428,   5,   1,   1, 118,   6,   1, 683,   1,2475,   1,   1],
        [   1,   1,   1, 550,   1,2174,   1,   5,   1,   1,   1,  97,   1,   6,   1,14568,13567, 7,   1,1514,   1,2292,   1,   1,   2,   2,   1,   1,   1,  16,   1,   1],
        [   1,   1,   1,   2,   1,  21,   1,  66,   1,   1,   1,  10,   1,   2,   1,   6,   1,1368,   1, 104,   1,  14,   1,   1, 604, 246,   1,   1,   1, 130,   1,   1],
        [   1,   1,   1, 432,   1, 383,   1,   1,   1,   1,   1,   2,   1,  46,   1,   2,   1,14814,  1,1301,   1,   2,   1,   1,   4,1303,   1,   1,   1,   8,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,1520,   1,2827,   1,10227,  1,   1,   1, 962,   1,8963,   1,   1,   5,2375,   1,   1,   1, 397,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,1241,   1,   1,   1,   1,   1, 745,   1,   1, 279, 339,   1,   1,   1,   1,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  10,   1,   1,   1,   1,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1,   1,   1,   1,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  59,   1,   1,   1,   1,   1,   1],
        [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,1648,   1,   1,   1,   1,   1,   1]])

tensor([  15,1467,1313, 866,  12,  64,  75, 309,1677,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1])
['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)

## 情境3: Dataset/BucketIterator
  - data.Dataset 與 data.TabularDataset 最大的差異就是 data.Dataset 需要自己準備 Example, TabularDataset可以透過檔案(json/csv)直接取得
```python
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
train_bucketiter = data.BucketIterator(dataset=train_dataset,batch_size=32, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))
#---------------------- 以下都是資料驗證
print(train_batch)
print('train_batch.en ==> col-basd data format \n' + str(train_batch.en))
print(train_batch.en[:,0])
# data is saved as col-based rather than row-based, we should use [:,0] to retrive first sentence in first batch 
print('trian_batch.en[:,0] ==> first column \n ' + str(train_batch.en[:,0]))
# using vacaburaty to maping from index to string(label)
first_en = list(map(lambda x: EN.vocab.itos[x],train_batch.en[:,0]))
first_de = list(map(lambda x: DE.vocab.itos[x],train_batch.de[:,0]))
print(first_en)
print(first_de)
#verify value is corrected
print('train.examples[0].en ==> ' + str(train.examples[0].en))
print('train.examples[0].de ==> ' + str(train.examples[0].de))
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)

## 情境4: Dataset/Iterator
```python
import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}        # json 的 field 是dict 格式
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
train_examples = train.examples
#----------------------
fields = [('de',DE),('en',EN)]                  # example 的 field 是 list 格式
train_dataset = data.Dataset(examples=train_examples,fields=fields)
DE.build_vocab(train_dataset)
EN.build_vocab(train_dataset)
# 可以看到 BucketIterator 跟 Iterator的用法幾乎一樣
#train_bucketiter = data.BucketIterator(dataset=train,batch_size=32, sort_key=False, train=True, shuffle=False)
train_iter = data.Iterator(dataset=train_dataset,batch_size=32,shuffle=False,train=True)
# retrive frist batche
train_batch = next(iter(train_iter))
#---------------------- 以下都是資料驗證
print(train_batch)
print('train_batch.en ==> col-basd data format \n' + str(train_batch.en))
print(train_batch.en[:,0])
# data is saved as col-based rather than row-based, we should use [:,0] to retrive first sentence in first batch 
print('trian_batch.en[:,0] ==> first column \n ' + str(train_batch.en[:,0]))
# using vacaburaty to maping from index to string(label)
first_en = list(map(lambda x: EN.vocab.itos[x],train_batch.en[:,0]))
first_de = list(map(lambda x: DE.vocab.itos[x],train_batch.de[:,0]))
print(first_en)
print(first_de)
#verify value is corrected
print('train.examples[0].en ==> ' + str(train.examples[0].en))
print('train.examples[0].de ==> ' + str(train.examples[0].de))
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
# 情境比較
## 情境1 vs 情境2
 - 情境1: TabularDataset + BucketIterator
 - 情境2: TabularDataset + Iterator
 
只看使用方法的化，用法幾乎是完全一樣
因此使用BucketIterator或Iterator的時機就是根據官網的說明，
對資料順序不敏感的時候用BucketIterator如training data
對資料循序較敏感的使用用      Iterator如testing data

![](https://i.imgur.com/lBHPdwY.jpg)

[[top]](#torchtext-toturial---如何秒殺NLP資料集)

## 情境3 vs 情境4
 - 情境3: Dataset + BucketIterator
 - 情境4: Dataset + Iterator
與情境1/2的比較結論一樣，只要Dataset的取用模式確定了，BucketIterator與Iterator用法沒有多大差異

![](https://i.imgur.com/5A2mh3m.jpg)

[[top]](#torchtext-toturial---如何秒殺NLP資料集)
# 其他使用方式

## 手動建置 Example
前面的情境中，Example的來源都是用偷吃步，先用data.TabularDataset從格式化的內容中做出TabularDataset object，直接把TabularDataset object中的 Example取出來。
再把 Example object 灌入 data.Dataset(example='偷出來的example object')

如果資料流是在記憶體中，或者想從文字檔直入data.Dataset
可以使用 torchtext.data.Example所提供的方法來手動建置Example object
torchtext.data.Example提供的方法可以觀察 
```python
>>> data.Example.__dict__.keys()
dict_keys(['fromCSV', 'fromlist', 'fromtree', 'fromdict', 'fromJSON', '__dict__', '__doc__', '__weakref__', '__module__'])
```
可得到有以下五種方法
- data.Example.fromlist()  # 基本用法一
- data.Example.fromdict()  # 基本用法二
- data.Example.fromCSV()   # 並非從csv文件直入，而是合併fromlist/fromdict用法
- data.Example.fromJSON()  
- data.Example.fromtree()


假若我有csv與json兩種格式的training data
```
# train.csv 【註：此處的csv是以tab作為dilimilator而不是用逗號】
Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.	Two young, White males are outside near many bushes.
Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.	Several men in hard hats are operating a giant pulley system.
Ein kleines Mädchen klettert in ein Spielhaus aus Holz.	A little girl climbing into a wooden playhouse.
```

```
# train.json
{"DE": "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.", "EN": "Two young, White males are outside near many bushes."}
{"DE": "Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.", "EN": "Several men in hard hats are operating a giant pulley system."}
{"DE": "Ein kleines Mädchen klettert in ein Spielhaus aus Holz.", "EN": "A little girl climbing into a wooden playhouse."}
```

[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### Example from list

```python
import codecs
import subprocess
import torchtext.data as data

path_csv  = './multi30k_csv/train.csv'

# 讀入文字檔，並以list方式儲存每一行
def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = f.read().splitlines()
    return content
# 由於 csv 是以 tab 分開，表示每一個list元素的樣式是
# ['de string'\t'en string']
# 所以需要將每一個list元素以tab切開變成
# [['de string'],['en string']]
string_list = read_text(path_csv)
list_from_csv = list(map(lambda x: x.split('\t'), string_list))
# 下面就是 Example.fromlist的作法
# 重點是需要準備該list的data.Field格式
# 如果只有一筆資料則不需要使用 list(map(lambda x:))作法
# single = [['de string'],['en string']]
# data.Example.fromlist(single, fields) 即可
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = [('de',DE),('en',EN)] 
examples = list(map(lambda x:data.Example.fromlist(x,fields),list_from_csv))

print('type of manual examples ==> ' + str(type(examples)))
print('__dict__.keys() in examples element ==> ' + str(examples[0].__dict__.keys()))
print('len of manual examples ==> ' + str(len(examples)))
print('examples[0].de ==> ' + str(examples[0].de))
print('examples[0].en ==> ' + str(examples[0].en))
```
```python
type of manual examples ==> <class 'list'>
__dict__.keys() in examples element ==> dict_keys(['de', 'en'])
len of manual examples ==> 29000
examples[0].de ==> ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche.']
examples[0].en ==> ['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.']
```

```
examples
[<torchtext.data.example.Example object at 0x1266d2e48>,
 <torchtext.data.example.Example object at 0x1266d2e10>,
 <torchtext.data.example.Example object at 0x127f9e550>, ...]
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### Example from dict
這裏的情況有點特別，要分兩種情況討論
 1. 文字檔(JSON)是以 json array 格式儲存
 2. 文字檔(JSON)是以 dict list 格式儲存
```
# train_array.json  // JSON array
[
    {
        "DE": "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.", 
        "EN": "Two young, White males are outside near many bushes."
    },
    {
        "DE": "Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.", 
        "EN": "Several men in hard hats are operating a giant pulley system."
    },
    {
        "DE": "Ein kleines Mädchen klettert in ein Spielhaus aus Holz.", 
        "EN": "A little girl climbing into a wooden playhouse."
    }
]
```
```
# train.json  // dict list
{"DE": "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.", "EN": "Two young, White males are outside near many bushes."}
{"DE": "Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.", "EN": "Several men in hard hats are operating a giant pulley system."}
{"DE": "Ein kleines Mädchen klettert in ein Spielhaus aus Holz.", "EN": "A little girl climbing into a wooden playhouse."}
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
#### JSON array 處理方式

如果文字檔已經是 json array 格式，那就可以用下面的code處理
```python
# file is save as json array
import torchtext.data as data
import codecs
import json

path_json = './multi30k_json/train_array.json'
with codecs.open(path_json, encoding='utf-8') as fp:
    json_array = json.load(fp)

DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {'DE':('de',DE),'EN':('en',EN)} # 重點是這裏
examples = list(map(lambda x:data.Example.fromdict(x,fields),json_array))

print('type of manual examples ==> ' + str(type(examples)))
print('__dict__.keys() in examples element ==> ' + str(examples[0].__dict__.keys()))
print('len of manual examples ==> ' + str(len(examples)))
print('examples[0].de ==> ' + str(examples[0].de))
print('examples[0].en ==> ' + str(examples[0].en))

```

```
__dict__.keys() in examples element ==> dict_keys(['de', 'en'])
len of manual examples ==> 3
examples[0].de ==> ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche.']
examples[0].en ==> ['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.']
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
#### dict list 處理方式

如果資料儲蓄的方式並不是json array，而是dict list
那就得用取巧的方式，把每一行讀進來後把str轉換成dict
這個方式就是 ast.literal_eval
```python
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
contents = read_text(path_json)    # type(contents[0]) = str
dict_list = list(map(lambda x: ast.literal_eval(x),contents))     # type(dict_list[0]) =dict

DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {'DE':('de',DE),'EN':('en',EN)}
examples = list(map(lambda x:data.Example.fromdict(x,fields),dict_list))

print('type of manual examples ==> ' + str(type(examples)))
print('__dict__.keys() in examples element ==> ' + str(examples[0].__dict__.keys()))
print('len of manual examples ==> ' + str(len(examples)))
print('examples[0].de ==> ' + str(examples[0].de))
print('examples[0].en ==> ' + str(examples[0].en))
```
```python
__dict__.keys() in examples element ==> dict_keys(['de', 'en'])
len of manual examples ==> 29000
examples[0].de ==> ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche.']
examples[0].en ==> ['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.']
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### fromlist/fromdict 比較
從前面的code來看，當 JSON file的儲存方式是以 dict list 格式儲存時
Example.fromlist與Example.fromdict的對應方法是很類似的

```python
# common subroutine to read file as string list
path_csv  = './multi30k_csv/train.csv'
path_json = './multi30k_json/train.json'
def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = f.read().splitlines()
    return content
```

```python
# file is saved as CSV string list
string_list = read_text(path_csv)
list_from_csv = list(map(lambda x: x.split('\t'), string_list))

DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = [('de',DE),('en',EN)]             # 差異處
examples = list(map(lambda x:data.Example.fromlist(x,fields),list_from_csv))

```
```python
# file is saved as dict list 
contents = read_text(path_json)    # type(contents[0]) = str
dict_list = list(map(lambda x: ast.literal_eval(x),contents))   # type(dict_list[0]) = <class 'dict'>

DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {'DE':('de',DE),'EN':('en',EN)}    # 差異處
examples = list(map(lambda x:data.Example.fromdict(x,fields),dict_list))
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### 情境3 修改為人工建置 Example
```python
# 情境3寫法
# Example 來自 TaburlarDataset 抽出
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
train_bucketiter = data.BucketIterator(dataset=train_dataset,batch_size=32, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))
```

修改為
```python
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
#---------------------- 此處與情境3寫法無異
fields = [('de',DE),('en',EN)]              # example 的 field 是 list 格式 
train_dataset = data.Dataset(examples=train_examples,fields=fields)
DE.build_vocab(train_dataset)
EN.build_vocab(train_dataset)
train_bucketiter = data.BucketIterator(dataset=train_dataset,batch_size=32, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)

## 自動拆分資料集 split
情境1,2,3,4中不管是data.Dataset還是data.TabularDataset都是用匯入各自需要的資料
如
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
train_dataset = data.Dataset(examples=train_examples,fields=fields)
其實如果細看[官方文件](https://torchtext.readthedocs.io/en/latest/data.html)對data.dataset的支援的method說明會發現,data.Dataset還支援四種功能
  - downalod
  - filter_example
  - split    # 一份資料集分解成多個子集
  - splits   # 一次處理多份文字檔
   
```javascript=
class torchtext.data.Dataset(examples, fields, filter_pred=None)

    __init__(examples, fields, filter_pred=None)
        examples – List of Examples.
        fields (List(tuple(str, Field))) – The Fields to use in this tuple. The string is a field name, and the Field is the associated field.
        filter_pred (callable or None) – Use only examples for which filter_pred(example) is True, or use all examples if None. Default is None.
        
    classmethod download(root, check=None)
    filter_examples(field_names)
    split(split_ratio=0.7, stratified=False, strata_field='label', random_state=None)
    classmethod splits(path=None, root='.data', train=None, validation=None, test=None, **kwargs)
```
這裏提到的自動拆分資料集就是要用到split這個功能，當我們需要自動拆分資料集的時候通常有三種狀況
  - 三個文字檔（train.json/valid.json/test.json)各自讀取，就是情境1,2,3,4所使用的方法
  - 三個文字檔（train.json/valid.json/test.json)使用data.dataset一次讀取三個json檔
  - 一個文字檔 (dataset.json)使用data.dataset將一個json檔拆成三個 data.dataset object

```python
# 使用 data.TabularDataset.splits 實作方法二
# 三個文字檔（train.json/valid.json/test.json)使用data.dataset一次讀取三個json檔
import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
#train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
'''
data.TabularDataset.splits 的用法跟 data.TabularDataset的用法差不多，
path 從 file 改成 direction而已，再多帶三個檔案的檔名即可
'''
train,valid,test = data.TabularDataset.splits(path='./multi30k_json/',format='json',fields=fields,
                                                train='train.json',
                                                validation='valid.json',
                                                test='test.json')

print('len train.examples = ' + str(len(train.examples)))
print('len valid.examples = ' + str(len(valid.examples))) 
print('len test.examples  = ' + str(len(test.examples)))
```
```
len train.examples = 29000
len valid.examples = 1014
len test.examples  = 1000
```

```python
# 使用 data.TabularDataset.split 實作方法三
# 一個文字檔 (train.json)使用data.dataset將一個json檔拆成三個 data.TabularDataset object
# 正確的說應該是 one TabularDataset object is divided into three TabularDataset object
import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
print('len train.examples = ' + str(len(train.examples)))
train_s1, valid_s1, test_s1 = train.split(split_ratio=[0.8,0.1,0.1])
print('train_s1 length = '+str(len(train_s1.examples)))
print('valid_s1 length = '+str(len(valid_s1.examples)))
print('test_s1 length = '+str(len(test_s1.examples)))
```
```
len train.examples = 29000
train_s1 length = 23200
valid_s1 length = 2900
test_s1 length = 2900
```

```python
# 使用 data.Dataset.split 實作方法三 (用法與 data.TabularDataset.split一樣)
# 就可以把一份很長的 Example list 切成train/valid/test三份
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
```

```
len train.examples = 23200
len valid.examples = 2900
len test.examples  = 2900
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### 注意事項

不能用這個方式做cross validation資料切割
```python
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
tencross = train.split(split_ratio=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
```
會出現error message
```
Exception has occurred: AssertionError
Length of split ratio list should be 2 or 3, got [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
## padding 策略
單純的LSTM固定長度輸入可能就沒有padding問題，但是如果是Seq2Seq模型，就需要能適應語句不定長度。

我們常看到trochtext就是由 Dataset/Iterator/Field三個元素組成
當我們需要處理一個NLP前處理的問題時，我們需要思考的是，要把這個問題放在 Dataset/Iterator/Field的哪一個來處理，
或者，這句話應該顛倒寫，Dataset/Iterator/Field哪一個可以解決我們所遇到的問題。

回到padding本身，以padding這個問題來說，似乎是從Iterator來處理比較合理，batch的時候長度要一樣才一起做padding似乎比較自然。但是...
這樣的答案並不完整

從Torchtest提供的說明來說來看，我把跟Field官方的文件的參數部份稍微挪動位置，可以看到跟pad相關的處理似乎都是在Field中都做了定義，而data.Dataset/data.TabularDataset/data.Iterator/data.BucketIterator都沒有相關的pad設定，所以可知，torchtext都是透過data.Field來設定padding相關功能。
但畢竟只有定義，Field本身不執行任何操作，僅是設定而已。

```
torchtext.data.Field(pad_token=’<pad>’, unk_token=’<unk>’, pad_first=False,init_token=None, eos_token=None, sequential=True, use_vocab=True,  fix_length=None, dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False, tokenize=None, tokenizer_language=‘en’, include_lengths=False, batch_first=False,  truncate_first=False, stop_words=None, is_target=False)

    ~Field.init_token – A token that will be prepended to every example using this field, or None for no initial token. Default: None.
    ~Field.eos_token – A token that will be appended to every example using this field, or None for no end-of-sentence token. Default: None.
    ~Field.pad_token – The string token used as padding. Default: “<pad>”.
    ~Field.unk_token – The string token used to represent OOV words. Default: “<unk>”.
    ~Field.pad_first – Do the padding of the sequence at the beginning. Default: False.
```

padding的結果是在Dataset階段還是Iterator階段顯示呢？
答案是Field + Iterator
只能多得到一個結論，padding 跟 Dataset 沒有任何關係
在我們透過TabularDataset匯入JSON檔後(情境1)

```python
import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
```
我們觀察train也就是data.TabulaDataset的狀態
發現只有兩個成員裡面
```
>>> train.__dict__.keys()
dict_keys(['fields', 'examples'])
```
而 train.examples 是由 data.Example所組成的list
train.exmples[0]的成員只有當初Field設定的內容，en與de
train.example[0].en
train.example[0].de
內容都是原始的rawdata，train.example每個index內容長度也不一
因此可知在 Dataset 階段是尚未處理 padding 機制的
```
>>> type(train.examples)
<class 'list'>
>>> train.examples[0]
<torchtext.data.example.Example object at 0x105bbc518>

>>> train.examples[0].__dict__.keys()
dict_keys(['de', 'en'])

>>> train.examples[0].de
['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche.']
>>> train.examples[0].en
['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.']

>>> [len(train.examples[x].en) for x in range(10)]
[9, 11, 8, 14, 8, 14, 8, 13, 11, 10]
```


```
DE.build_vocab(train)
EN.build_vocab(train)
train_bucketiter = data.BucketIterator(dataset=train,batch_size=32, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))
```
到了 BucketIterator 階段我們可以看到
每個batch印出來的結果，每個batch都是32筆資料，但是不同batch長度就不一樣了

```python
train_bucket_iter = iter(train_bucketiter)
>>> next(train_bucket_iter)
[torchtext.data.batch.Batch of size 32]
        [.de]:[torch.LongTensor of size 17x32]
        [.en]:[torch.LongTensor of size 20x32]
>>> next(train_bucket_iter)
[torchtext.data.batch.Batch of size 32]
        [.de]:[torch.LongTensor of size 21x32]
        [.en]:[torch.LongTensor of size 20x32]
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### 每句話前後自動補\<sos\>\<eos\>


```python
# 由情境1修改
import torchtext.data as data
#只改 Field DE/EN 的設定，增加init_token,eos_token可以自訂內容
DE = data.Field(is_target=False,init_token='<sos>',eos_token='<eos>')
EN = data.Field(is_target=True,init_token='<sos>',eos_token='<eos>')
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)

DE.build_vocab(train)
EN.build_vocab(train)
# batch_size 改小一點 32->5
train_bucketiter = data.BucketIterator(dataset=train,batch_size=5, sort_key=False, train=True, shuffle=False)
train_batch = next(iter(train_bucketiter))
```

```python
>>> train_batch
[torchtext.data.batch.Batch of size 5]
        [.de]:[torch.LongTensor of size 16x5]
        [.en]:[torch.LongTensor of size 16x5]
>>> train_batch.en
tensor([[    2,     2,     2,     2,     2],
        [   17,   131,     5,     5,    17],
        [ 1469,    36,    54,    11,    36],
        [ 1315,     6,    31,     6,    14],
        [  868,   321,   216,     4,    18],
        [   14,   318,    64,    28,     7],
        [   66,    14,     4,    27,  2375],
        [   77,  1344,   221,     9,   359],
        [  311,     4, 13310,    32,   427],
        [ 1679,   741,     3,     8,     3],
        [    3,  5525,     1,     4,     1],
        [    1,  7739,     1,   705,     1],
        [    1,     3,     1,   552,     1],
        [    1,     1,     1,     4,     1],
        [    1,     1,     1,   434,     1],
        [    1,     1,     1,     3,     1]])
# 每一句話(column)的前後都增加了2,3
>>> list(map(lambda x: EN.vocab.itos[x], train_batch.en[:,0]))
['<sos>', 'Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.', '<eos>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
# 查表確認
>>> >>> [EN.vocab.itos[2], EN.vocab.itos[3]]
['<sos>', '<eos>']

```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
# Multi30 DE2EN 資料集

## introduction
### Multi30k by github 
  - git clone https://github.com/multi30k/dataset
 ``` 
 ./git_repository/multi30k$ tree -d
.
├── data
│   ├── task1
│   │   ├── image_splits
│   │   ├── raw
│   │   │   └── raw_data # 將raw gz解壓縮移至此 新建目錄
│   │   │   │   ├── test_2016_flickr.de     # 1,000
│   │   │   │   ├── test_2016_flickr.en     # 1,000
│   │   │   │   ├── train.de                #29,000
│   │   │   │   ├── train.en                #29,000
│   │   │   │   ├── val.de                  # 1,014
│   │   │   │   └── val.en                  # 1,014
│   │   │   ├── test_2016_flickr.cs.gz
│   │   │   ├── test_2016_flickr.de.gz    <-- this one
│   │   │   ├── test_2016_flickr.en.gz    <-- this one
│   │   │   ├── test_2016_flickr.fr.gz
│   │   │   ├── test_2017_flickr.de.gz
│   │   │   ├── test_2017_flickr.en.gz
│   │   │   ├── test_2017_flickr.fr.gz
│   │   │   ├── test_2017_mscoco.de.gz
│   │   │   ├── test_2017_mscoco.en.gz
│   │   │   ├── test_2017_mscoco.fr.gz
│   │   │   ├── test_2018_flickr.en.gz
│   │   │   ├── train.cs.gz
│   │   │   ├── train.de.gz    <-- this one
│   │   │   ├── train.en.gz    <-- this one 
│   │   │   ├── train.fr.gz
│   │   │   ├── val.cs.gz
│   │   │   ├── val.de.gz      <-- this one
│   │   │   ├── val.en.gz      <-- this one
│   │   │   └── val.fr.gz
│   │   └── tok
│   └── task2
│       ├── image_splits
│       ├── raw
│       └── tok
└── scripts
    ├── moses-3a0631a
    │   ├── share
    │   │   └── nonbreaking_prefixes
    │   └── tokenizer
    └── subword-nmt
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### sample
  - `train.de` (29,000 samples)
  ```
  Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.
    ...
Ein Mann in Shorts und Hawaiihemd lehnt sich über das Geländer eines Lotsenboots, mit Nebel und Bergen im Hintergrund. 
  ```
  - `train.en` (29,000 samples)
```
Two young, White males are outside near many bushes.
Several men in hard hats are operating a giant pulley system.
...
A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background.
```

## JSON 
 - train.json 29,000 [[download](https://reurl.cc/m9ZpyA)]
 - valid.json  1,014 [[download](https://reurl.cc/4mXlbY)]
 - test.json   1,000 [[download](https://reurl.cc/e8WpQR)]

### JSON sample 
\# json 格式對特殊字元較敏感，從raw轉換為json時移除 ‘ 與 ‘’，請參考multi30k2json.py中 escape  character handling前處理程式，此處並未處理所有json escape character，若要考慮所有json escape character 請google json escape character，用類似作法移除即可
```python
{"DE": "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.", "EN": "Two young, White males are outside near many bushes."}
{"DE": "Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.", "EN": "Several men in hard hats are operating a giant pulley system."}
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### multi30k raw to JSON 

`multi30k2json.py`
```python
import json
import codecs
import subprocess
import re

dataset_path = './git_repository/multi30k/data/task1/raw/raw_data/'
de_path_train = dataset_path + 'train.de'
de_path_val = dataset_path + 'val.de'
de_path_test = dataset_path + 'test_2016_flickr.de'
en_path_train = dataset_path + 'train.en'
en_path_val = dataset_path + 'val.en'
en_path_test = dataset_path + 'test_2016_flickr.en'

def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        #content = f.readlines()
        content = f.read().splitlines()
    #content = [x.strip() for x in content]
    # JSON escape character handling
    content = [re.sub(r'\"\"','\"',x) for x in content] # train_list[7365] bug: dobule quota in front-end
    content = [re.sub(r'\t','',x) for x in content]     # train_list[7365] bug: specail character \t
    content = [re.sub(r'\s+',' ',x) for x in content]   # waring 
    content = [re.sub(r'\"','\'',x) for x in content]   # double quota is special char, traing_list have 147 samples  
    return content

train_de = read_text(de_path_train)
train_en = read_text(en_path_train)
val_de = read_text(de_path_val)
val_en = read_text(en_path_val)
test_de = read_text(de_path_test)
test_en = read_text(en_path_test)

train_list = list(map(lambda x,y: '{"DE": "'+ x +'", "EN": "'+y +'"}',train_de, train_en))
valid_list = list(map(lambda x,y: '{"DE": "'+ x +'", "EN": "'+y +'"}',val_de, val_en))
test_list = list(map(lambda x,y: '{"DE": "'+ x +'", "EN": "'+y +'"}',test_de, test_en))

with codecs.open('./multi30k_json/train.json', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_list))

with codecs.open('./multi30k_json/valid.json', 'w', encoding='utf-8') as f:
    f.write('\n'.join(valid_list))

with codecs.open('./multi30k_json/test.json', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_list))

# json format check
err_count = 0
for i in range(len(train_list)):
    try:
        aa = json.loads(train_list[i])
    except:
        err_count += 1
        print(str(i) + ' error ==> \t' + train_list[i])
print('err_count = ' + str(err_count))
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
## CSV format
you can download Multi30k w/ csv format in the following link
  - train.csv 29,000 [[download](https://reurl.cc/EzXYgm)]
  - valid.csv  1,014 [[download](https://reurl.cc/EzXYV0)]
  - test.csv   1,000 [[download](https://reurl.cc/v1mGLk)]
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### CSV sample
```
Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.	Two young, White males are outside near many bushes.
Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.	Several men in hard hats are operating a giant pulley system.
Ein kleines Mädchen klettert in ein Spielhaus aus Holz.	A little girl climbing into a wooden playhouse.
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### multi30k raw to csv format 

`multi30k2csv.py`

```python
import json
import codecs
import subprocess

dataset_path = './git_repository/multi30k/data/task1/raw/raw_data/'
de_path_train = dataset_path + 'train.de'
de_path_val = dataset_path + 'val.de'
de_path_test = dataset_path + 'test_2016_flickr.de'
en_path_train = dataset_path + 'train.en'
en_path_val = dataset_path + 'val.en'
en_path_test = dataset_path + 'test_2016_flickr.en'

def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = f.read().splitlines()
    return content

def write_csv(filename, zip_deen):
    with open(filename, 'w', encoding='utf-8') as fp:
        output = '\n'.join('{}\t{}'.format(de,en) for de,en in zip_deen)
        fp.write(output)
    
train_de = read_text(de_path_train)
train_en = read_text(en_path_train)
valid_de = read_text(de_path_val)
valid_en = read_text(en_path_val)
test_de = read_text(de_path_test)
test_en = read_text(en_path_test)

write_csv('./multi30k_csv/train.csv', zip(train_de,train_en))
write_csv('./multi30k_csv/valid.csv', zip(valid_de,valid_en))
write_csv('./multi30k_csv/test.csv', zip(test_de,test_en))
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
# seq2seq 應用
一個deep learning軟體架構大致是以三個部分組成
1. dataset 處理 ===> `dataset.py` 或稱 `utils.py`
2. model 處理             ===> `model.py`
3. main code 喬接 1,2     ===> `train.py`

torchtext可以讓我們省去大量的時間去準備dataset
以一個 seq2seq 模型為例

![](https://i.imgur.com/KQKWCSx.jpg)

![](https://i.imgur.com/DvwriNY.jpg)

[[top]](#torchtext-toturial---如何秒殺NLP資料集)
## KEON seq2seq encode ([github](https://github.com/keon/seq2seq))
```python
from utils import load_dataset

def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256

    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size,macbook=True)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    
    encoder = Encoder(de_size, embed_size, hidden_size, n_layers=2, dropout=0.5) 
    decoder = Decoder(embed_size, hidden_size, en_size, n_layers=1, dropout=0.5)   
    seq2seq = Seq2Seq(encoder, decoder)
    
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)

    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter, en_size, args.grad_clip, DE, EN)
              
              
def train(e, model, optimizer, train_iter, vocab_size, grad_clip, DE, EN):
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        output = model(src, trg)
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size))

        encoder_output, hidden = self.encoder(src)
        
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size # 1693
        self.hidden_size = hidden_size # 512
        self.embed_size = embed_size # 256
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None): #src.shape=[29,32]
        embedded = self.embed(src) #[29,32,256]
        outputs, hidden = self.gru(embedded, hidden) #outputs.shape=[29,32,1024] hidden.shape=[4,32,512]
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:]) #[29,32,512]
        return outputs, hidden #[29,32,512], [4,32,512]
```
![](https://i.imgur.com/fYtt3Xa.jpg)

```
len(DE.vocab) = 1693
len(DE.vocab) = 3516
src = [29,32]
embedded = [23,32,256]

# 影像 output.shape/hidden.shape 因素
1. Encoder GRU n_layers
2. Encoder GRU Bidirectional

# case a: n_layers=2, Bidirectional=True
outputs.shape = [29,32,512]
hidden.shape = [4,32,512]

# case b: n_layers=1, Biderectional=True
outputs.shape = [29,32,512]
hidden.shape = [2,32,512]

# case c: n_layers=1, Biderectional=False
outputs.shape = [29,32,512]
hidden.shape = [1,32,512]

```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
## load_dataset by spacy
```python=
# utils.py 
import re
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k


def load_dataset(batch_size, macbook=False):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]
    
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
## load_dataset by json
```python=
def load_dataset_txt(batch_size,macbook=False):
    DE = Field(include_lengths=True,
                init_token='<sos>', eos_token='<eos>')
    EN = Field(include_lengths=True,
                init_token='<sos>', eos_token='<eos>')
    fields = {"DE":('src',DE),"EN":('trg',EN)}
    train, val, test = TabularDataset.splits(path='./multi30k_json/',format='json',fields=fields,
                                                train='train.json',validation='valid.json',test='test.json')
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]

    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
## 假如不用torchtext

from Global-Encoding ([github](https://github.com/lancopku/Global-Encoding)), data loading need additional two files
1. utils.dict_helper.py (200 lines,14 subroutines)
2. utils.data_helper.py (150 lines, 2 class w/ additonal 4 subroutines)
```python=
# dict_helper.py
import torch
from collections import OrderedDict

PAD_WORD = '<blank>'
UNK_WORD = '<unk> '
BOS_WORD = '<s>'
EOS_WORD = '</s>'

class Dict(object):
    def __init__(self, data=None, lower=True):
    def size(self):
    def loadFile(self, filename):
    def writeFile(self, filename):
    def loadDict(self, idxToLabel):
    def lookup(self, key, default=None):
    def getLabel(self, idx, default=None):
    def addSpecial(self, label, idx=None):
    def addSpecials(self, labels):
    def add(self, label, idx=None):
    def prune(self, size):
    def convertToIdxandOOVs(self, labels, unkWord, bosWord=None, eosWord=None):
    def convertToIdxwithOOVs(self, labels, unkWord, bosWord=None, eosWord=None, oovs=None):
    def convertToLabels(self, idx, stop, oovs=None):
```

```python
# data_helper.py
import torch.utils.data as torch_data

class MonoDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None):
    def __getitem__(self, index):
    def __len__(self):

class BiDataset(torch_data.Dataset):
    def __init__(self, infos, indexes=None, char=False):
    def __getitem__(self, index):
    def __len__(self):

def splitDataset(data_set, sizes):
def padding(data):
def ae_padding(data):
def split_padding(data):
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
# 進階用法

## 中文處理/斷詞
以LCSTS data 為例，LCSTS是一個摘要資料集
摘要的作法與翻譯問題在模型的處理上是一樣的，
我們可以想像，翻譯是將A語言轉變成B語言
摘要也可以是同樣的邏輯，長句是A語言，短句是B語言

因此處理Multi30k的模型Seq2Seq來說，模型端完全不用改變
我們只要讓LCSTS的資料格式與Multi30k一樣就好
因此我們唯一需要改動的是 utils.py (或稱dataset.py)即可


標準的LCSTS資料有三個檔案，並且兩種格式
```
.
├── PART_I.txt        # 2,400,591筆 (1GB)
├── PART_II.txt       #    10,666筆 (5.1MB)
├── PART_III.txt      #     1,065筆 (536KB)
```

PART_I.txt （無human_label欄位)
```xml
doc id=0>
    <summary>
        修改后的立法法全文公布
    </summary>
    <short_text>
        新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。
    </short_text>
</doc>
<doc id=1>
    <summary>
        深圳机场9死24伤续：司机全责赔偿或超千万
    </summary>
    <short_text>
        一辆小轿车，一名女司机，竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元。
    </short_text>
</doc>
```
PART_II.txt/PART_III.txt 包含human_label欄位，其餘與PART_I一致
```xml
<doc id=0>
    <human_label>5</human_label>
    <summary>
        可穿戴技术十大设计原则
    </summary>
    <short_text>
        本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代
人
    </short_text>
</doc>
<doc id=1>
    <human_label>5</human_label>
    <summary>
        经济学人：智能手机将成为“真正的个人电脑”
    </summary>
    <short_text>
        2007年乔布斯向人们展示iPhone并宣称“它将会改变世界”，还有人认为他在夸大其词，然而在8年后，以iPhone为代表的触屏智能手机已经席卷全球各个角落。未来，智能手机将会成为“真正的个人电脑”，为人类发展做出更>大的贡献。
    </short_text>
</doc>
```

原先我們處理 Multi30k 降至轉換成 batch_iter 的程式碼為
```python=
# multi30k batch dataloader

def load_dataset_txt(batch_size,macbook=False):
    DE = Field(include_lengths=True,
                init_token='<sos>', eos_token='<eos>')
    EN = Field(include_lengths=True,
                init_token='<sos>', eos_token='<eos>')
    fields = {"DE":('src',DE),"EN":('trg',EN)}
    train, val, test = TabularDataset.splits(path='./multi30k_json/',format='json',fields=fields,
                                                train='train.json',validation='valid.json',test='test.json')
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]

    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN
```

為了處理 LCSTS 資料，我們要以下面兩個函式來處理
```python=
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

def load_dataset_lcsts(batch_size,macbook=False,filename=None):
    filename = './lcsts_xml/PART_I_10000.txt' if filename == None else filename
    TRG = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')
    SRC = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')
    fields = [('trg',TRG),('src',SRC)] 
    lcsts_list = get_xmllcsts(filename)

    examples = list(map(lambda x :Example.fromlist(x,fields),lcsts_list))
    all_data = Dataset(examples=examples,fields=fields) 
    train, val, test = all_data.split(split_ratio=[0.8,0.1,0.1]) 
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]

    SRC.build_vocab(train.src, min_freq=2)
    TRG.build_vocab(train.trg, max_size=10000)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False, shuffle=False)
    return train_iter, val_iter, test_iter, SRC, TRG
```
after above code, we will get the following resut

```python
# 因為使用部份 PART_I.txt 的資料，故資料筆數只有1000筆
>>> len(lcsts_list)
10000

# PART_I 沒有人工標記分數的欄位資料 human_label
>>> lcsts_list[0]
('修改后的立法法全文公布', '新华社受权于18日全文播发修改后的《中华...6章，共计105条。')

>>> len(examples)
10000

>>> examples[0].__dict__.keys()
dict_keys(['trg', 'src'])

# 經過 Dataset.split(split_ratio=[0.8,0.1,0.1])後
>>> len(train.examples)
8000
>>> len(val.examples)
1000
>>> len(testing.examples)
1000

# example 內容為
>>> train.examples[0].src
['假日办提3种假期调休方案供网友投票，截至27日0时40分，凤凰网的投票结果显示，支持率最高的是C方案，该方案提出继续保持国庆黄金周七天长假；即全年11天假期总量未变，马年春节可能从2014年1月30日到2月5日起休7天。]
>>> train.examples[0].trg
['假日办提3种假期调休方案超半网友支持保留国庆7天假']
```

相較Multi30k的 example 內容，可以知道中文需要以char based或者jieba斷詞來產生 char-array 或者 word-array，而不是像上面一樣只有一整句話
```python
>>> train.examples[0].de
['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche.']
>>> train.examples[0].en
['Two', 'young,', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes.']

```

修改 Field SRC與TRG的定義

```python
TRG = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')
SRC = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')
```

改為

```python
TRG = Field(tokenize=list, include_lengths=True, init_token='<sos>', eos_token='<eos>')
SRC = Field(tokenize=list, include_lengths=True, init_token='<sos>', eos_token='<eos>')
```
可得
```python
>>> train.examples[0].src
['大', '数', '据', '：', '针', '对', '增', '量', '中', '海', '量', '的', '结', '构', ...]
>>> train.examples[0].trg
['H', 'a', 'd', 'o', 'o', 'p', '一', '般', '用', '在', '哪', '些', '业', '务', ...]
```

若要以詞做單位，則需要以jieba當作tokenzie
```python
def jieba_tokenizer(text): # create a tokenizer function
    #return [tok.text for tok in spacy_en.tokenizer(text)]
    return [tok for tok in jieba.lcut(text)]
```
並修改Field定義為
```python
TRG = Field(tokenize=jieba_tokenizer,include_lengths=True, init_token='<sos>', eos_token='<eos>')
SRC = Field(tokenize=jieba_tokenizer,include_lengths=True, init_token='<sos>', eos_token='<eos>')
```
example的結果就會變成以詞做單位

```python
>>> train.examples[0].src
['金逸', '影视', '在', 'IPO', '招股', '说明书', '（', '申报', '稿', '）', '中', '武汉', '国资委', '看', ...]
>>> train.examples[0].src
['武汉', '国资委', '花', '百万', '看', '电影', '？', '金逸', '影视', '称', '理解', '有误']
```
最後，在utils.py中完整的處理LCSTS data的函式有三段
```python=
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
    #return [tok.text for tok in spacy_en.tokenizer(text)]
    return [tok for tok in jieba.lcut(text)]

def load_dataset_lcsts(batch_size,macbook=False,filename=None):
    filename = './lcsts_xml/PART_I_10000.txt' if filename == None else filename
    TRG = Field(tokenize=jieba_tokenizer, include_lengths=True,
                    init_token='<sos>', eos_token='<eos>')
    SRC = Field(tokenize=jieba_tokenizer, include_lengths=True,
                    init_token='<sos>', eos_token='<eos>')
    fields = [('trg',TRG),('src',SRC)] 
    lcsts_list = get_xmllcsts(filename)

    examples = list(map(lambda x :Example.fromlist(x,fields),lcsts_list))
    all_data = Dataset(examples=examples,fields=fields) 
    train,val, test = all_data.split(split_ratio=[0.8,0.1,0.1]) 
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]

    SRC.build_vocab(train.src, min_freq=2)
    TRG.build_vocab(train.trg, max_size=10000)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False, shuffle=False)
    return train_iter, val_iter, test_iter, SRC, TRG
```

原先處理Multi30k的 train.py 可以幾乎不用修改
只要在train.py中，修改資料集匯入的地方即可
Field SRC/TRG 在retrun後也會被DE/EN取代，對模型完全無影響

```python
from utils import load_dataset,load_dataset_txt,load_dataset_lcsts

#train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size,macbook=True)
#train_iter, val_iter, test_iter, DE, EN = load_dataset_txt(args.batch_size,macbook=True)
filename =  './lcsts_xml/PART_I_10000.txt'
train_iter, val_iter, test_iter, DE, EN = load_dataset_lcsts(args.batch_size,filename=filename)
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
## preprocessing 與 postprocessing


首先看看使用方式
```
[docs]    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """


[docs]    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.
        Postprocess the batch with user-provided Pipeline.                
```

接著看看[定義](https://pytorch.org/text/data.html#fields)
```
~Field.preprocessing – The Pipeline that will be applied to examples using this field after tokenizing but before numericalizing. Many Datasets replace this attribute with a custom preprocessor. Default: None.
~Field.postprocessing – A Pipeline that will be applied to examples using this field after numericalizing but before the numbers are turned into a Tensor. The pipeline function takes the batch as a list, and the field’s Vocab. Default: None.
```

簡單的來說要點有二
1. preprocessing 是以 example 為單位，postprocessing 是以 example-list 為單位
2. preprocessing 是Dataset階段處理，而且是在tokennize 之後，
postprocessing是在iterator階段處理，而且是在數值化之後，tensor化之前

用架構圖來看會更清楚 ref [link](https://kknews.cc/code/5r4g2l8.html)

![](https://i.imgur.com/Frc9VKy.jpg)
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### preprocessing

下面是一個 Field 調用 preprocess 的範例，使用LCSTS資料集的時候都會有一個動作是簡轉繁，有時我們是在file階段直接做好簡轉繁，程式開始執行是就直接讀繁體的目錄，但這裡是透過 Field的 preprocess 方式，將每一個 exmaple 都做簡轉繁都處理，在做後續的 batch 處理。

為了做差異比較，下面的程式是並沒有將LCSTS全部都做簡轉繁，而是只將 example內容的 SRC Field 變成繁體中文，而 TRG Field 因為不調動 preprocessing 所以還是維持簡體。

從 simple2trad(x) 這個函式來看，輸入是個 list，輸出也是 list
也就是說，simple2trad的input其實是 tokenize 後的結果(list)
``` Field 的 tokenize 先於 preprocess```
這樣會造成執行的效率非常的緩慢，jieba斷詞後的list元素每個都執行OpenCC.s2t 跟整句話執行一次 OpenCC.s2t 效率可以差異20倍以上。

如果真的要做線上的簡繁互換，應該坐在 tokenize 階段，可以維持效率
此處是為了做 preprocess 的測試，因此特意將簡轉繁做在 preprocess

```
    50 個 LCSTS example
    在 preporcess 執行 OpenCC('s2t') 耗時 236 秒
    在 toeknize 執行 OpenCC('s2t') 耗時 9 秒
    差異 236/9 = 26 倍
```

```python
from torchtext.data import Field,Example
import re
import jieba
import codecs
import subprocess
from opencc import OpenCC

# 使用 OpenCC做簡繁互換的副程式
def simple2trad(x):
    return list(map(lambda y: OpenCC('s2t').convert(y),x))
```

```python

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
    #return [tok.text for tok in spacy_en.tokenizer(text)]
    return [tok for tok in jieba.lcut(text)]


filename = './lcsts_xml/PART_I_10000.txt' 
TRG = Field( tokenize=jieba_tokenizer,include_lengths=True,
                init_token='<sos>', eos_token='<eos>')
SRC = Field(tokenize=jieba_tokenizer, include_lengths=True,
                init_token='<sos>', eos_token='<eos>',
                preprocessing=simple2trad)  #<-- 只修改這
fields = [('trg',TRG),('src',SRC)] 
lcsts_list = get_xmllcsts(filename,limit=50)

examples = list(map(lambda x :Example.fromlist(x,fields),lcsts_list))
print(len(examples))
```

```
### 繁體
examples[0].src 
['新華社', '受權', '於', '18', '日', '全文', '播發', '修改', '後', '的', '《', '中華人民共和國', '立法法', '》', ...]

### 簡體
examples[0].trg 
['修改', '后', '的', '立法法', '全文', '公布']
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### 使用 data.Pipeline 加速 preprocessing
上面將 SRC 做簡轉繁的核心程式為，preprocessing帶入一個subroutine，
subroutine 的 input 是一個 list (tokenzie 後的結果)，
subroutine 的內容就是自己寫一段程式去處理這個 list 當作自己想要的前處理

如果這個 list 內容的處理方式都是一樣的，那可以再調用 torchtext.data.Pipeline來加速處理 Field(preprocessing=xxx)，

使用方式是，宣告Field的preprocessing的函式xxx要用Pipeline處理
語法大致是
```python
    Field(preprocessing=Pipeline(xxx))
```
這裏的xxx就不是以list為考量的的函式了，而是list裡面的元素
每個list裡面的元素都要經過xxx來處理

下面是較完整的範例

```
# 不使用Pipeline的情況下，OpenCC做簡繁互換的副程式

def simple2trad(x):
    return list(map(lambda y: OpenCC('s2t').convert(y),x))
    
SRC = Field(tokenize=jieba_tokenizer, include_lengths=True,
                init_token='<sos>', eos_token='<eos>',
                preprocessing=simple2trad)  #<-- 只修改這
```
若想要做平行處理可利用 data.Pipeline 來做加速
上面的核心 code 可以改成

```python
from torchtext.data import Pipeline

simple2trad_pipe = Pipeline(lambda x: OpenCC('s2t').convert(x))

SRC = Field(tokenize=jieba_tokenizer, include_lengths=True,
                init_token='<sos>', eos_token='<eos>',
                preprocessing=simple2trad_pipe)  #<-- 只修改這
```
但是也許是因為我用Notebook核心數不夠，這樣做並沒有加快，反而比一般的 preprocessing for loop 處理還要慢

```
    50 個 LCSTS example
    在 preporcess 執行 OpenCC('s2t') 耗時 236 秒
    在 toeknize 執行 OpenCC('s2t') 耗時 9 秒
    差異 236/9 = 26 倍
    
    在 preprocess 使用 Pipeline(OpenCC) 耗時 263 秒
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### postprocessing

直接給範例，以Multi30k DE-EN的情況來說
如果我把每個 example 的前三個字都隨即替換掉，用 post-processing方式處理，修改

```python
# post-processing 要處理的fnction
def custom_batch(batch, vocab):
    r = random.randint(4,len(vocab))
    # overwrite index 2,3,4 as random word 'r=46'
    # such as [2, 46, 46, 46, 30, 305, 17, 82, 6, 12, 52, 0, 0, 3, 1, 1, 1, 1, 1]

    return list(map(lambda example: [r if i>0 and i<4 else x for i,x in enumerate(example)] , batch))

def load_dataset_txt(batch_size,macbook=False):
    DE = Field(include_lengths=True,init_token='<sos>', eos_token='<eos>', 
                postprocessing=custom_batch) # <-- 修改此處
    EN = Field(include_lengths=True,init_token='<sos>', eos_token='<eos>', 
                postprocessing=custom_batch) # <-- 修改此處
    fields = {"DE":('src',DE),"EN":('trg',EN)}
    train, val, test = TabularDataset.splits(path='./multi30k_json/',format='json',fields=fields,
                                                train='train.json',validation='valid.json',test='test.json')
                                                
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=True, shuffle=False)
    batch = next(iter(train_iter))
    return train_iter, val_iter, test_iter, DE, EN

```

```
batch = next(iter(train_iter))
batch.src[0][:,0]
tensor([  2, 223, 223, 223,  32, 239,  17,  62,   6,  13,  53,   0,   0,   3,
          1,   1,   1,   1,   1])
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
### 使用 data.Pipeline 加速 postprocessing


## 使用 Word2Vector (W2V)
ref: https://www.twblogs.net/a/5d66689fbd9eee5327fea549

```python
from torchtext.vocab import GloVe
from torchtext import data
TEXT = data.Field(sequential=True)

TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
TEXT.build_vocab(train, vectors="glove.6B.300d")
```

torchtext支持的詞向量 ref: [torchtext source](https://pytorch.org/text/_modules/torchtext/vocab.html#GloVe)：

* charngram.100d ` url = ('http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/'
           'jmt_pre-trained_embeddings.tar.gz')`
* fasttext.en.300d // `url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec`
* fasttext.simple.300d
* glove.42B.300d
* glove.840B.300d
* glove.twitter.27B.25d
* glove.twitter.27B.50d
* glove.twitter.27B.100d
* glove.twitter.27B.200d
* glove.6B.50d
* glove.6B.100d
* glove.6B.200d
* glove.6B.300d <--- we assign this in above example

借接模型

```python
embedding = nn.Embedding(2000, 300)
weight_matrix = TEXT.vocab.vectors
embedding.weight.data.copy_(weight_matrix)
```
[[top]](#torchtext-toturial---如何秒殺NLP資料集)
