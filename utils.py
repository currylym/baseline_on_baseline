'''

'''

import codecs
import json
import jieba
jieba.load_userdict('models/nerDict.txt')

global nerDict

# 把txt转json
def transform_data(path1,path2):
    data = []
    with open(path1) as f:
        for line in f:
            line = json.loads(line)
            data.append(line)
    with open(path2,'w') as writer:
        writer.write(json.dumps(data,indent=1,ensure_ascii=False))

# 分词---未去除停用词/未加额外字典
def word_segment(sentence):
    sentence = sentence.strip().replace(' ','')
    return list(jieba.cut(sentence))

# 读取ner字典
def loadNerDict(path='models/nerDict.txt'):
    nerDictFile = codecs.open(path,'r','utf-8')
    nerDict = set()
    for line in nerDictFile:
        nerDict.add(line.strip())
    return nerDict

def train_ws(out,path='data/coreEntityEmotion_train.json'):
    data = json.loads(open(path).read())
    count = 0
    for item in data:
        count += 1
        title_ws = word_segment(item['title'])
        content_ws = word_segment(item['content'])
        ner = [i for i in title_ws+content_ws if i in nerDict]
        item.update({'title_ws':title_ws,
                     'content_ws':content_ws,
                     'ner':ner})
        if count % 1000 == 0:
            print('processed:%d,all:%d' % (count,len(data)))
    with open(out,'w') as writer:
        writer.write(json.dumps(data,indent=1,ensure_ascii=False))

if __name__ == '__main__':
    #nerDict = loadNerDict()
    #train_ws(out='data/coreEntityEmotion_train_ws.json')
    #transform_data('data/coreEntityEmotion_train.txt','data/coreEntityEmotion_train.json')
    transform_data('data/coreEntityEmotion_example.txt','data/coreEntityEmotion_example.json')
