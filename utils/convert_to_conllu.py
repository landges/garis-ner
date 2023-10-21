import pandas as pd
import re

df = pd.read_csv('dns.csv', sep=';')
df.columns = ['title', 'text']
ents, dsc = df['title'].to_list(), df['text'].to_list()
i = 0
clear_ents, clear_sents = [], []
for ent, desc in zip(ents, dsc):
    sents = desc.split('.')
    ent = ' '.join(ent.split(' ')[1:])
    for sent in sents:
        if ent not in sent:
            continue
        tokens = sent.split(' ')
        while '' in tokens:
            tokens.remove('')
        emb = []
        repeat = False
        for token in tokens:
            if token in ent and repeat is False:
                emb.append('B-ELEC')
                repeat = True
                continue
            if token in ent and repeat is True:
                emb.append('I-ELEC')
                continue
            if token not in ent:
                emb.append('O')
                continue
        print(emb)
        print(sent)
        clear_ents.append(emb)
        clear_sents.append(tokens)
        assert len(emb) == len(tokens)

print(len(clear_ents))
i = 0
# for sent,ents in zip(clear_sents,clear_ents):
#     if i<(0.8*len(clear_ents)):
#         path='data_electronic/train.txt'
#     elif i>(0.8*len(clear_ents)) and i<(0.9*len(clear_ents)):
#         path='data_electronic/test.txt'
#     else:
#         path = 'data_electronic/val.txt'
#     with open(path,'a+',encoding='utf-8') as fout:
#         for index,(token, ent) in enumerate(zip(sent,ents)):
#             fout.write(str(index)+'\t'+token+'\t'+ent)
#             fout.write('\n')
#         fout.write('\n')
#     i+=1

for sent, ents in zip(clear_sents, clear_ents):
    path = 'data_electronic/val_el.txt'
    with open(path, 'a+', encoding='utf-8') as fout:
        for index, (token, ent) in enumerate(zip(sent, ents)):
            fout.write(str(index) + '\t' + token + '\t' + ent)
            fout.write('\n')
        fout.write('\n')
    i += 1

a = 'a, c'
b = a.split(',')
print(b)
