import requests
from urllib.parse import urlencode
import zipfile
import googletrans
from googletrans import Translator
import pandas as pd
import numpy as np
import ast
import json


def uload_from_yadisk(href, path, zip=True):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = href
    # Сюда вписываете вашу ссылку
    # Получаем загрузочную ссылку
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
    # Загружаем файл и сохраняем его
    download_response = requests.get(download_url)
    with open(path, 'wb') as f:  # Здесь укажите нужный путь к файлу
        f.write(download_response.content)
    if zip is True:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall()


def translate(text):
    translator = Translator()
    result = translator.translate(text, dest='ru')
    ru_result = result.text
    return ru_result


def read_el_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        sentences = text.split('\n\n')
        data = []
        for sent in sentences:
            tokens = sent.split('\n')
            sentence = []
            for token in tokens:
                tollab = tuple(token.split('\t'))
                if tollab[0] is '' or tollab[1] is '':
                    continue
                sentence.append(tollab)
            if len(sentence) == 0:
                continue
            else:
                data.append(sentence)
    return data


def read_conllu(path):
    df = pd.read_table(path, delimiter='\t', engine='python', encoding='utf-8', error_bad_lines=False)
    df.columns = ['index', 'token', 'tag']
    sents_df = []
    last_index = 0
    for index, row in df.iterrows():
        if row['index'] == 0:
            sents_df.append(df[last_index:index])
            last_index = index
    return sents_df


class Logger(object):
    def __init__(self, fn='', tofile=False):
        self.fn = fn
        self.tofile = tofile
        return

    def printml(self, *args):
        toprint = ''
        for v in args:
            toprint = toprint + str(v) + ' '
        if self.tofile:
            f = open(self.fn, 'a')
            f.write(toprint + "\n")
            f.close()
        else:
            print(toprint)
        return


cnv = {
    'O': 'O',
    'I-PERSON': 'I-PER',
    'B-PERSON': 'B-PER',
    'B-NORP': 'O',
    'I-NORP': 'O',
    'B-FAC': 'O',
    'I-FAC': 'O',
    'B-ORG': 'B-ORG',
    'I-ORG': 'I-ORG',
    'B-GPE': 'B-LOC',
    'I-GPE': 'I-LOC',
    'B-LOC': 'B-LOC',
    'I-LOC': 'I-LOC',
    'B-PRODUCT': 'O',
    'I-PRODUCT': 'O',
    'B-EVENT': 'O',
    'I-EVENT': 'O',
    'B-WORK_OF_ART': 'O',
    'I-WORK_OF_ART': 'O',
    'B-LAW': 'O',
    'I-LAW': 'O',
    'B-LANGUAGE': 'O',
    'I-LANGUAGE': 'O',
    'B-DATE': 'O',
    'I-DATE': 'O',
    'B-TIME': 'O',
    'I-TIME': 'O',
    'B-PERCENT': 'O',
    'I-PERCENT': 'O',
    'B-MONEY': 'O',
    'I-MONEY': 'O',
    'B-QUANTITY': 'O',
    'I-QUANTITY': 'O',
    'B-ORDINAL': 'O',
    'I-ORDINAL': 'O',
    'B-CARDINAL': 'O',
    'I-CARDINAL': 'O',
}

cnv2 = {
    'O': 'O',
    'I-PERSON': 'I-ELEC',
    'B-PERSON': 'B-ELEC',
    'B-NORP': 'B-ELEC',
    'I-NORP': 'I-ELEC',
    'B-FAC': 'B-ELEC',
    'I-FAC': 'I-ELEC',
    'B-ORG': 'B-ELEC',
    'I-ORG': 'I-ELEC',
    'B-GPE': 'B-ELEC',
    'I-GPE': 'I-ELEC',
    'B-LOC': 'B-ELEC',
    'I-LOC': 'I-ELEC',
    'B-PRODUCT': 'B-ELEC',
    'I-PRODUCT': 'I-ELEC',
    'B-EVENT': 'B-ELEC',
    'I-EVENT': 'I-ELEC',
    'B-WORK_OF_ART': 'B-ELEC',
    'I-WORK_OF_ART': 'I-ELEC',
    'B-LAW': 'B-ELEC',
    'I-LAW': 'I-ELEC',
    'B-LANGUAGE': 'B-ELEC',
    'I-LANGUAGE': 'I-ELEC',
    'B-DATE': 'B-ELEC',
    'I-DATE': 'I-ELEC',
    'B-TIME': 'B-ELEC',
    'I-TIME': 'I-ELEC',
    'B-PERCENT': 'B-ELEC',
    'I-PERCENT': 'I-ELEC',
    'B-MONEY': 'B-ELEC',
    'I-MONEY': 'I-ELEC',
    'B-QUANTITY': 'B-ELEC',
    'I-QUANTITY': 'I-ELEC',
    'B-ORDINAL': 'B-ELEC',
    'I-ORDINAL': 'I-ELEC',
    'B-CARDINAL': 'B-ELEC',
    'I-CARDINAL': 'I-ELEC'
}


def conv_from_deeppavlov(tokens):
    return_tokens = []
    print(tokens)
    for token in tokens:
        return_tokens.append(cnv[token])
    return return_tokens


def read_api_keys(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
        dict = ast.literal_eval(text)
    return dict


def write_api_keys(dict, filepath):
    str = json.dumps(dict)
    with open(filepath, 'w') as f:
        f.write(str)
