from bs4 import BeautifulSoup
import requests
from scipy.spatial import distance
from itertools import groupby, combinations
import re
import os
from tqdm import tqdm
from utils.timer import Timer

try:
    from googlesearch import search
except ImportError:
    print("No module named 'google' found")
from sentence_transformers import SentenceTransformer, util
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
from utils.utils import Logger, read_api_keys, write_api_keys
from gensim.models import Doc2Vec
import gensim
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import tqdm as tq
from serpapi import GoogleSearch
from langdetect import detect, DetectorFactory

# module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
# use_model = hub.load(module_url)
# model_sb = SentenceTransformer('distiluse-base-multilingual-cased')
# model_d2v = Doc2Vec.load('doc2vec_model')
#

def use_get_embedding(text):
    return use_model([text]).numpy()[0]


def use_records_to_embeds(texts):
    embeddings = np.zeros((len(texts), 512))
    for i, record in tq.tqdm(enumerate(texts)):
        embeddings[i] = use_get_embedding(record)
    return embeddings


def get_d2v_embeddings(text):
    embeds = []
    for t in text:
        embeds.append(model_d2v.infer_vector(gensim.utils.simple_preprocess(t)))
    return embeds


def get_bert_embeddings(text):
    embedding = model_sb.encode(text)
    return embedding


def get_html(url):
    return requests.get(url, timeout=10).text


def search_links(query, keywords, language='ru'):
    # to search
    links = []
    headers = 'Mozilla/5.0'
    for i in range(len(keywords) + 1):
        time.sleep(10)
        for pair in combinations(keywords, i):
            time.sleep(2)
            comb_query = query + ' '.join(pair)
            try:
                search_urls = search(comb_query, tld="com", num=20, stop=20, pause=2)
                for url in search_urls:
                    links.append(url)
            except:
                return list(set(links))
    links = list(set(links))
    return links


def serpapi_search(query, keywords):
    links = []
    api_keys_dict = read_api_keys('api_keys.txt')
    api_keys = api_keys_dict.keys()
    for key in api_keys:
        if api_keys_dict[key] < 100:
            for i in range(len(keywords) + 1):
                for pair in combinations(keywords, i):
                    comb_query = query + ' '.join(pair)
                    client = GoogleSearch(
                        {"q": comb_query,
                         "api_key": key}  # API Key from SERP
                    )
                    result = client.get_dict()
                    try:
                        res_dict = result['organic_results']
                        api_keys_dict[key] = api_keys_dict[key] + 1
                        for org_res in res_dict:
                            links.append(org_res['link'])
                    except:
                        print("error search")
        else:
            continue
    write_api_keys(api_keys_dict, 'api_keys.txt')
    return links


def processing_text_page(text):
    # предобработка всего текста страницы
    # в разработке
    pass


def get_vector(text, model):
    # use glove model for start, in future will use another model for text representation
    # vector = get_bert_embeddings(text)
    vector = []
    if model == 'sbert':
        vector = model_sb.encode(text)
    if model == 'doc2vec':
        vector = get_d2v_embeddings(text)
    if model == 'use':
        vector = use_records_to_embeds(text)
    return vector


def find_and_mark_text_not_alpha(text_obj, entity_name):
    text_obj = re.sub(r'\s+', ' ', text_obj, flags=re.UNICODE)
    text_obj = text_obj.strip()
    texts = text_obj.split('.')
    markable_text_obj = []
    for text in texts:
        pass
    return markable_text_obj


def find_and_mark_text(text_obj, entity_name, alphabet=True):
    # rules
    text_obj = re.sub(r'\s+', ' ', text_obj, flags=re.UNICODE)
    text_obj = text_obj.strip()
    texts = text_obj.split('.')
    if alphabet is False:
        markable_text_obj = find_and_mark_text_not_alpha(text_obj, entity_name)
        return markable_text_obj
    # что делать если предложение одно и без точки
    markable_text_obj = []
    for text in texts:
        text = text.strip()
        if text == '':
            continue
        tokens = text.split(' ')
        if len(tokens) == 0 or len(tokens) == 1:
            continue
        ners = []
        for index, token in enumerate(tokens):
            if tokens[0][0].isupper() and tokens[1][0].islower() and index == 0:
                # случай когда слово просто начинает предложение и не является частью сущности
                ners.append('O')
                continue
            elif tokens[index][0].isupper() and tokens[index - 1][0].islower():
                # начало сущности
                ners.append('B-' + entity_name)
            elif index < len(tokens) - 1 and tokens[index][0].isupper() and tokens[index + 1][0].isupper():
                if tokens[index - 1][0].isupper() and index != 0:
                    ners.append('I-' + entity_name)
                elif index == 0:
                    ners.append('B-' + entity_name)
                else:
                    ners.append('B-' + entity_name)
            elif tokens[index][0].isupper() and tokens[index - 1][0].isupper():
                # середина или конец сущности
                if index < len(tokens) - 1 and tokens[index + 1][0].islower():
                    # конец сущности
                    ners.append('O')
                else:
                    # середина сущности
                    ners.append('I-' + entity_name)
            elif tokens[index][0].isdecimal() and tokens[index - 1][0].isupper():
                # продолжение сущности в виде числа
                ners.append('I-' + entity_name)
            else:
                ners.append('O')
        assert len(tokens) == len(ners)
        if list(set(ners)) == ['O']:
            # проверка на то, что в предложении отсутствуют сущности
            continue
        markable_text = [(token, ner) for token, ner in zip(tokens, ners)]
        markable_text_obj.append(markable_text)
    return markable_text_obj


def _write_array(array, path, indexing=False, delemeter='\t'):
    with open(path, 'a+', encoding='utf-8') as fout:
        for id, sample in enumerate(array):
            if id == 0:
                fout.write("Sentence #;Word;Tag\n")
            for index, (token, ent) in enumerate(sample):
                if delemeter in token:
                    continue
                if indexing is True:
                    if index == 0:
                        fout.write("Sentence: " + str(id) + delemeter + token + delemeter + ent)
                        fout.write('\n')
                    else:
                        fout.write(delemeter + token + delemeter + ent)
                        fout.write('\n')
                else:
                    fout.write(token + '\t' + ent)
                    fout.write('\n')
            if indexing is False:
                fout.write('\n')


def write_files(links_info, path_folder='/', indexing=False, split_k=(0.8, 0.1, 0.1), delemeter='\t'):
    train, test, val = split_k[0], split_k[0] + split_k[1], 1
    train, test_val = train_test_split(links_info, test_size=0.2, train_size=0.8)
    test, valid = train_test_split(test_val, train_size=0.5, test_size=0.5)
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    path_train = os.path.join(path_folder, 'train.txt')
    path_test = os.path.join(path_folder, 'test.txt')
    path_val = os.path.join(path_folder, 'valid.txt')
    _write_array(train, path_train, indexing=indexing, delemeter=delemeter)
    _write_array(test, path_test, indexing=indexing, delemeter=delemeter)
    _write_array(valid, path_val, indexing=indexing, delemeter=delemeter)


def get_texts(objects):
    texts = [text[0] for text in objects]
    return texts


def get_page_info(html, query):
    # посмотреть по всем интересный тегам и забрать текста. проверит контекст и искомые слова
    info = []
    query = query.split(' ')
    soup = BeautifulSoup(html, 'html5lib')
    all_text = soup.text
    preproc_text = processing_text_page(all_text)
    tags = ['h1', 'h2', 'h3', 'a', 'p', 'b', 'td']
    in_vectors, out_vectors = [], []
    for tag in tags:
        blocks = soup.find_all(tag)
        for block in blocks:
            text = block.text
            # print(text)
            # vector = get_vector(text)
            if '�' in text:
                continue
            for q in query:
                if q in text:
                    in_vectors.append(text)
                else:
                    out_vectors.append(text)
    return in_vectors, out_vectors


def data_saturation(marked_data, saturate_percent=0.3):
    # функция по насыщению данных, чтобы сущности с большой буквы стали с маленькой для повышения качества
    sat_data = []
    total_sample = len(marked_data)
    sat_samples = int(total_sample * saturate_percent)
    array_index = np.random.randint(0, total_sample, size=(1, sat_samples)).tolist()[0]
    for i in range(sat_samples):
        data = marked_data[array_index[i]]
        new_data = []
        for token in data:
            if token[1] == 'B-' or token[1] == 'I-' or token[1] == 'O-':
                new_token = token[0].lower()
            else:
                new_token = token[0]
            new_data.append((new_token, token[1]))
        sat_data.append(new_data)
    return sat_data


def get_neighbours(in_vectors, out_vectors, threshold=0.7):
    in_vectors = [(text, vec) for text, vec in zip(list(set(in_vectors)), get_vector(in_vectors, model='sbert'))]
    out_vectors = [(text, vec) for text, vec in zip(list(set(out_vectors)), get_vector(out_vectors, model='sbert'))]
    res_vectors = []
    for ouv in out_vectors:
        k = 0
        for inv in in_vectors:
            corr = distance.cosine(inv[1], ouv[1])
            if corr > 0.9:
                k += 1
        if k > len(in_vectors) * threshold:
            # print("TEXT:"+ouv[0])
            res_vectors.append(ouv)
    return res_vectors


def main():
    ENTITY = 'ELEC'
    LOGFILE = "logs/elec_without_search.txt"
    PRINT_TO_FILE = False
    log = Logger(LOGFILE, PRINT_TO_FILE)
    print = log.printml
    language = 'ru'
    language_subwords = {
        'ru': ['описание', 'на русском', 'похожие', 'аналоги'],
        'en': ['description', 'english', 'analogs'],
        'ar': ['الوصف', 'باللغة العربية', 'مماثلة', 'النظير'],
        'ch': ["描述", "中文", "类似", "类似"],
    }
    keywords = language_subwords[language]
    total_timer = Timer()
    func_timer = [Timer() for t in range(6)]
    total_timer.start()
    query = "cisco tp-link Zyxel"
    link_path = 'links/elec_links.txt'
    func_timer[0].start()
    if os.path.exists(link_path):
        with open(link_path, 'r') as f:
            text = f.read()
            query_links = text.split('\n')
            print('Read links from file: ' + link_path + " Total: " + str(len(query_links)))
    else:
        query_links = search_links(query, keywords, language=language)
        with open(link_path, 'w') as f:
            for link in query_links:
                f.write(link + "\n")
        print('Search links to query: ' + query + " Total: " + str(len(query_links)))
    # final_query = query + keywords
    func_timer[0].stop(text="Query search time")
    links_info = []
    func_timer[1].start()
    in_vectors, out_vectors = [], []
    for index, link in enumerate(tqdm(query_links)):
        try:
            in_vectors_page, out_vectors_page = get_page_info(get_html(link), query)
        except:
            continue
        in_vectors.extend(in_vectors_page)
        out_vectors.extend(out_vectors_page)
    print("In vectors total: " + str(len(in_vectors)))
    print("Out vectors total: " + str(len(out_vectors)))
    func_timer[1].stop(text='get_page_info time:')
    func_timer[2].start()
    in_vectors.extend(get_neighbours(in_vectors, out_vectors))
    print("In vectors total after similarity: " + str(len(in_vectors)))
    func_timer[2].stop(text="get_neighbours time:")
    func_timer[3].start()
    for vec in in_vectors:
        marked_vec = find_and_mark_text(vec[0], ENTITY)
        if len(marked_vec) != 0:
            if type(marked_vec[0]) is list:
                links_info.extend(marked_vec)
            else:
                links_info.append(marked_vec)
    print("Marked samples: " + str(len(links_info)))
    func_timer[3].stop(text="marked text time:")
    func_timer[4].start()
    # процедура насыщения данных
    links_info.extend(data_saturation(links_info))
    print("Marked samples after saturation: " + str(len(links_info)))
    func_timer[4].stop(text="data saturation time:")
    random.shuffle(links_info)
    func_timer[5].start()
    write_files(links_info, path_folder='data/sbert_elec', indexing=True, delemeter=';')
    func_timer[5].stop('write file time')
    total_timer.stop(text='Total time:')


if __name__ == "__main__":
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    use_model = hub.load(module_url)
    model_sb = SentenceTransformer('distiluse-base-multilingual-cased')
    model_sb.save('./use_model')
    model_d2v = Doc2Vec.load('doc2vec_model')
