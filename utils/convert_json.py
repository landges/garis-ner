import json
import pandas as pd
from google_scrap import _write_array
import re

path = 'data/ner_job_annotations.json'
with open(path, encoding='utf-8') as f:
    a = json.load(f)
p = 0
all_test = []
for ex in a['examples']:
    content = ex['content']
    # content = re.sub(r'[,.!?()]', '#', content)
    content_tokens = content.split(' ')
    annotations = sorted(ex['annotations'], key=lambda x: x['start'])
    if len(annotations) == 0:
        total_tags = ['O' for r in range(len(content_tokens))]
        # text_tag = [(token, tag) for token, tag in zip(content_tokens, total_tags)]
        # all_test.append(text_tag)
        p += 1
    else:
        total_tags = []
        for index, ent in enumerate(annotations):
            start = ent['start']
            end = ent['end']
            tag = ent['tag']
            value = ent['value']
            print((content, value))
            tokens_value = value.split(' ')
            tags = [' ' for i in range(len(tokens_value))]
            tags[0] = 'B-' + tag.upper()
            if len(tags) > 1:
                tags[-1] = 'I-' + tag.upper()
                if len(tags) > 2:
                    for j in range(1, len(tags) - 1):
                        tags[j] = 'I-' + tag.upper()
            if start == 0:
                total_tags.extend(tags)
                if len(annotations) == 1:
                    other_content = content[annotations[index]['end'] + 1:].split(' ')
                    if '' in other_content:
                        other_content.remove('')
                    other_tag = ['O' for n in range(len(other_content))]
                    total_tags.extend(other_tag)
            elif start > 0:
                if len(annotations) == 1:
                    other_content_1 = content[0:annotations[index]['start'] - 1].split(' ')
                    if '' in other_content_1:
                        other_content_1.remove('')
                    other_tag_1 = ['O' for n in range(len(other_content_1))]
                    total_tags.extend(other_tag_1)
                    total_tags.extend(tags)
                    other_content_2 = content[annotations[index]['end'] + 1:].split(' ')
                    if other_content_2 != ['']:
                        other_tag_2 = ['O' for n in range(len(other_content_2))]
                        total_tags.extend(other_tag_2)
                if len(annotations) > 1:
                    if index == 0:
                        start_other_content = content[:annotations[index]['start'] - 1].split(' ')
                        if '' in start_other_content:
                            start_other_content.remove('')
                        start_other_tags = ['O' for t in range(len(start_other_content))]
                        total_tags.extend(start_other_tags)
                        total_tags.extend(tags)
                    elif len(annotations) == 2:
                        other_content = content[
                                        annotations[index - 1]['end'] + 1:annotations[index]['start'] - 1].split(' ')
                        if '' in other_content:
                            other_content.remove('')
                        other_tag = ['O' for n in range(len(other_content))]
                        total_tags.extend(other_tag)
                        total_tags.extend(tags)
                        other_content_2 = content[annotations[index]['end'] + 1:].split(' ')
                        if other_content_2 != ['']:
                            if '' in other_content_2:
                                other_content_2.remove('')
                            other_tag_2 = ['O' for n in range(len(other_content_2))]
                            total_tags.extend(other_tag_2)
                    elif index == len(annotations) - 1:
                        end_other = content[annotations[index]['end'] + 1:].split(' ')
                        if end_other != ['']:
                            if '' in end_other:
                                end_other.remove('')
                            other_tag_end = ['O' for n in range(len(end_other))]
                            total_tags.extend(other_tag_end)
                        total_tags.extend(tags)

                    else:
                        other_content = content[
                                        annotations[index - 1]['end'] + 1:annotations[index]['start'] - 1].split(' ')
                        if '' in other_content:
                            other_content.remove('')
                        other_tag = ['O' for n in range(len(other_content))]
                        total_tags.extend(other_tag)
                        total_tags.extend(tags)

        # if len(content_tokens) == len(total_tags):

    text_tag = [(token, tag) for token, tag in zip(content_tokens, total_tags)]
    all_test.append(text_tag)
print(p)
print(len(all_test))
_write_array(all_test, 'json_data_expert3.csv', indexing=False, delemeter='\t')
