import csv
import re
import functools


@functools.lru_cache
def load_signatures(path: str) -> dict:
    rlt = dict()
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for item in reader:
            keyword = re.sub('\(.*\)', '', item['text'])
            rlt[item['sign']] = keyword
    return rlt
