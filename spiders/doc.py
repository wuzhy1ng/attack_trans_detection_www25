import asyncio
import json
import os
import re
import time
from typing import Iterator, List

import requests
from web3 import Web3

from settings import PROJECT_PATH
from spiders.contract import ContractDao
from spiders.downloader import ContractSourceDownloader
from utils import split_camel_case

DOCU_CACHE_PATH = os.path.join(PROJECT_PATH, 'cache', 'document')
if not os.path.exists(DOCU_CACHE_PATH):
    os.makedirs(DOCU_CACHE_PATH)


def generate_transactions(block_number: int, apikey: str) -> Iterator[dict]:
    domain = 'https://api.etherscan.io/api'
    params = '?module=proxy&action=eth_getBlockByNumber&tag={}&boolean=true&apikey={}'.format(
        hex(block_number), apikey,
    )
    url = domain + params
    print(url)
    response = requests.get(url=url)
    data = json.loads(response.text)
    transactions = data['result']['transactions']
    for tx in transactions:
        if not tx.get('input') or tx['input'] == '0x':
            continue
        yield tx


def fetch_contract_documentation(addresses: List[str], apikey: str, currency: int = 3) -> dict:
    rlt = dict()  # addr -> sign -> doc
    downloader = ContractSourceDownloader('https://api.etherscan.io/api?apikey=' + apikey)
    for i in range(0, len(addresses), currency):
        addrs = list()

        # process cache
        for addr in addresses[i: i + currency]:
            cache_path = os.path.join(DOCU_CACHE_PATH, '%s.json' % addr)
            if not os.path.exists(cache_path):
                addrs.append(addr)
                continue
            with open(cache_path, 'r', encoding='utf-8') as cache_file:
                rlt[addr] = json.load(cache_file)
        if len(addrs) == 0:
            continue

        # run on the request
        tasks = asyncio.gather(*[
            ContractDao(downloader).get_compile_item(address)
            for address in addrs
        ])
        items = asyncio.get_event_loop().run_until_complete(tasks)
        time.sleep(1.0)
        addr2item = dict()
        for offset, item in enumerate(items):
            addr2item[addrs[offset]] = item

        # formatting the result
        for address, compile_item in addr2item.items():
            sign2doc = dict()
            for sign, doc in compile_item.devdoc.items():
                if not isinstance(doc, dict):
                    continue
                hex_sign = Web3.keccak(text=sign).hex()[:2 + 8]
                sign2doc[hex_sign] = doc
                sign_words = re.sub('\(.*\)', '', sign)
                sign2doc[hex_sign]['func_name'] = ' '.join(split_camel_case(sign_words))
            for sign, doc in compile_item.userdoc.items():
                if not isinstance(doc, dict):
                    continue
                hex_sign = Web3.keccak(text=sign).hex()[:2 + 8]
                sign2doc[hex_sign] = doc
                sign_words = re.sub('\(.*\)', '', sign)
                sign2doc[hex_sign]['func_name'] = ' '.join(split_camel_case(sign_words))
            rlt[address] = sign2doc
            cache_path = os.path.join(DOCU_CACHE_PATH, '%s.json' % address)
            with open(cache_path, 'w', encoding='utf-8') as cache_file:
                json.dump(sign2doc, cache_file)

    return rlt
