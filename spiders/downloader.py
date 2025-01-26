import json
import os
import urllib.parse
from typing import Dict

import aiohttp

from settings import CACHE_DIR


class Downloader:
    async def download(self, *args, **kwargs):
        result = await self._preprocess(*args, **kwargs)
        if result is not None:
            return result
        result = await self._fetch(*args, **kwargs)
        return await self._process(result, **kwargs)

    async def _preprocess(self, *args, **kwargs):
        raise NotImplemented()

    async def _fetch(self, *args, **kwargs):
        raise NotImplemented()

    async def _process(self, result, *args, **kwargs):
        raise NotImplemented()


class EtherscanDownloader(Downloader):
    def __init__(self, apikey: str):
        self.apikey = apikey

    def get_request_param(self, *args, **kwargs) -> Dict:
        raise NotImplemented()

    async def _fetch(self, *args, **kwargs):
        params = self.get_request_param(*args, **kwargs)
        print(params)
        client = aiohttp.ClientSession()
        async with client.get(**params) as response:
            rlt = await response.text()
        await client.close()
        return rlt


class ContractSourceDownloader(EtherscanDownloader):
    def get_request_param(self, contract_address: str) -> Dict:
        query_params = urllib.parse.urlencode({
            "module": "contract",
            "action": "getsourcecode",
            "address": contract_address.lower(),
        })
        return {"url": '{}&{}'.format(self.apikey, query_params)}

    async def _preprocess(self, contract_address: str, **kwargs):
        path = os.path.join(CACHE_DIR, 'source', '%s.json' % contract_address)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    async def _process(self, result: str, **kwargs):
        result = json.loads(result)
        result = result['result'][0]

        # cache data
        contract_address = kwargs['contract_address']
        path = os.path.join(CACHE_DIR, 'source')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '%s.json' % contract_address)
        with open(path, 'w') as f:
            json.dump(result, f)
        return result


class TransactionDownloader(EtherscanDownloader):
    def get_request_param(self, contract_address: str) -> Dict:
        query_params = urllib.parse.urlencode({
            "module": "account",
            "action": "txlist",
            "address": contract_address.lower(),
            "startblock": 0,
            "sort": "asc",
        })
        return {"url": '{}&{}'.format(self.apikey, query_params)}

    async def _preprocess(self, contract_address: str, **kwargs):
        path = os.path.join(CACHE_DIR, 'transaction', '%s.json' % contract_address)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    async def _process(self, result: str, **kwargs):
        result = json.loads(result)
        result = result['result']

        # cache data
        contract_address = kwargs['contract_address']
        path = os.path.join(CACHE_DIR, 'transaction')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '%s.json' % contract_address)
        with open(path, 'w') as f:
            json.dump(result, f)
        return result


async def test():
    d = ContractSourceDownloader('https://api.etherscan.io/api?apikey=7MM6JYY49WZBXSYFDPYQ3V7V3EMZWE4KJK')
    print(await d.download(
        contract_address='0xde744d544a9d768e96c21b5f087fc54b776e9b25',
    ))
    d = TransactionDownloader('https://api.etherscan.io/api?apikey=7MM6JYY49WZBXSYFDPYQ3V7V3EMZWE4KJK')
    rlt = await d.download(
        contract_address='0xde744d544a9d768e96c21b5f087fc54b776e9b25',
    )
    print(type(rlt))


if __name__ == '__main__':
    import asyncio

    data = asyncio.run(test())
