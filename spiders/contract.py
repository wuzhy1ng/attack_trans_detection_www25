import asyncio
import json
import re
from json import JSONDecodeError
from typing import Dict, Union

from settings import NODE_PATH, SOLCJS_CODE
from spiders.downloader import Downloader
from utils.solc import SolcJS
from utils.tmpfile import wrap_run4tmpfile


class ContractCompileItem:
    def __init__(
            self, contract_address: str, bytecode: str,
            userdoc: dict, devdoc: dict,
    ):
        self.contract_address = contract_address
        self.bytecode = bytecode
        self.userdoc = userdoc
        self.devdoc = devdoc


class ContractDao:
    def __init__(self, downloader: Downloader):
        self.downloader = downloader

    async def get_compile_item(self, contract_address: str) -> ContractCompileItem:
        """
        Compile the contract source code, which is fetched from etherscan.

        :param contract_address: the address of specific contract
        :return: the compiled result
        """
        # fetch source code and save to tmp file
        result = await self.downloader.download(contract_address=contract_address)
        if result.get('SourceCode') is None or result['SourceCode'] == '':
            return ContractCompileItem(contract_address, '', dict(), dict())

        # use solc-js to document one sol source code file
        product = await self._get_compile_item_by_solcjs(
            contract_address=contract_address,
            result=result,
        )
        return product if product is not None else ContractCompileItem(contract_address, '', dict(), dict())

    async def _get_compile_item_by_solcjs(
            self, contract_address: str, result: Dict
    ) -> Union[ContractCompileItem, None]:
        """
        Compile the contract source code by `solc-js`,
        and return the compiled item if available,
        otherwise return None.

        :param contract_address: the address of specific contract
        :param result: the result of the source code request
        :return: the compiled item or None
        """

        solc_version = re.search('v(.*?)\+commit', result["CompilerVersion"])
        if solc_version is None:
            return None
        solc_version = 'v%s' % solc_version.group(1)
        contract_name = result["ContractName"]
        _tmp_filename = "this_is_a_tmp_filename.sol"
        try:
            standard_json = json.loads(result['SourceCode'][1:-1])
        except JSONDecodeError:
            standard_json = {
                "language": "Solidity",
                "settings": {
                    "optimizer": {
                        "enabled": result["OptimizationUsed"] == '1',
                        "runs": int(result["Runs"]),
                    },
                },
                "sources": {
                    _tmp_filename: {
                        "content": result['SourceCode'].replace('\r\n', '\n'),
                    }
                }
            }
            if result['Library'] != '':
                libraries = result['Library'].split(',')
                standard_json['settings']['libraries'] = {
                    lib.split(':')[0]: '0x%s' % lib.split(':')[1]
                    for lib in libraries
                }

        # return the compilation result
        standard_json['settings']['outputSelection'] = {
            "*": {
                "*": ["evm.deployedBytecode", 'userdoc', 'devdoc'],
            },
        }
        product = await wrap_run4tmpfile(
            data=SOLCJS_CODE % (solc_version, json.dumps(standard_json)),
            async_func=lambda p: SolcJS(NODE_PATH).compile_json(p, contract_name)
        )
        if product is None:
            return None
        return ContractCompileItem(
            contract_address=contract_address,
            bytecode=product.bytecode,
            userdoc=product.userdoc,
            devdoc=product.devdoc,
        ) if product is not None else None

    async def is_contract(self, contract_address: str) -> bool:
        result = await self.downloader.download(contract_address=contract_address)
        return result != '0x'


async def test():
    from downloader import ContractSourceDownloader

    dao = ContractDao(
        ContractSourceDownloader('https://api.etherscan.io/api?apikey=7MM6JYY49WZBXSYFDPYQ3V7V3EMZWE4KJK'))
    item = await dao.get_compile_item('0x4d224452801aced8b2f0aebe155379bb5d594381')
    print(item)


if __name__ == '__main__':
    data = asyncio.run(test())
