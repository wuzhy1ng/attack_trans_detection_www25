import json
from asyncio import subprocess
from typing import Union


class CompileResult:
    def __init__(self, bytecode: str, userdoc: dict, devdoc: dict):
        self.bytecode = bytecode
        self.userdoc = userdoc
        self.devdoc = devdoc


class SolcJS:
    """
    A solidity code compiler, based on `solc-js`.
    """

    def __init__(self, path: str, timeout: float = 10.0):
        self.path = path
        self.timeout = timeout

    async def compile_json(
            self, standard_json_path: str, contract_name: str
    ) -> Union[CompileResult, None]:
        cmd = [self.path, standard_json_path]
        process = await subprocess.create_subprocess_shell(
            ' '.join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, _ = await process.communicate()
        try:
            product = json.loads(output.decode())
        except:
            return None

        # parse data
        contracts = product.get("contracts", dict())
        sources = product.get("sources", dict())
        if len(contracts) == len(sources) == 0:
            return None

        for path in contracts.keys():
            for _contract_name in contracts[path].keys():
                if _contract_name == contract_name:
                    target_contract = contracts[path]
                    break

        # extract the document info, e.g., bytecode, source mapping
        bytecode = target_contract[contract_name]["evm"]["deployedBytecode"]["object"]
        devdoc = target_contract[contract_name]['devdoc']['methods']
        userodc = target_contract[contract_name]['userdoc']['methods']
        return CompileResult(
            bytecode=bytecode,
            devdoc={method: item for method, item in devdoc.items()},
            userdoc={method: item for method, item in userodc.items()},
        )
