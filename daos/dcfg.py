from typing import Iterator

from daos.defs import CSVReader
from utils import split_camel_case


class DCFGBlockReader(CSVReader):
    def __init__(self, path: str, fn: str = 'DCFGBlock.csv'):
        super().__init__(path, fn)

    def iter_read(self) -> Iterator[dict]:
        for row in super().iter_read():
            block_id = '{}#{}'.format(row['contract_address'], row['start_pc'])
            row['operations'] = eval(row['operations'])
            yield {'block_id': block_id, **row}


class DCFGEdgeReader(CSVReader):
    def __init__(self, path: str, fn: str = 'DCFGEdge.csv', signature2keyword: dict = None):
        super().__init__(path, fn)
        self.signature2keyword = signature2keyword
        assert self.signature2keyword is not None

    def iter_read(self) -> Iterator[dict]:
        for row in super().iter_read():
            from_block_id = '{}#{}'.format(row['address_from'], row['start_pc_from'])
            to_block_id = '{}#{}'.format(row['address_to'], row['start_pc_to'])
            index = str(row.get('index'))
            row['index'] = int(index) if index.isdigit() else -1
            func_name = self.signature2keyword.get(row.get('selector'), '')
            func_name = [token.lower() for token in split_camel_case(func_name)]
            func_name = ' '.join(func_name)
            yield {
                'from_block_id': from_block_id,
                'to_block_id': to_block_id,
                'func_name': func_name,
                **row
            }
