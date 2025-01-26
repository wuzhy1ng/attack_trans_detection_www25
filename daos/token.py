from typing import Iterator

from daos.defs import CSVReader


class Token20TransferReader(CSVReader):
    def __init__(self, path: str, fn: str = 'Token20TransferItem.csv'):
        super().__init__(path, fn)


class Token721TransferReader(CSVReader):
    def __init__(self, path: str, fn: str = 'Token721TransferItem.csv'):
        super().__init__(path, fn)


class Token1155TransferReader(CSVReader):
    def __init__(self, path: str, fn: str = 'Token1155TransferItem.csv'):
        super().__init__(path, fn)

    def iter_read(self) -> Iterator[dict]:
        for row in super().iter_read():
            token_ids, values = eval(row['token_ids']), eval(row['values'])
            for token_id, value in zip(token_ids, values):
                yield {'token_id': token_id, 'value': value, **row}


class TokenApprovalReader(CSVReader):
    def __init__(self, path: str, fn: str = 'TokenApprovalItem.csv'):
        super().__init__(path, fn)


class TokenApprovalAllReader(CSVReader):
    def __init__(self, path: str, fn: str = 'TokenApprovalAllItem.csv'):
        super().__init__(path, fn)
