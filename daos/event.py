from typing import Iterator

from daos.defs import CSVReader
from utils import split_camel_case


class EventLogReader(CSVReader):
    def __init__(self, path: str, fn: str = 'EventLogItem.csv', signature2keyword: dict = None):
        super().__init__(path, fn)
        self.signature2keyword = signature2keyword
        assert self.signature2keyword is not None

    def iter_read(self) -> Iterator[dict]:
        for row in super().iter_read():
            event_name = list()
            row['topics'] = eval(row['topics'])
            if isinstance(row['topics'], list) and len(row['topics']) > 0:
                event_name = self.signature2keyword.get(row['topics'][0], '')
                event_name = [token.lower() for token in split_camel_case(event_name)]
            yield {'event_name': ' '.join(event_name), **row}
