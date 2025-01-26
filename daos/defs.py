import csv
import os
from typing import Iterator

import pandas as pd

csv.field_size_limit(2 ** 20)
chunk_size = 2 ** 10


class CSVReader:
    def __init__(self, path: str, fn: str):
        self.path = path
        self.fn = fn

    def iter_read(self) -> Iterator[dict]:
        fn = os.path.join(self.path, self.fn)
        if not os.path.exists(fn):
            return

        data = pd.read_csv(fn, chunksize=chunk_size)
        while True:
            try:
                data_slice = data.get_chunk()
            except StopIteration:
                return
            for item in data_slice.iterrows():
                yield item[1].to_dict()
            if len(data_slice) < chunk_size:
                return
                # with open(fn, 'r', encoding='utf-8') as f:
        # reader = csv.DictReader(f)
        # yield from reader
