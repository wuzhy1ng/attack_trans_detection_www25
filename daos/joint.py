from typing import Iterator

from daos.defs import CSVReader


class JointReader(CSVReader):
    def __init__(self, path: str, fn: str, joint_reader: CSVReader, joint_key: str):
        super().__init__(path, fn)
        self.joint_reader = joint_reader
        self.joint_key = joint_key

    def iter_read(self) -> Iterator[dict]:
        key2row = dict()
        for row in super().iter_read():
            key2row[row[self.joint_key]] = row

        # generate joint item
        for row in self.joint_reader.iter_read():
            extra_row = key2row.get(row[self.joint_key])
            if extra_row is None:
                yield row
                continue
            for k, v in extra_row.items():
                row[k] = v
            yield row
