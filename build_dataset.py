import argparse
import datetime
import os

from dataset.pyg import MultiModalTransactionDataset
from settings import PROJECT_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--slice_len', type=int, default=1000)
    parser.add_argument('--currency', type=int, default=3)
    args = parser.parse_args()

    print(datetime.datetime.now(), 'start building dataset...')
    signature_path = os.path.join(PROJECT_PATH, 'misc', 'SignItem.csv')
    dataset = MultiModalTransactionDataset(
        root=args.data_path,
        signature_path=signature_path,
        slice_len=args.slice_len,
        currency=args.currency,
    )

    node_types, edge_types = dataset.metadata
    print(dataset.metadata)
    print(datetime.datetime.now(), 'finished!')

    print(datetime.datetime.now(), 'checking data now...')
    node_type2dims, edge_type2dims = dict(), dict()
    for i, data in enumerate(dataset):
        try:
            data.validate()
        except:
            print('error data format in: #%d' % i)
    print('ok')
