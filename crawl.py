import argparse
import csv
import random
import traceback

from web3 import Web3

from spiders.doc import generate_transactions, fetch_contract_documentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--apikey', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    f = open(args.out, 'w', encoding='utf-8', newline='\n')
    writer = csv.writer(f)
    writer.writerow(['transaction_hash', 'label'])
    random.seed(args.seed)
    while True:
        try:
            block_number = random.choice(range(1, 18 * 1000000))
            transactions = [tx for tx in generate_transactions(block_number, args.apikey)]
            addresses = set([tx['to'] for tx in transactions if Web3.is_address(tx['to'])])
            rlt = fetch_contract_documentation(addresses=list(addresses), apikey=args.apikey)
            for tx in transactions:
                sign2docu = rlt.get(tx['to'])
                if sign2docu is None:
                    continue
                sign = tx['input'][:2 + 8]
                docu = sign2docu.get(sign)
                if docu is None:
                    continue
                row = [tx['hash'], docu]
                writer.writerow(row)
                print(row)
        except Exception as _:
            print(traceback.format_exc())
