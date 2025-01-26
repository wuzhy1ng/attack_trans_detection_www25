import asyncio
import csv

import os
from hashlib import sha256
from typing import List, Dict

PROVIDER = 'https://eth-mainnet.nodereal.io/v1/317f6d43dd4c4acea1fa00515cf02f90'


async def run_collect(txhashes: List, out_path: str):
    command = ' '.join([
        'scrapy crawl trans.web3',
        '-a hash={}'.format(','.join(txhashes)),
        '-a providers=%s' % PROVIDER,
        '-a out=%s' % out_path,
        '-s CONCURRENT_REQUESTS=4',
    ])
    process = await asyncio.create_subprocess_shell(
        cmd=command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output, err_out = await process.communicate()
    print(output, err_out)


def save_doc(transactions: Dict, out_path: str):
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(['transaction_hash', 'documentation'])
        for k, v in transactions.items():
            writer.writerow([k, v])


def generate_transactions(path: str):
    count = 0
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rlt = dict()
        for row in reader:
            transaction_hash = row['transaction_hash']
            rlt[transaction_hash] = row['doc']

            if len(rlt) == 100:
                yield rlt
                rlt = dict()

            if count == 100000:
                break
            count += 1


if __name__ == '__main__':
    csv_file_path = r'H:\python_projects\multimodal_trans_semantics\spiders\merged_data.csv'
    out_path = r'J:\multimodal_trans_semantics_data\train'
    waits, tasks = 2, list()
    sid = 0
    for transactions in generate_transactions(path=csv_file_path):
        sid += 1
        print('process:', sid)
        folder_name = ','.join([txhash for txhash in transactions.keys()])
        folder_name = sha256(folder_name.encode()).hexdigest()
        if os.path.exists(os.path.join(out_path, folder_name)):
            continue
        os.makedirs(os.path.join(out_path, folder_name))

        # save doc
        save_doc(
            transactions=transactions,
            out_path=os.path.join(out_path, folder_name, 'Documentation.csv'),
        )

        # create task
        tasks.append(run_collect(
            txhashes=[txhash for txhash in transactions.keys()],
            out_path=os.path.join(out_path, folder_name),
        ))
        if len(tasks) < waits:
            continue
        tasks = asyncio.gather(*tasks)
        asyncio.get_event_loop().run_until_complete(tasks)
        tasks = list()
