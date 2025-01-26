This is a basic implementation of `Hunting in the Dark Forest: A Pre-trained Model for On-chain Attack Transaction Detection in Web3`.

# installation
Please run the following command in your console:
```shell
npm install
pip install -r requirements.txt
```

# Build your dataset
Please run the following command for collecting function comments:
```shell
python crawl.py --out=/path/to/out \
  --apikey=/your/apikey/in/etherscan
```
And use the following command for make the dataset:
```shell
python build_dataset.py --data_path=/path/to/dataset
```
There is a file tree example of the raw data in the path of `/path/to/dataset`:
```shell
- raw
  - path_01
    - DCFGEdge.csv
    - DCFGBlock.csv
    - TransactionItem.csv
    - TransactionReceiptItem.csv
    - EventLogItem.csv
    - TokenPropertyItem.csv
    - TokenApprovalItem.csv
    - TokenApprovalAllItem.csv
    - Token1155TransferItem.csv
    - Token721TransferItem.csv
    - Token20TransferItem.csv
    - Label.csv
  - path_02
    - ...
```
Note that the data can be collect by `BlockchainSpider` and the `Label.csv` can be collect by `crawl.py`.

# Pre-training
Please run the following command in your console:
```shell
python pretrain.py --data_path=/path/to/dataset \
  --num_layers=3 --num_heads=6 --hidden_channels=384 \
  --num_workers=4 --batch_size=64 \
  --report_step=10 --epoch=10
```

# Evaluation
- `test_few_shot.py`: perform the few-shot on-chain attack detection.
- `test_wild_detection.py`: perform the zero-shot on-chain attack detection.

# Collect transaction data
Use the following commend to collect the data related to one transaction:
```shell
scrapy crawl trans.web3 -a hash=<your transaction hash> \
  -a out=/path/for/output \
  -a providers=<your http rpc provider> \
  -a enable=BlockchainSpider.middlewares.trans.TransactionReceiptMiddleware,BlockchainSpider.middlewares.trans.DCFGMiddleware,BlockchainSpider.middlewares.trans.TokenMiddleware
```
For more usage detail, please search `BlockchainSpider` on github.
And we will make the pre-trained model parameters online available, after the paper is published.