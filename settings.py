import json
import os

PROJECT_PATH, _ = os.path.split(os.path.realpath(__file__))

SOLCJS_CODE_RELATED_PATH = 'misc/solcjs.js'
with open(os.path.join(PROJECT_PATH, SOLCJS_CODE_RELATED_PATH), 'r') as f:
    SOLCJS_CODE = f.read()

CACHE_DIR = os.path.join(PROJECT_PATH, 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

TMP_FILE_DIR = os.path.join(PROJECT_PATH, 'tmp')
if not os.path.exists(TMP_FILE_DIR):
    os.makedirs(TMP_FILE_DIR)

NODE_PATH = 'node'

SCAN_APIKEYS = {
    'Ethereum': [
        'https://api.etherscan.io/api?apikey=7MM6JYY49WZBXSYFDPYQ3V7V3EMZWE4KJK',
        'https://api.etherscan.io/api?apikey=J9996KUX8WNA5I86WY67ZMZK72SST1BIW8',
    ],
    'BNBChain': [
        'https://api.bscscan.com/api?apikey=3FYU1X8HNHNQ287PUIXZBFYWT78TBPG4P6',
    ],
}
JSONRPCS = {
    'Ethereum': [
        'https://eth-mainnet.nodereal.io/v1/317f6d43dd4c4acea1fa00515cf02f90',
    ],
    'BNBChain': [
        'https://bsc-mainnet.nodereal.io/v1/dcee58db7567445f811778fa1029fbb1',
    ]
}

HUGGING_MODEL_PATH = os.path.join(PROJECT_PATH, 'hugging_cache')
if not os.path.exists(HUGGING_MODEL_PATH):
    HUGGING_MODEL_PATH = None

PT_CACHE_SIZE = 32
with open(os.path.join(PROJECT_PATH, 'misc/metadata.json'), 'r') as f:
    HETERO_GRAPH_METADATA = json.load(f)
