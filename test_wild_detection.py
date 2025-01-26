import argparse
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from dataset.pyg import MultiModalTransactionDataset
from pretrain import Model
from settings import PROJECT_PATH
from utils.transform import TextFeatEmbedding, format_data_type


def get_detect_model(model_path: str, data_path: str, signature_path: str):
    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
    model = Model(
        hidden_channels=384,
        out_channels=384,
        num_layers=3,
        num_heads=6,
        metadata=(
            ["Contract", "Log", "Block", "Account"],
            [["Contract", "CREATE", "Contract"], ["Contract", "Select", "Block"],
             ["Contract", "TokenApprovalAll", "Account"], ["Contract", "SELFDESTRUCT", "Contract"],
             ["Contract", "TokenApproval", "Contract"], ["Contract", "Token1155Transfer", "Account"],
             ["Account", "Token1155Transfer", "Contract"], ["Contract", "Token721Transfer", "Account"],
             ["Account", "Token721Transfer", "Contract"], ["Account", "Token20Transfer", "Account"],
             ["Contract", "CALL", "Contract"], ["Account", "Transaction", "Contract"],
             ["Account", "TokenApprovalAll", "Account"], ["Block", "JUMP", "Block"],
             ["Contract", "TokenApproval", "Account"], ["Account", "Token721Transfer", "Account"],
             ["Account", "Token1155Transfer", "Account"], ["Account", "TokenApproval", "Contract"],
             ["Block", "JUMPI", "Block"], ["Block", "Emit", "Log"], ["Contract", "Token20Transfer", "Contract"],
             ["Account", "Transaction", "Account"], ["Contract", "DELEGATECALL", "Contract"],
             ["Account", "TokenApproval", "Account"], ["Contract", "TokenApprovalAll", "Contract"],
             ["Contract", "Token1155Transfer", "Contract"], ["Contract", "Token721Transfer", "Contract"],
             ["Contract", "Token20Transfer", "Account"], ["Account", "Token20Transfer", "Contract"],
             ["Contract", "STATICCALL", "Contract"], ["Contract", "CREATE2", "Contract"],
             ["Account", "TokenApprovalAll", "Contract"]]
        ),
        device=device,
    )
    model.load_state_dict(
        state_dict=torch.load(model_path, map_location=torch.device('cpu')),
        strict=False,
    )
    model = model.to(device)
    text_emb = TextFeatEmbedding().to(device)
    model.eval()
    text_emb.eval()

    # init dataset
    dataset = MultiModalTransactionDataset(
        root=data_path,
        signature_path=signature_path,
    )

    # init train data
    y_true, x_train = [], []
    targets_attack = [
        'Unknown',
        'NFT_Sale', 'NFT_Mint', 'NFT_Burn',
        'GameFi_Birth', 'GameFi_Auction',
        'DeFi_Trade', 'DeFi_FlashLoan', 'DeFi_Burn',
        'DeFi_Remove Liquidity', 'DeFi_Add Liquidity',
        'Basic_Transfer', 'Basic_Approve',
        'Hack_Business Logic Flaw', 'Hack_Precision Loss', 'Hack_Misconfiguration',
        'Hack_Flawed Price Dependency', 'Hack_Storage Collision', 'Hack_Untrusted Input', 'Hack_Access Control Issue',
        'Hack_Arbitrary Calls', 'Hack_Flawed Price Calculation', 'Hack_Reward Calculation Error',
        'Hack_Fee Machenism Exploitation', 'Hack_Lack Slippage Protection',
        'Hack_Reentrancy',
        'Hack_Integer Overflow', 'Hack_Flashloan attack',
        'Hack_Call Injection', 'Hack_Honeypot',
    ]
    target2index = {t: i for i, t in enumerate(targets_attack)}
    for data in dataset:
        if data.label not in target2index:
            continue
        label_idx = target2index[data.label]
        y_true.append(-1 if label_idx > 12 else 1)
        with torch.no_grad():
            data = format_data_type(data)
            data = data.to(device)
            data = text_emb(data)
            graph_embeds = model.graph_model(data)
        x_train.append(graph_embeds.flatten().detach().cpu().numpy())

    # few-shot learning
    det_model = RandomForestClassifier(
        n_estimators=300, n_jobs=32,
        max_samples=1 / 300, random_state=42,
    )
    det_model.fit(x_train, y_true)
    return det_model


def wild_detect(
        model_path: str, data_path: str, signature_path: str,
        detect_model: RandomForestClassifier,
):
    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
    model = Model(
        hidden_channels=384,
        out_channels=384,
        num_layers=3,
        num_heads=6,
        metadata=(
            ["Contract", "Log", "Block", "Account"],
            [["Contract", "CREATE", "Contract"], ["Contract", "Select", "Block"],
             ["Contract", "TokenApprovalAll", "Account"], ["Contract", "SELFDESTRUCT", "Contract"],
             ["Contract", "TokenApproval", "Contract"], ["Contract", "Token1155Transfer", "Account"],
             ["Account", "Token1155Transfer", "Contract"], ["Contract", "Token721Transfer", "Account"],
             ["Account", "Token721Transfer", "Contract"], ["Account", "Token20Transfer", "Account"],
             ["Contract", "CALL", "Contract"], ["Account", "Transaction", "Contract"],
             ["Account", "TokenApprovalAll", "Account"], ["Block", "JUMP", "Block"],
             ["Contract", "TokenApproval", "Account"], ["Account", "Token721Transfer", "Account"],
             ["Account", "Token1155Transfer", "Account"], ["Account", "TokenApproval", "Contract"],
             ["Block", "JUMPI", "Block"], ["Block", "Emit", "Log"], ["Contract", "Token20Transfer", "Contract"],
             ["Account", "Transaction", "Account"], ["Contract", "DELEGATECALL", "Contract"],
             ["Account", "TokenApproval", "Account"], ["Contract", "TokenApprovalAll", "Contract"],
             ["Contract", "Token1155Transfer", "Contract"], ["Contract", "Token721Transfer", "Contract"],
             ["Contract", "Token20Transfer", "Account"], ["Account", "Token20Transfer", "Contract"],
             ["Contract", "STATICCALL", "Contract"], ["Contract", "CREATE2", "Contract"],
             ["Account", "TokenApprovalAll", "Contract"]]
        ),
        device=device,
    )
    model.load_state_dict(
        state_dict=torch.load(model_path, map_location=torch.device('cpu')),
        strict=False,
    )
    model = model.to(device)
    text_emb = TextFeatEmbedding().to(device)
    model.eval()
    text_emb.eval()

    # init dataset
    dataset = MultiModalTransactionDataset(
        root=data_path,
        signature_path=signature_path,
    )

    cache_feats_fn = os.path.join(PROJECT_PATH, 'wild_feats_cache.npy')
    cache_txhashes_fn = os.path.join(PROJECT_PATH, 'wild_txhashes_cache.npy')
    if os.path.exists(cache_feats_fn) and os.path.exists(cache_txhashes_fn):
        feats = np.load(cache_feats_fn)
        txhashes = np.load(cache_txhashes_fn)
    else:
        feats, txhashes = [], []
        for data in dataset:
            try:
                with torch.no_grad():
                    data = format_data_type(data)
                    data = data.to(device)
                    data = text_emb(data)
                    graph_embeds = model.graph_model(data)
            except:
                continue
            txhashes.append(data.transaction_hash)
            feats.append(graph_embeds.flatten().detach().cpu().numpy())
        np.save(cache_feats_fn, feats)
        np.save(cache_txhashes_fn, txhashes)

    # infer
    y_pred = detect_model.predict(feats)
    return {
        txhash: int(pred)
        for txhash, pred in zip(txhashes, y_pred)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--wild_data_path', type=str, required=True)
    parser.add_argument('--signature_path', type=str, default='')
    parser.add_argument('--pretrain_model_path', type=str, required=True)
    args = parser.parse_args()

    print('loading detection model')
    model = get_detect_model(
        data_path=args.train_data_path,
        signature_path=args.signature_path,
        model_path=args.pretrain_model_path,
    )

    print('detect attack in the wild')
    rlt = wild_detect(
        data_path=args.wild_data_path,
        signature_path=args.signature_path,
        model_path=args.pretrain_model_path,
        detect_model=model,
    )

    print('save result...')
    with open('wild_detection.json', 'w') as f:
        json.dump(rlt, f)
