import argparse
import csv
import datetime
import os
from typing import List, Dict, Set

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import mse_loss
from torch_geometric.loader import DataLoader

from dataset.pyg import MultiModalTransactionDataset
from pretrain import Model
from utils.transform import format_data_type, TextFeatEmbedding


def infer(
        model: Model,
        dataset: MultiModalTransactionDataset,
        batch_size: int, device: torch.device,
) -> List[Dict]:
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )

    result = list()
    batch_cnt, batch_total = 0, len(dataloader)
    print('start inferring, total batch:', len(dataloader))
    text_emb = TextFeatEmbedding().to(device)
    for batch_data in dataloader:
        with torch.no_grad():
            batch_data = batch_data.to(device)
            batch_data = text_emb(batch_data)
            graph_feats = model.graph_model(batch_data)
            encoded_graph_feats = model.encoder(graph_feats)
            decoded_graph_feats = model.decoder(encoded_graph_feats)
            reconstruct_loss = mse_loss(graph_feats, decoded_graph_feats)
            reconstruct_loss = reconstruct_loss.flatten().tolist()
            batch_result = list()
            for txhash, loss, embedding in zip(
                    batch_data.transaction_hash,
                    reconstruct_loss, encoded_graph_feats.tolist(),
            ):
                batch_result.append({
                    "transaction_hash": txhash,
                    "restruct_loss": loss,
                    "embedding": [*embedding, loss],
                })
            if getattr(batch_data, "label"):
                for label, row in zip(batch_data.label, batch_result):
                    row["label"] = label
            result.extend(batch_result)
            batch_cnt += 1
            if batch_cnt % 1000 == 0:
                print(datetime.datetime.now(), "batch #%d / %d" % (batch_cnt, batch_total))
    return result


def evaluate(result: List[Dict], attack_types: Set):
    # 1. 看看阈值分布，确定阈值
    attack_score, non_attack_score = list(), list()
    for row in result:
        label = row.get('label')
        if label in attack_types:
            attack_score.append(row['restruct_loss'])
        # elif label == 'Non-attack':
        else:
            non_attack_score.append(row['restruct_loss'])
    print('attack: {} (mean), {} (std), {} (max), {} (min)'.format(
        np.mean(attack_score), np.std(attack_score),
        np.max(attack_score), np.min(attack_score)
    ))
    print('non-attack: {} (mean), {} (std), {} (max), {} (min)'.format(
        np.mean(non_attack_score), np.std(non_attack_score),
        np.max(non_attack_score), np.min(non_attack_score)
    ))

    # 2. 画分布图
    plt.figure(figsize=(8, 5), dpi=256)
    box = plt.boxplot(
        [non_attack_score, attack_score],
        tick_labels=['non-attack', 'attack'],
        showfliers=False,
    )
    boundaries = np.concatenate([
        box['whiskers'][0].get_data()[1],
        box['whiskers'][1].get_data()[1],
    ])
    non_attack_min, non_attack_max = np.min(boundaries), np.max(boundaries)
    boundaries = np.concatenate([
        box['whiskers'][2].get_data()[1],
        box['whiskers'][3].get_data()[1],
    ])
    attack_min, attack_max = np.min(boundaries), np.max(boundaries)
    plt.clf()

    fig, ax = plt.subplots(figsize=(5, 4))
    hist_data = [s for s in non_attack_score if non_attack_min <= s <= non_attack_max]
    print('non-attack:', hist_data)
    ax.hist(
        hist_data, bins=25, alpha=0.8,
        density=True, stacked=True, label="Non-attack",
    )
    hist_data = [s for s in attack_score if attack_min <= s <= attack_max]
    print('attack:', hist_data)
    ax.hist(
        hist_data, bins=25, alpha=0.8,
        density=True, stacked=True, label="Attack",
    )
    ax.axhline(0, color="k")
    ax.set_xlabel('Reconstruction error', fontsize=18)
    ax.set_ylabel('Proportion (%)', fontsize=18)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=18, loc='upper right')
    fig.savefig('hist.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretrain_path', type=str, required=True)
    parser.add_argument('--hidden_channels', type=int, default=384)
    parser.add_argument('--out_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    torch.manual_seed(43)

    # loading model params
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('using cuda now....')
    model = Model(
        device=device,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
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
    )
    model.load_state_dict(
        state_dict=torch.load(args.pretrain_path) if torch.cuda.is_available()
        else torch.load(args.pretrain_path, map_location=torch.device('cpu')),
        strict=False,
    )

    # loading dataset
    dataset = MultiModalTransactionDataset(
        root=args.data_path,
        signature_path='',
        transform=format_data_type,
    )

    # infer
    result = infer(
        model=model, dataset=dataset,
        batch_size=args.batch_size, device=device,
    )

    # evaluation
    attack_types = set()
    for path in os.listdir(dataset.raw_dir):
        fn = os.path.join(dataset.raw_dir, path, 'Label.csv')
        if not os.path.exists(fn):
            continue
        with open(fn, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                label = row['label']
                if label.startswith('Hack'):
                    attack_types.add(label)
    evaluate(result=result, attack_types=attack_types)
