import argparse
import datetime
import os

import torch
from torch.cuda import OutOfMemoryError
from torch.nn.functional import mse_loss
from torch.utils.data import SequentialSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP

from models.simloss import SimLoss
from settings import PROJECT_PATH

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from dataset.pyg import MultiModalTransactionDataset
from models.Roberta import RoBERTa
from models.UniMP import UniMP
from utils.sampler import DeduplicateBatchSampler
from utils.transform import text_prompting, TransformSequence, format_data_type, TextFeatEmbedding


class Model(torch.nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.text_model = RoBERTa(**kwargs).to(device)
        self.graph_model = UniMP(**kwargs).to(device)
        self.sim_loss = SimLoss().to(device)
        self.encoder = MLP(
            in_channels=kwargs['hidden_channels'],
            hidden_channels=(kwargs['hidden_channels'] + kwargs['out_channels']) // 2,
            out_channels=kwargs['out_channels'],
            num_layers=2,
            norm='batch_norm',
            dropout=0.3,
        ).to(device)
        self.decoder = MLP(
            in_channels=kwargs['out_channels'],
            hidden_channels=(kwargs['hidden_channels'] + kwargs['out_channels']) // 2,
            out_channels=kwargs['hidden_channels'],
            num_layers=2,
            norm='batch_norm',
            dropout=0.3,
        ).to(device)
        # self.ae_loss = AELoss(embedding_dim=kwargs['out_channels']).to(device)

    def forward(self, batch_data):
        text_feats = self.text_model(batch_data.prompt)
        graph_feats = self.graph_model(batch_data)
        encoded_graph_feats = self.encoder(graph_feats)
        decoded_graph_feats = self.decoder(encoded_graph_feats)
        loss = mse_loss(graph_feats, decoded_graph_feats)
        loss += self.sim_loss(text_feats, encoded_graph_feats)
        return loss


def train(
        data_path: str, signature_path: str,
        model_args: dict, **kwargs
) -> torch.nn.Module:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('using cuda now....')

    # build dataset
    dataset = MultiModalTransactionDataset(
        root=data_path,
        signature_path=signature_path,
        transform=TransformSequence([
            # TextFeatEmbedding(),
            format_data_type,
            text_prompting,
        ]),
    )
    dsampler = DeduplicateBatchSampler(
        sampler=SequentialSampler(dataset),
        batch_size=kwargs.get('batch_size', 16),
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=dsampler,
        num_workers=kwargs.get('num_workers', 4),
    )

    # init models
    model = Model(**{
        "device": device,
        "metadata": dataset.metadata,
        **model_args,
    })
    text_emb = TextFeatEmbedding().to(device)
    if os.path.exists(kwargs['pretrain_path']):
        model.load_state_dict(
            state_dict=torch.load(kwargs['pretrain_path']) if torch.cuda.is_available()
            else torch.load(kwargs['pretrain_path'], map_location=torch.device('cpu')),
            strict=False,
        )

    # define loss and optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=kwargs.get('lr', 1e-3),
        weight_decay=kwargs.get('weight_decay', 5e-4),
    )

    # start training
    print('start training...')
    model.train()
    report_step = kwargs.get('report_step', 5)
    for epoch in range(kwargs.get('epoch')):
        total_loss, batch_cnt = 0, 0
        for batch_data in dataloader:
            optimizer.zero_grad()
            try:
                batch_data = batch_data.to(device)
                batch_data = text_emb(batch_data)
                loss = model(batch_data)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
            except OutOfMemoryError:
                print('Warning: out of memory at batch#%d, but you can ignore.' % batch_cnt)

            batch_cnt += 1
            if batch_cnt % report_step == 0:
                print('{}, epoch #{}, batch #{}, loss {}'.format(
                    datetime.datetime.now(), epoch,
                    batch_cnt, total_loss / report_step
                ))
                total_loss = 0
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--signature_path', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=384)
    parser.add_argument('--out_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--report_step', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pretrain_path', type=str, default='')
    args = parser.parse_args()

    torch.manual_seed(43)
    model = train(
        data_path=args.data_path,
        signature_path=args.signature_path,
        model_args=dict(
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ), **{
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epoch': args.epoch,
            'batch_size': args.batch_size,
            'report_step': args.report_step,
            'pretrain_path': args.pretrain_path,
            'num_workers': args.num_workers,
        }
    )

    save_path = os.path.join(PROJECT_PATH, 'model.pth')
    print('training process finished! save the model to: %s' % save_path)
    torch.save(model.state_dict(), save_path)
