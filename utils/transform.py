from typing import Callable, List

import torch.cuda
from torch_geometric.data import HeteroData
from transformers import RobertaModel, RobertaTokenizer

from settings import HUGGING_MODEL_PATH


class TransformSequence:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: HeteroData, *args, **kwargs):
        for transform in self.transforms:
            data = transform(data)
        return data


def text_prompting(data: HeteroData) -> HeteroData:
    item = eval(data.label)
    data.prompt = 'A function means `{}` has been triggered. ' \
                  'More descriptions of this function ' \
                  'are as follows: {}'.format(
        item['func_name'] if item.get('func_name', '') != '' else '?',
        item['details'] if item.get('details', '') != '' else 'null',
    )
    return data


def format_data_type(data: HeteroData) -> HeteroData:
    for nt in data.node_types:
        data[nt].x = data[nt].x.float()
    for et in data.edge_types:
        data[et].edge_index = data[et].edge_index.long()
        data[et].edge_attr = data[et].edge_attr.float()
    return data


class TextFeatEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._tokenizer = RobertaTokenizer.from_pretrained(
            '{}/codebert-base'.format(HUGGING_MODEL_PATH)
            if HUGGING_MODEL_PATH is not None
            else 'microsoft/codebert-base'
        )
        self._emb_model = RobertaModel.from_pretrained(
            '{}/codebert-base'.format(HUGGING_MODEL_PATH)
            if HUGGING_MODEL_PATH is not None
            else 'microsoft/codebert-base'
        ).embeddings

    def _embed_node_text(self, data: HeteroData) -> HeteroData:
        for node_type in data.node_types:
            x = data[node_type].x
            if x.shape[1] < self._tokenizer.model_max_length:
                continue
            truncated_token_length = 16
            idx_from = -self._tokenizer.model_max_length
            idx_to = -self._tokenizer.model_max_length + truncated_token_length
            text_feats = x[:, idx_from: idx_to]
            text_feats = text_feats.type(dtype=torch.int64)
            attn_masks = (text_feats == self._tokenizer.pad_token_id)

            with torch.no_grad():
                batch_size, seq_length = text_feats.size()
                buffered_token_type_ids = self._emb_model.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
                embedding_output = self._emb_model(
                    input_ids=text_feats,
                    position_ids=None,
                    token_type_ids=token_type_ids,
                    inputs_embeds=None,
                    past_key_values_length=0,
                )

                # mul to set the pad token vec as 0
                mask = (~attn_masks).unsqueeze(-1).expand(embedding_output.size()).float()
                embedding_output = embedding_output * mask

                # sum and avg
                sum_embeddings = torch.sum(embedding_output, dim=1)
                non_padding_tokens = torch.sum(mask, dim=1)
                embedding_output = sum_embeddings / non_padding_tokens

            data[node_type].x = torch.cat([
                x[:, :-self._tokenizer.model_max_length],
                embedding_output,
            ], dim=1)
        return data

    def _embed_edge_text(self, data: HeteroData) -> HeteroData:
        for edge_type in data.edge_types:
            edge_attr = data[edge_type].edge_attr
            if edge_attr.shape[1] < self._tokenizer.model_max_length:
                continue
            truncated_token_length = 32
            idx_from = -self._tokenizer.model_max_length
            idx_to = -self._tokenizer.model_max_length + truncated_token_length
            text_feats = edge_attr[:, idx_from: idx_to]
            text_feats = text_feats.type(dtype=torch.int64)
            attn_masks = (text_feats == self._tokenizer.pad_token_id)

            with torch.no_grad():
                batch_size, seq_length = text_feats.size()
                buffered_token_type_ids = self._emb_model.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
                embedding_output = self._emb_model(
                    input_ids=text_feats,
                    position_ids=None,
                    token_type_ids=token_type_ids,
                    inputs_embeds=None,
                    past_key_values_length=0,
                )
                # mul to set the pad token vec as 0
                mask = (~attn_masks).unsqueeze(-1).expand(embedding_output.size()).float()
                embedding_output = embedding_output * mask

                # sum and avg
                sum_embeddings = torch.sum(embedding_output, dim=1)
                non_padding_tokens = torch.sum(mask, dim=1)
                embedding_output = sum_embeddings / non_padding_tokens

            data[edge_type].edge_attr = torch.cat([
                edge_attr[:, :-self._tokenizer.model_max_length],
                embedding_output,
            ], dim=1)
        return data

    def forward(self, data: HeteroData, *args, **kwargs):
        data = self._embed_node_text(data)
        data = self._embed_edge_text(data)
        return data
