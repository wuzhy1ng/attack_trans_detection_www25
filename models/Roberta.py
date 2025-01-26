import torch
from transformers import RobertaTokenizer, RobertaModel

from settings import HUGGING_MODEL_PATH


class RoBERTa(torch.nn.Module):
    def __init__(self, out_channels: int, **kwargs):
        super().__init__()
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(
            '{}/codebert-base'.format(HUGGING_MODEL_PATH)
            if HUGGING_MODEL_PATH is not None
            else 'microsoft/codebert-base'
        )
        self.roberta_encoder = RobertaModel.from_pretrained(
            '{}/codebert-base'.format(HUGGING_MODEL_PATH)
            if HUGGING_MODEL_PATH is not None
            else 'microsoft/codebert-base'
        )
        freeze_layers = len(self.roberta_encoder.encoder.layer)
        for params in self.roberta_encoder.encoder.layer[:freeze_layers].parameters():
            params.requires_grad = False

        self.out_lin = torch.nn.Linear(768, out_channels)

    def forward(self, text_input):
        text_features = self.roberta_tokenizer(
            text_input, return_tensors='pt', padding=True,
            truncation=True, max_length=self.roberta_tokenizer.model_max_length,
        ).to(self.roberta_encoder.device)

        with torch.no_grad():
            input_ids, attention_mask = text_features['input_ids'], text_features['attention_mask']
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)

            if hasattr(self.roberta_encoder.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.roberta_encoder.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask: torch.Tensor = self.roberta_encoder.get_extended_attention_mask(attention_mask, input_shape)

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
            # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.roberta_encoder.get_head_mask(None, self.roberta_encoder.config.num_hidden_layers)

            embedding_output = self.roberta_encoder.embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )
            encoder_outputs = self.roberta_encoder.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=self.roberta_encoder.config.output_attentions,
                output_hidden_states=self.roberta_encoder.config.output_hidden_states,
                return_dict=self.roberta_encoder.config.use_return_dict,
            )
            sequence_output = encoder_outputs[0]

        pooled_output = self.roberta_encoder.pooler(sequence_output)
        return self.out_lin(pooled_output)
