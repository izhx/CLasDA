"""
General mismatched transformer embedder.
"""
import inspect
from typing import Optional

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.nn import util


@TokenEmbedder.register("transformer_mismatched")
class TransformerMismatchedEmbedder(TokenEmbedder):
    """
    Modified from `PretrainedTransformerMismatchedEmbedder`.

    Use this embedder to embed wordpieces given by `matched_embedder` in
    transformer-style and to get word-level representations.

    Registered as a `TokenEmbedder` with name "transformer_mismatched".

    # Parameters

    matched_embedder: `TokenEmbedder`
        The matched *transforer embedder.

    """

    def __init__(
        self,
        matched_embedder: TokenEmbedder,
        sub_token_mode: Optional[str] = "avg"
    ) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self.matched_embedder = matched_embedder
        self.sub_token_mode = sub_token_mode

    def get_output_dim(self):
        return self.matched_embedder.get_output_dim()

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        domain: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedTransformerEmbedder`.
        domain: `Optional[torch.LongTensor]`
            See `PgnAdapterTransformerEmbedder`.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # hack for pass `domain` into `PgnAdapterTransformerEmbedder`.
        forward_params = inspect.signature(self.matched_embedder.forward).parameters
        forward_params_values = dict(type_ids=type_ids, segment_concat_mask=segment_concat_mask)
        if "domain" in forward_params:
            forward_params_values.update(domain=domain)

        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self.matched_embedder(
            token_ids, wordpiece_mask, **forward_params_values
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)

        span_mask = span_mask.unsqueeze(-1)

        # Shape: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        span_embeddings *= span_mask  # zero out paddings

        # If "sub_token_mode" is set to "first", return the first sub-token embedding
        if self.sub_token_mode == "first":
            # Select first sub-token embeddings from span embeddings
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            orig_embeddings = span_embeddings[:, :, 0, :]

        # If "sub_token_mode" is set to "avg", return the average of embeddings of all sub-tokens of a word
        elif self.sub_token_mode == "avg":
            # Sum over embeddings of all sub-tokens of a word
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            span_embeddings_sum = span_embeddings.sum(2)

            # Shape (batch_size, num_orig_tokens)
            span_embeddings_len = span_mask.sum(2)

            # Find the average of sub-tokens embeddings by dividing `span_embedding_sum` by `span_embedding_len`
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

            # All the places where the span length is zero, write in zeros.
            orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        # If invalid "sub_token_mode" is provided, throw error
        else:
            raise ConfigurationError(f"Do not recognise 'sub_token_mode' {self.sub_token_mode}")

        return orig_embeddings
