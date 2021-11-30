from typing import Optional, Tuple

import torch
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import min_value_of_dtype
from overrides import overrides


@Seq2VecEncoder.register("cnn_adaptive")
class AdaptiveCnnEncoder(CnnEncoder):
    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
        conv_layer_activation: Activation = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__(
            embedding_dim,
            num_filters,
            ngram_filter_sizes=ngram_filter_sizes,
            conv_layer_activation=conv_layer_activation,
            output_dim=output_dim,
        )

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)
        else:
            # If mask doesn't exist create one of shape (batch_size, num_tokens)
            mask = torch.ones(
                tokens.shape[0], tokens.shape[1], device=tokens.device
            ).bool()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # masking, then do max pooling over each filter for the whole input sequence.
        # Because our max pooling is simple, we just use `torch.max`.  The resultant tensor has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        # To ensure the cnn_encoder respects masking we add a large negative value to
        # the activations of all filters that convolved over a masked token. We do this by
        # first enumerating all filters for a given convolution size (torch.arange())
        # then by comparing it to an index of the last filter that does not involve a masked
        # token (.ge()) and finally adjusting dimensions to allow for addition and multiplying
        # by a large negative value (.unsqueeze())
        filter_outputs = []
        batch_size = tokens.shape[0]
        # shape: (batch_size, 1)
        last_unmasked_tokens = mask.sum(dim=1).unsqueeze(dim=-1)
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            pool_length = tokens.shape[2] - convolution_layer.kernel_size[0] + 1

            # Forward pass of the convolutions.
            # shape: (batch_size, num_filters, pool_length)
            try:
                activations = self._activation(convolution_layer(tokens))

                # Create activation mask.
                # shape: (batch_size, pool_length)
                indices = (
                    torch.arange(pool_length, device=activations.device)
                    .unsqueeze(0)
                    .expand(batch_size, pool_length)
                )
                # shape: (batch_size, pool_length)
                activations_mask = indices.ge(
                    last_unmasked_tokens - convolution_layer.kernel_size[0] + 1
                )
                # shape: (batch_size, num_filters, pool_length)
                activations_mask = activations_mask.unsqueeze(1).expand_as(activations)

                # Replace masked out values with smallest possible value of the dtype so
                # that max pooling will ignore these activations.
                # shape: (batch_size, pool_length)
                activations = activations + (
                    activations_mask * min_value_of_dtype(activations.dtype)
                )
            except RuntimeError:
                activations = torch.zeros(
                    (tokens.size(0), self._num_filters, 1),
                    device=tokens.device,
                    requires_grad=True,
                )

            # Pick out the max filters
            filter_outputs.append(activations.max(dim=2)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = (
            torch.cat(filter_outputs, dim=1)
            if len(filter_outputs) > 1
            else filter_outputs[0]
        )

        # Replace the maxpool activations that picked up the masks with 0s
        maxpool_output[maxpool_output == min_value_of_dtype(maxpool_output.dtype)] = 0.0

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result
