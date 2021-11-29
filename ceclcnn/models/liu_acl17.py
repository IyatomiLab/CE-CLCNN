from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import GruSeq2VecEncoder
from allennlp.nn.parallel import DdpAccelerator
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, Metric
from overrides import overrides
from torchtyping import TensorType

from ceclcnn.models.base import CharEncSeqModel, ClassificationBaseModel
from ceclcnn.modules.character_encoders import CharacterEncoder


@Model.register("liu_acl17_visual")
class LiuVisual(CharEncSeqModel):
    def __init__(
        self,
        vocab: Vocabulary,
        character_encoder: CharacterEncoder,
        rnn_encoder: GruSeq2VecEncoder,
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
        ddp_accelerator: Optional[DdpAccelerator] = None,
    ) -> None:
        super().__init__(
            vocab,
            character_encoder,
            rnn_encoder,
            regularizer=regularizer,
            serialization_dir=serialization_dir,
            ddp_accelerator=ddp_accelerator,
        )

    def gru_model(self, tokens: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return self.sequence_encoder(tokens, mask)

    @overrides
    def predict(
        self,
        char_imgs: TensorType["batch_size", "char_len", "char_w", "char_h"],  # type: ignore # NOQA: F821
        mask: TensorType["batch_size"],  # type: ignore # NOQA: F821
        return_char_embeds: bool = False,
    ) -> Tuple[  # type: ignore # NOQA: F821
        TensorType["batch_size", "num_class"],  # type: ignore # NOQA: F821
        Optional[TensorType["batch_size", "char_len", "encode_dim"]],  # type: ignore # NOQA: F821
    ]:

        char_embeds = self.character_encoder(char_imgs)
        seq_embeds = self.gru_model(char_embeds, mask)
        logit = self.mlp(seq_embeds)

        if return_char_embeds:
            return logit, char_embeds
        else:
            return logit, None


@Model.register("liu_acl17_lookup")
class LiuLookup(ClassificationBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        lookup_encoder: TextFieldEmbedder,
        rnn_encoder: GruSeq2VecEncoder,
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
        ddp_accelerator: Optional[DdpAccelerator] = None,
    ) -> None:
        super().__init__(
            vocab,
            regularizer=regularizer,
            serialization_dir=serialization_dir,
            ddp_accelerator=ddp_accelerator,
            seq_enc_dim=rnn_encoder.get_output_dim(),
        )

        self.lookup_encoder = lookup_encoder
        self.rnn_encoder = rnn_encoder

    def predict(
        self,
        char_text: TextFieldTensors,
        mask: torch.BoolTensor,
        return_char_embeds: bool = False,
    ) -> Tuple[  # type: ignore # NOQA: F821
        TensorType["batch_size", "num_class"],  # type: ignore # NOQA: F821
        Optional[TensorType["batch_size", "char_len", "encode_dim"]],  # type: ignore # NOQA: F821
    ]:

        char_embeds = self.lookup_encoder(char_text)
        seq_embeds = self.rnn_encoder(char_embeds, mask)
        logit = self.mlp(seq_embeds)

        if return_char_embeds:
            return logit, char_embeds
        else:
            return logit, None

    @overrides
    def forward(
        self,
        char_text: TextFieldTensors,
        label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(char_text)
        logits, char_embeds = self.predict(char_text, mask)

        output_dict: Dict[str, torch.Tensor] = {
            "logits": logits,
            "tokens": char_text,  # type: ignore
            "embeds": char_embeds,  # type: ignore
        }

        if label is not None:
            loss = self._loss(logits, label)
            output_dict["loss"] = loss

            for metric_func in self._metrics.values():
                metric_func(logits, label)  # type: ignore

        return output_dict
