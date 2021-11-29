from typing import Optional, Tuple

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.parallel import DdpAccelerator
from allennlp.nn.regularizers import RegularizerApplicator
from overrides import overrides
from torchtyping import TensorType

from ceclcnn.models.base import CharEncSeqModel
from ceclcnn.modules.character_encoders import CharacterEncoder
from ceclcnn.modules.seq2vec_encoders import AdaptiveCnnEncoder


@Model.register("ceclcnn")
class CECLCNN(CharEncSeqModel):
    def __init__(
        self,
        vocab: Vocabulary,
        character_encoder: CharacterEncoder,
        clcnn_model: AdaptiveCnnEncoder,
        wildcard_ratio: Optional[float] = None,
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
        ddp_accelerator: Optional[DdpAccelerator] = None,
    ) -> None:
        super().__init__(
            vocab,
            character_encoder,
            clcnn_model,
            regularizer=regularizer,
            serialization_dir=serialization_dir,
            ddp_accelerator=ddp_accelerator,
        )

        self.wildcard_training: Optional[nn.Dropout] = None
        if wildcard_ratio is not None:
            self.wildcard_training = nn.Dropout(p=wildcard_ratio)

    def clcnn_model(self, tokens: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return self.sequence_encoder(tokens, mask)

    def encode_character_images(
        self,
        char_imgs: TensorType["batch_size", "char_len", "char_w", "char_h"],  # type: ignore # NOQA: F82)
    ) -> TensorType["batch_size", "char_len", "encode_dim"]:  # type: ignore # NOQA: F821:
        return self.character_encoder(char_imgs)

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

        if self.wildcard_training is not None:
            char_embeds = self.wildcard_training(char_embeds)

        seq_embeds = self.clcnn_model(char_embeds, mask)
        logit = self.mlp(seq_embeds)

        if return_char_embeds:
            return logit, char_embeds
        else:
            return logit, None
