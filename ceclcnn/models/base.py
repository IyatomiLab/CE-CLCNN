from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.parallel import DdpAccelerator
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, get_token_ids_from_text_field_tensors
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, Metric
from overrides import overrides
from torchtyping import TensorType

from ceclcnn.modules.character_encoders import CharacterEncoder


class ClassificationBaseModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        seq_enc_dim: int,
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
        ddp_accelerator: Optional[DdpAccelerator] = None,
    ) -> None:
        super().__init__(
            vocab,
            regularizer=regularizer,
            serialization_dir=serialization_dir,
            ddp_accelerator=ddp_accelerator,
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=seq_enc_dim,
                out_features=seq_enc_dim // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=seq_enc_dim // 2,
                out_features=seq_enc_dim // 2 // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=seq_enc_dim // 2 // 2,
                out_features=vocab.get_vocab_size("labels"),
            ),
        )

        self._loss = nn.CrossEntropyLoss()

        self._metrics: Dict[str, Metric] = {
            "acc1": CategoricalAccuracy(top_k=1),
            "_acc3": CategoricalAccuracy(top_k=3),
            "f1_macro": FBetaMeasure(average="macro"),
            "f1_micro": FBetaMeasure(average="micro"),
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        tmp_metric_scores = {k: v.get_metric(reset) for k, v in self._metrics.items()}

        metric_scores: Dict[str, float] = {}
        for k, v in tmp_metric_scores.items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    metric_scores[f"_{k}_{_k}"] = _v
            else:
                metric_scores[k] = v

        return metric_scores


class CharEncSeqModel(ClassificationBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        character_encoder: CharacterEncoder,
        sequence_encoder: Seq2VecEncoder,
        char_vocab_namespace: str = "chars",
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
        ddp_accelerator: Optional[DdpAccelerator] = None,
    ) -> None:
        super().__init__(
            vocab,
            regularizer=regularizer,
            serialization_dir=serialization_dir,
            ddp_accelerator=ddp_accelerator,
            seq_enc_dim=sequence_encoder.get_output_dim(),
        )

        self.character_encoder = character_encoder
        self.sequence_encoder = sequence_encoder

        self.char_vocab_namespace = char_vocab_namespace

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
        seq_embeds = self.sequence_encoder(char_embeds, mask)
        logit = self.mlp(seq_embeds)

        if return_char_embeds:
            return logit, char_embeds
        else:
            return logit, None

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        batch_char_ids = get_token_ids_from_text_field_tensors(
            text_field_tensors=output_dict["tokens"]  # type:ignore
        )

        batch_chars = []
        for i in range(len(batch_char_ids)):
            char_ids = batch_char_ids[i].detach().cpu().numpy().tolist()
            chars = [
                self.vocab.get_token_from_index(c, namespace=self.char_vocab_namespace)
                for c in char_ids
            ]
            chars = list(filter(lambda x: x != DEFAULT_PADDING_TOKEN, chars))
            batch_chars.append(chars)

        output_dict["tokens"] = batch_chars  # type: ignore

        return output_dict

    @overrides
    def forward(
        self,
        char_imgs: TensorType["batch_size", "char_len", "char_w", "char_h"],  # type: ignore # NOQA: F821
        char_text: TextFieldTensors,
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(char_text)
        logits, char_embeds = self.predict(char_imgs, mask, return_char_embeds=True)

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

            output_dict["labels"] = label

        return output_dict
