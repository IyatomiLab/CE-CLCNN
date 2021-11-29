from typing import Dict, Iterable, List, Optional

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TensorField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from ceclcnn.modules.character_image_processor import CharacterImageProcessor
from ceclcnn.modules.random_erasing import RandomErasing
from overrides import overrides


@DatasetReader.register("wiki_title")
class WikiTitleDatasetReader(DatasetReader):
    def __init__(
        self,
        character_tokenizer: CharacterTokenizer,
        category_list_path: str,
        character_image_processor: Optional[CharacterImageProcessor] = None,
        random_erasing: Optional[RandomErasing] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multiprocess_sharding: bool = False,
        serialization_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            max_instances=max_instances,
            manual_distributed_sharding=manual_distributed_sharding,
            manual_multiprocess_sharding=manual_multiprocess_sharding,
            serialization_dir=serialization_dir,
        )
        self._character_tokenizer = character_tokenizer
        self._character_image_processor = character_image_processor
        self._random_erasing = random_erasing

        self._category_list = self.load_category_list(category_list_path)
        self._token_indexers = token_indexers or {
            "chars": SingleIdTokenIndexer(namespace="chars")
        }

    def load_category_list(self, category_list_path: str) -> List[str]:
        with open(cached_path(category_list_path), "r") as rf:
            category_list = [line.strip() for line in rf]
        return category_list

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as rf:
            for line in self.shard_iterable(rf):
                label, title = line.strip().split("\t")
                yield self.text_to_instance(title=title, label=label)

    @overrides
    def text_to_instance(self, title: str, label: Optional[str] = None) -> Instance:

        fields: Dict[str, Field] = {}

        char_title_tokens = self._character_tokenizer.tokenize(title)
        fields["char_text"] = TextField(char_title_tokens)

        if self._character_image_processor is not None:
            char_title_list = list(map(lambda x: x.text, char_title_tokens))
            char_image_title = self._character_image_processor(char_title_list)

            if self._random_erasing is not None:
                char_image_title = self._random_erasing.apply_augmentations(
                    xs=char_image_title
                )

            fields["char_imgs"] = TensorField(char_image_title)

        if label is not None:
            fields["label"] = LabelField(self._category_list[int(label)])

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["char_text"].token_indexers = self._token_indexers  # type: ignore
