import pytest
from allennlp.common.params import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from ceclcnn.common.testing import CeClcnnTestCase
from ceclcnn.data.dataset_readers import WikiTitleDatasetReader
from ceclcnn.modules import CharacterImageProcessor

TEST_CASES = (
    (
        "NotoSansCJKjp-Regular.otf",
        "https://raw.githubusercontent.com/frederick0329/Wikipedia-Title-Dataset/master/category_list_ja.txt",
        "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/ja_train.txt",
        485862,
    ),
    (
        "NotoSansCJKsc-Regular.otf",
        "https://raw.githubusercontent.com/frederick0329/Wikipedia-Title-Dataset/master/category_list_zh.txt",
        "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/zh_simplified_train.txt",
        355843,
    ),
    (
        "NotoSansCJKtc-Regular.otf",
        "https://raw.githubusercontent.com/frederick0329/Wikipedia-Title-Dataset/master/category_list_zh.txt",
        "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/zh_traditional_train.txt",
        355843,
    ),
)


class TestWikiTitleDatasetReader(CeClcnnTestCase):
    @pytest.mark.parametrize(
        "font_name, category_list_url, dataset_url, expected_num_instances", TEST_CASES
    )
    def test_read(
        self,
        font_name: str,
        category_list_url: str,
        dataset_url: str,
        expected_num_instances: int,
    ):
        font_path = str(self.PROJECT_ROOT / "dataset" / "fonts" / font_name)
        character_tokenizer = CharacterTokenizer()
        character_image_processor = CharacterImageProcessor(
            font_name=font_path,
            font_size=36,
        )
        reader = WikiTitleDatasetReader(
            character_tokenizer=character_tokenizer,
            character_image_processor=character_image_processor,
            category_list_path=category_list_url,
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
        )
        instances = ensure_list(reader.read(file_path=dataset_url))
        assert len(instances) == expected_num_instances

    def test_instance(self):
        params = Params.from_file(
            self.FIXTURES_ROOT / "data" / "dataset_readers" / "wiki_title_ja.jsonnet"
        )
        reader = DatasetReader.from_params(params)
        assert isinstance(reader, WikiTitleDatasetReader)
