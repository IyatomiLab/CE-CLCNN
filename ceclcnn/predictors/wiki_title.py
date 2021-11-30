from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor


@Predictor.register("wiki_title")
class WikiTitlePredictor(Predictor):
    def __init__(
        self, model: Model, dataset_reader: DatasetReader, frozen: bool = True
    ) -> None:
        super().__init__(model, dataset_reader, frozen=frozen)
