import pathlib

from allennlp.common.testing.test_case import AllenNlpTestCase


class CeClcnnTestCase(AllenNlpTestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "ceclcnn"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
