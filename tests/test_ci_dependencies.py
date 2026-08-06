import importlib


def test_import_dotenv():
    module = importlib.import_module('dotenv')
    assert module is not None


def test_import_flashtext_and_gliner():
    keyword_processor = importlib.import_module('flashtext').KeywordProcessor
    gliner_module = importlib.import_module('gliner')
    assert keyword_processor is not None
    assert gliner_module is not None
