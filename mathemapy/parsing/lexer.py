from tokenize import tokenize
from io import BytesIO

def tokenizer(source: str):
    return list(tokenize(BytesIO(source.encode('utf-8')).readline))