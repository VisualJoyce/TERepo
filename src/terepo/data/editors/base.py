import collections
from enum import unique, Enum


@unique
class OpCodes(Enum):
    # 为序列值指定value值
    replace = 'replace'
    delete = 'delete'
    insert = 'insert'
    equal = 'equal'


class Operations(object):

    def __init__(self, start, end, operations):
        self.start = start
        self.end = end
        self.operations = operations if isinstance(operations, list) else [operations]

    def __repr__(self):
        return f'start={self.start}, end={self.end}, operations={self.operations}'


class Alignment(object):

    def __init__(self, action, target, shift):
        self.action = action
        self.target = target
        self.shift = shift


def load_vocab(vocab_file, index_offset=0):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index + index_offset
    return vocab
