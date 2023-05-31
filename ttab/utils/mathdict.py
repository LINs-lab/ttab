# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Union, Callable
from collections.abc import Mapping, ItemsView, ValuesView

C = Union[float, int]


class MathDict:
    def __init__(self, dictionary: Dict[Any, Union[int, float]]) -> None:
        self.dictionary = dictionary
        self.keys = set(dictionary.keys())

    def __str__(self) -> str:
        return f"MathDict({self.dictionary})"

    def __repr__(self) -> str:
        return f"MathDict({repr(self.dictionary)})"

    def map(self: "MathDict", mapfun: Mapping[Any, Union[int, float]]) -> "MathDict":
        new_dict = {}
        for key in self.keys:
            new_dict[key] = mapfun(self.dictionary[key])
        return MathDict(new_dict)

    def filter(self: "MathDict", condfun: Mapping[Any, bool]) -> "MathDict":
        new_dict = {}
        for key in self.keys:
            if condfun(key):
                new_dict[key] = self.dictionary[key]
        return MathDict(new_dict)

    def detach(self) -> None:
        for key in self.keys:
            self.dictionary[key] = self.dictionary[key].detach()

    def values(self) -> ValuesView:
        return self.dictionary.values()

    def items(self) -> ItemsView:
        return self.dictionary.items()


def _mathdict_binary_op(operation: Callable[[C, C], C]) -> MathDict:
    def op(self: MathDict, other: Union[MathDict, Dict]) -> MathDict:
        new_dict = {}
        if isinstance(other, MathDict):
            assert other.keys == self.keys
            for key in self.keys:
                new_dict[key] = operation(self.dictionary[key], other.dictionary[key])
        else:
            for key in self.keys:
                new_dict[key] = operation(self.dictionary[key], other)
        return MathDict(new_dict)

    return op


def _mathdict_map_op(
    operation: Callable[[ValuesView, List[Any], List[Any]], Any]
) -> MathDict:
    def op(self: MathDict, *args, **kwargs) -> MathDict:
        new_dict = {}
        for key in self.keys:
            new_dict[key] = operation(self.dictionary[key], args, kwargs)
        return MathDict(new_dict)

    return op


def _mathdict_binary_in_place_op(operation: Callable[[Dict, Any, C], None]) -> MathDict:
    def op(self: MathDict, other: Union[MathDict, Dict]) -> MathDict:
        if isinstance(other, MathDict):
            assert other.keys == self.keys
            for key in self.keys:
                operation(self.dictionary, key, other.dictionary[key])
        else:
            for key in self.keys:
                operation(self.dictionary, key, other)
        return self

    return op


def _iadd(dict: Dict, key: Any, b: C) -> None:
    dict[key] += b


def _isub(dict: Dict, key: Any, b: C) -> None:
    dict[key] -= b


def _imul(dict: Dict, key: Any, b: C) -> None:
    dict[key] *= b


def _itruediv(dict: Dict, key: Any, b: C) -> None:
    dict[key] /= b


def _ifloordiv(dict: Dict, key: Any, b: C) -> None:
    dict[key] //= b


MathDict.__add__ = _mathdict_binary_op(lambda a, b: a + b)
MathDict.__sub__ = _mathdict_binary_op(lambda a, b: a - b)
MathDict.__rsub__ = _mathdict_binary_op(lambda a, b: b - a)
MathDict.__mul__ = _mathdict_binary_op(lambda a, b: a * b)
MathDict.__rmul__ = _mathdict_binary_op(lambda a, b: a * b)
MathDict.__truediv__ = _mathdict_binary_op(lambda a, b: a / b)
MathDict.__floordiv__ = _mathdict_binary_op(lambda a, b: a // b)
MathDict.__getitem__ = _mathdict_map_op(
    lambda x, args, kwargs: x.__getitem__(*args, **kwargs)
)
MathDict.__iadd__ = _mathdict_binary_in_place_op(_iadd)
MathDict.__isub__ = _mathdict_binary_in_place_op(_isub)
MathDict.__imul__ = _mathdict_binary_in_place_op(_imul)
MathDict.__itruediv__ = _mathdict_binary_in_place_op(_itruediv)
MathDict.__ifloordiv__ = _mathdict_binary_in_place_op(_ifloordiv)
