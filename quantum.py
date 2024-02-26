from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from sympy import I, Symbol, Rational, im, re, simplify, sqrt


class Types:
    Bra = 'Bra'
    Ket = 'Ket'


class State:
    multiplier: list
    state: list[list]
    type: str

    def __init__(self, state, multiplier: int | float | Symbol | list = 1, type: str = Types.Bra):
        if isinstance(state[0], list):
            self.state = state
        else:
            self.state = [state]
        if isinstance(multiplier, list):
            self.multiplier = multiplier
        else:
            self.multiplier = [multiplier]

        self.type = type

    def __add__(self, other):
        if other.type != self.type:
            raise TypeError(f'{other.type} state cannot be added to {self.type}')
        if isinstance(other, State):
            new_state = self.state.copy()
            new_state.extend(other.state)
            new_multiplier = self.multiplier.copy()
            new_multiplier.extend(other.multiplier)
            return State(new_state, new_multiplier, self.type)

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, State):
            if other.type == self.type:
                raise TypeError(f'{other.type} state cannot be multiplied by {self.type}')
            answer = 0
            for i in range(len(self.state)):
                for j in range(len(other.state)):
                    if self.state[i] == other.state[j]:
                        answer += self.multiplier[i] * other.multiplier[j]
            return answer
        elif isinstance(other, int | float | Symbol):
            return State(self.state, [mul * other for mul in self.multiplier], self.type)

    __rmul__ = __mul__

    def __str__(self):
        if self.type == Types.Bra:
            return ' + '.join([f'{self.multiplier[index]}<{state}|' for index, state in enumerate(self.state)])
        elif self.type == Types.Ket:
            return ' + '.join([f'{self.multiplier[index]}|{state}>' for index, state in enumerate(self.state)])


class OrbitalOperator:
    multiplier: float

    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def __mul__(self, other: State | int | float | Symbol):
        if isinstance(other, State):
            return State(other.state, [mul * other.state[index][0] * self.multiplier for index, mul in enumerate(other.multiplier)], other.type)
        elif isinstance(other, int | float | Symbol):
            return self.__init__(multiplier=other)

    def __rmul__(self, other: int | float | Symbol):
        return SpinOperator(multiplier=other)

    def __add__(self, other):
        pass


class SpinOperator:
    multiplier: float

    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def __mul__(self, other: State | int | float | Symbol):
        if isinstance(other, State):
            return State(other.state, [mul * other.state[index][1] * self.multiplier for index, mul in enumerate(other.multiplier)], other.type)
        elif isinstance(other, int | float | Symbol):
            return self.__init__(multiplier=other)

    def __rmul__(self, other: int | float | Symbol):
        return self.__init__(multiplier=other)

    def __add__(self, other):
        pass


if __name__ == '__main__':
    a = Symbol('a')
    b = Symbol('b')

    a_imb = re(a) - im(a) * I
    b_imb = re(b) - im(b) * I
    a = re(a) + im(a) * I
    b = re(b) + im(b) * I

    state1 = State([1, 0.5], a, type=Types.Ket)
    state2 = State([3, -0.5], b, type=Types.Ket)
    state3 = State([1, 0.5], a_imb, type=Types.Bra)
    state4 = State([3, -0.5], b_imb, type=Types.Bra)
    print(simplify((state1 + OrbitalOperator()*state2) * state4))
