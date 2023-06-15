import itertools
from typing import Iterable

import numpy as np
import scipy.signal as sps


def normalize(x):
    k = 1
    while x[-k] == 0 and k < len(x):
        k += 1
    if k > 1:
        del x[-(k - 1):]
    return x


class Polynomial(list[complex]):

    def __init__(self, coeffs: Iterable[complex] | complex | float | int = None):
        super().__init__()
        if coeffs is None:
            coeffs = [0]
        if isinstance(coeffs, Iterable):
            lst = list(coeffs)
            if len(lst) == 0:
                lst = [0]
            else:
                lst = normalize(lst)
            self.extend(lst)
        elif isinstance(coeffs, float) or isinstance(coeffs, int) or isinstance(coeffs, complex):
            self.append(coeffs)

    def mul(self, other: Iterable[float] | float | 'Polynomial') -> 'Polynomial':
        if isinstance(other, float):
            return Polynomial([other * e for e in self])
        arr = sps.convolve(self, other)
        return Polynomial(arr)

    def add(self, other):
        if isinstance(other, Polynomial):
            x = [a + b for a, b in itertools.zip_longest(self, other, fillvalue=0)]
            return Polynomial(x)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
            return Polynomial([other + self[0]] + self[1:])
        return None

    def __sub__(self, other):
        if isinstance(other, Polynomial):
            x = [a - b for a, b in itertools.zip_longest(self, other, fillvalue=0)]
            x = normalize(x)
            return Polynomial(x)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
            return Polynomial([self[0] - other] + self[1:])
        return None

    def __neg__(self):
        return Polynomial([-x for x in self])

    def mul_by_power(self, power: int):
        if len(self) == 1 and self[0] == 0 and power > 0:
            return [0]
        if power < 0:
            return self.div_by_power(power)
        return Polynomial(np.append([0] * power, self))

    def div_by_power(self, power: int):
        if power < 0:
            return self.mul_by_power(-power)
        if power < len(self):
            x = self[power:]
            y = self[:power]
            return Polynomial(x), Polynomial(y)
        return Polynomial(0), Polynomial(self)

    def trim(self, max_degree):
        return Polynomial(self[:(max_degree+1)])

    # Computes the polynomial Q(x) such that self * P(x) is congruent to 1 modulo x ** max_degree
    def recip(self, max_degree: int = -1) -> 'Polynomial':
        # Source: https://www.csa.iisc.ac.in/~chandan/courses/CNT/notes/lec6.pdf
        # Time complexity: O(n log n), as compared to O(n^2) for traditional algorithms.
        if max_degree == -1:
            return self.recip(len(self))
        elif max_degree == 0 or len(self) == 0:
            raise ArithmeticError("You cannot have an inverse of nothing!")
        elif max_degree == 1:
            return Polynomial(1 / self[0])  # should throw error for zero.
        m: int = int(2 ** (np.ceil(np.log2(max_degree)) - 1))
        a: Polynomial = self.recip(m)
        h0: Polynomial = Polynomial(self[:m])
        h1: Polynomial = Polynomial(self[m:(2 * m)])  # a polynomial with degree k // 2
        c, _ = a.mul(h0).div_by_power(m)
        return (a - a.mul(h1.mul(a).add(c).mul_by_power(m))).trim(max_degree)

    def degree(self):
        return len(list(self)) - 1

    def iszero(self):
        return len(list(self)) == 1 and list(self)[0] == 0

    def __divmod__(self, divisor):
        if divisor.iszero():
            raise ArithmeticError("/ by 0")
        quotient: Polynomial
        self.reverse()
        divisor.reverse()

        # Compute Rev(q) = Rev(f) Rev(g)^-1

        deg = self.degree() - divisor.degree()

        quotient = self.mul(divisor.recip(self.degree())).trim(deg)

        # The degree of the quotient must be always equal to self.degree() - divisor.degree()
        quotient.extend([0] * max(0, deg - len(quotient)))
        quotient.reverse()
        self.reverse()
        divisor.reverse()
        remainder = self - quotient.mul(divisor)
        return quotient, remainder

    def derivative(self):
        if len(self) < 2:
            return 0
        return [p * x for p, x in enumerate(self[1:], 1)]

    def __truediv__(self, divisor):
        return divmod(self, divisor)[0]

    def __mod__(self, divisor):
        return divmod(self, divisor)[1]


if __name__ == '__main__':
    x = Polynomial([-2, 0, 1, 1, -1, 1, -1, 1])
    y = Polynomial([-2, 0, 1])

    print(y.derivative())
