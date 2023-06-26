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
        return Polynomial(self[:(max_degree + 1)])

    # Computes the polynomial Q(x) such that self * P(x) is congruent to 1 modulo x ** max_degree
    def recip(self, max_degree: int = -1) -> 'Polynomial':
        # Source: http://people.csail.mit.edu/madhu/ST12/scribe/lect06.pdf
        # Time complexity: O(n log n), as compared to O(n^2) for traditional algorithms.
        if max_degree == -1:
            return self.recip(len(self))
        elif max_degree == 0 or len(self) == 0:
            raise ArithmeticError("You cannot have an inverse of nothing!")
        elif max_degree == 1:
            return Polynomial(1 / self[0])  # should throw error for zero.
        m: int = int(2 ** (np.ceil(np.log2(max_degree)) - 1))
        a: Polynomial = self.recip(m)  # q1
        h0: Polynomial = Polynomial(self[:m])  # p1, a polynomial with degree m
        h1: Polynomial = Polynomial(self[m:(2 * m)])  # p2, another polynomial with degree m
        c, _ = a.mul(h0).div_by_power(m)  # q2 = c = (a * h0) / x^m
        return (a - a.mul(h1.mul(a).add(c).mul_by_power(m))).trim(max_degree - 1)  # a = max_degree // 2

    def degree(self):
        return len(list(self)) - 1

    def iszero(self):
        return len(list(self)) == 1 and list(self)[0] == 0

    def __divmod__(self, divisor: 'Polynomial'):
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

    def __call__(self, x: Iterable[complex] | complex):
        r = 0
        if isinstance(x, complex) or isinstance(x, float) or isinstance(x, int):
            if len(self) == 0:
                return 0
            for c in reversed(self):
                r = (r * x) + c
            return r
        else:
            # Multiply polynomials
            # Pre-compute a list
            polys: list[list[Polynomial]] = [[]]
            for r in x:
                polys[0].append(Polynomial([-r, 1]))
            # Until there are two polynomials left.
            while len(polys[-1]) > 2:
                # Multiply two polynomials, keep the 3rd polynomial.
                lst = polys[-1]
                polys.append([])
                for i in range(len(lst) // 2):
                    polys[-1].append(lst[2 * i].mul(lst[2 * i + 1]))
                if len(lst) % 2 == 1:
                    polys[-1].append(lst[-1])

            return self.eval_multipoint(list(x), polys, 0)

    # Takes O(n log(n)^2) time to evaluate O(n) points, so it takes O(log(n)^2) amortized time per root.
    # This evaluation process takes in a point and
    def eval_multipoint(self, x, polys, idx):
        # If n is a power of 2, divide by 2.
        # Else, divide the rounded up version of n by 2.
        n = len(x)

        # Calculate m = 2^(ceil(log2 n)) / 2
        if n == 1:
            # Use single-point evaluation instead.
            return [self(x[0])]
        m = 2 ** ((n - 1).bit_length() - 1)
        p0 = self % polys[-1][2 * idx]
        r0 = p0.eval_multipoint(x[:m], polys[:-1], 2 * idx)
        p1 = self % polys[-1][2 * idx + 1]
        r1 = p1.eval_multipoint(x[m:], polys[:-1], 2 * idx + 1)

        results = r0 + r1
        return results


if __name__ == '__main__':
    # x = Polynomial([1, 2, 3])
    y = Polynomial([1, -2]).recip(11)
    print(y)
