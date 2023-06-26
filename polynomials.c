#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

typedef struct polynomial polynomial;

#define EPSILON 1e-12
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

struct polynomial {
    complex *coeffs;
    int degree;
};

void arr_deep_copy(complex *, complex *, size_t);
void display_array(complex *, size_t, size_t);
void display_polynomial(polynomial, size_t);
void fft(complex *, size_t, int);
void fft_it(complex *, size_t, int);
unsigned int log2_int(unsigned int);

polynomial padd(polynomial, polynomial);
void padd_ip(polynomial *, polynomial);

polynomial *pdiv(polynomial, polynomial);
polynomial *pdiv_bp(polynomial, size_t);

complex peval(polynomial, complex);
complex *pevalm(polynomial, complex *, size_t);
void pevalmr(complex *, polynomial *, polynomial **, complex *, unsigned int, int, int);
polynomial pmul(polynomial, polynomial);
void pmul_ip(polynomial *, polynomial);
polynomial pmul_bp(polynomial, size_t);
void pmul_bp_ip(polynomial *, size_t);


polynomial pneg(polynomial);
void pneg_ip(polynomial *);

polynomial prev(polynomial);
void prev_ip(polynomial *);

polynomial psub(polynomial p, polynomial q);
void psub_ip(polynomial *p, polynomial q);
void psubr_ip(polynomial *p, polynomial q);

void pset(polynomial *, complex *, size_t);
void pset_p(polynomial *, polynomial);

void ps_zero(polynomial *);

unsigned int reverse(unsigned int, unsigned int);
unsigned int round_up(unsigned int);

// Deep copys from an array to an array.
void arr_deep_copy(complex *output, complex *input, size_t size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i];
    }
}

// Displays an array.
void display_array(complex *arr, size_t size, size_t precision) {
    int i;
    double ci;

    char* format_string = malloc(35);
    printf("[");

    strcpy(format_string, "%%.%df%%+.%df*i     ");
    sprintf(format_string, format_string, precision, precision);
    for (i = 0; i < size - 1; i++) {
        printf(format_string, creal(arr[i]), cimag(arr[i]));
    }

    strcpy(format_string, "%%.%df%%+.%df*i");
    sprintf(format_string, format_string, precision, precision);
    printf(format_string, creal(arr[i]), cimag(arr[i]));
    printf("]\n");
}

// Displays the coefficients of the polynomial, little endian.
void display_polynomial(polynomial p, size_t precision) {
    display_array(p.coeffs, p.degree + 1, precision);
}

// A recursive implmentation of the FFT algorithm.
void fft(complex *output, size_t size, int inverse) {
    // if size is not a power of two:
    if (size & (size - 1)) {
        fprintf(stderr, "Size must be a power of two.");
        exit(1);
    }

    int idx, new_size;
    complex *odds, *evens;
    if (size > 1) {
        new_size = size / 2;

        odds = malloc(new_size * sizeof(complex));
        evens = malloc(new_size * sizeof(complex));
        if (odds == NULL || evens == NULL) {
            fprintf(stderr, "Cannot allocate memory for the Fourier transform.");
            exit(1);
        }

        for (idx = 0; idx < new_size; idx++) {
            evens[idx] = output[2*idx];
            odds[idx] = output[2*idx + 1];
        }

        fft(evens, new_size, inverse);
        fft(odds, new_size, inverse);


        for (idx = 0; idx < new_size; idx++) {
            complex omega = cexp(2 * M_PI * I * idx * (inverse ? 1 : -1) / size),
            p, q;
            p = evens[idx];
            q = odds[idx] * omega;
            output[idx] = p + q;
            output[new_size + idx] = p - q;
            if (inverse) {
                output[idx] /= 2;
                output[new_size + idx] /= 2;
            }
        }
    }
}

// An iterative implementation of the FFT algorithm.
void fft_it(complex *output, size_t size, int inverse) {
    int i, j, k, l, r;
    complex temp, u, v, w, wl;
    
    k = log2_int(size);
    for (i = 0; i < size; i++) {
        r = reverse(i, k);
        if (i < r) {
            temp = output[i];
            output[i] = output[r];
            output[r] = temp;
        }
    }
    
    for (l = 2; l <= size; l <<= 1) {
        wl = cexp(2 * M_PI * I * (inverse ? 1 : -1) / l);
        for (i = 0; i < size; i += l) {
            w = 1;
            for (j = 0; j < l / 2; j++) {
                u = output[i + j];
                v = output[i + j + l/2] * w;
                output[i + j] = u + v;
                output[i + j + l/2] = u - v;
                w *= wl;
            }
        }
    }
    
    if (inverse) {
        for (i = 0; i < size; i++) {
            output[i] /= size;
        }
    }
}

unsigned int log2_int(unsigned int x) {
    int i, r = 0, t = x;
    
    for (i = 4 * sizeof(int); t && i; i >>= 1) {
        if (t >> i) {
            t >>= i;
            r += i;
        }
    }
    
    return r;
}

// Returns p + q.
polynomial padd(polynomial p, polynomial q) {
    complex *result, *curr, *pc, *qc;
    int idx, size, pd, qd;
    polynomial r;
    
    pd = p.degree;
    qd = q.degree;
    
    // Ensure p has degree >= q.
    if (qd > pd) {
        return padd(q, p);
    }
    
    pc = p.coeffs;
    qc = q.coeffs;
    
    // Since p has degree >= q, it would make sense to add coeffs of q to p.
    
    result = malloc((1 + pd) * sizeof(complex));
    if (result == NULL) {
        fprintf(stderr, "Cannot allocate memory to result.");
        exit(1);
    }
    for (idx = 0, curr = pc; idx <= pd; curr++, idx++) {
        result[idx] += *curr;
    }
    for (idx = 0, curr = qc; idx <= qd; curr++, idx++) {
        result[idx] += *curr;
    }
    pset(&r, result, 1 + pd);
    return r;
}

// Adds p to q and assigns the result to p.
void padd_ip(polynomial *p, polynomial q) {
    int pd, qd, td, idx, rd;
    complex *pc, *qc, *curr;
    
    pd = p->degree;
    pc = p->coeffs;
    qd = q.degree;
    qc = q.coeffs;
    
    if (pd < qd) {
        pc = realloc(pc, (1 + qd) * sizeof(complex));
        if (pc == NULL) {
            fprintf(stderr, "Cannot reallocate memory for the new coefficients.");
            exit(1);
        }
        p->degree = qd;
    }
    for (idx = 0, curr = qc; idx <= qd; curr++, idx++) {
        pc[idx] += *curr;
    }
    // Trim trailing zeros
    curr--;
    while (p->degree && cabs(*curr--) < EPSILON) {
        (p->degree)--;
    }
}

// Compute p / q and p % q. This algorithm uses polynomial reciprocals and multiplication to calculate
// the quotient and modulus in O(n log n) time, as opposed to worst case O(n^2) time for the schoolbook algorithm.
// Source: http://people.csail.mit.edu/madhu/ST12/scribe/lect06.pdf
polynomial *pdiv(polynomial p, polynomial q) {
    polynomial *result;
    complex *pc, *qc;
    int pd, qd, dd, degree;
    
    pc = p.coeffs;
    qc = q.coeffs;
    pd = p.degree;
    qd = q.degree;

    result = malloc(2 * sizeof(polynomial));
    if (pd < qd) {
        ps_zero(result);
        pset_p(result + 1, q);
        return result;
    }
    
    prev_ip(&p);
    prev_ip(&q);
    degree = pd - qd;
    if (result == NULL) {
        fprintf(stderr, "Cannot allocate memory to quotient and remainder.\n");
        exit(1);
    }
    pset_p(result, p);
    pmul_ip(result, precip(q, pd));
    pset(result, result->coeffs, degree + 1);
    // Reverse quotient.
    poly_reverse_ip(result);
    if ((dd = degree - result->degree) > 0) {
        pmul_bp_ip(&result, dd);
    }
    prev_ip(&p);
    prev_ip(&q);
    pset_p(result + 1, *result);
    pmul_ip(result + 1, q);
    psubr_ip(result + 1, );
    
    return result;
}


// Calculates p / x^m and p % x^m.
polynomial* pdiv_bp(polynomial p, size_t m) {
    polynomial *result;
    complex *pc;
    int pd;
    
    pc = p.coeffs;
    pd = p.degree;
    result = malloc(2 * sizeof(polynomial));
    if (result == NULL) {
        fprintf(stderr, "Cannot allocate memory to quotient and remainder.\n");
        exit(1);
    }
    
    if (m <= pd) {
        pset(&result[0], pc + m, pd - m + 1);
        pset(&result[1], pc, m);
    } else {
        ps_zero(&result[0]);
        pset(&result[1], pc, pd + 1);
    }
    
    return result;
}

// Calculates p(z) using Horner's method
complex peval(polynomial p, complex z) {
    complex r = 0;
    int i;
    
    for (i = p.degree; i > 0; i--) {
        r += p.coeffs[i];
        r *= z;
    }
    r += p.coeffs[0];
    return r;
}

// Calculates recursive evaluation on n points simultaneously in O(n (log n)^2) time.
// The naive implementation would take O(n^2) time using Horner's method.
// Source: http://people.csail.mit.edu/madhu/ST12/scribe/lect06.pdf
complex *pevalm(polynomial p, complex *z, size_t n) {
    // 9, 5, 3, 2; => 4, 19
    // 5, 3, 2 => 3, 10
    complex *coeffs, *output, *pc;
    polynomial **polys = malloc(log2_int(n) * sizeof(polynomial *)), *pt;
    int i, j, k, b;
    
    polys[0] = malloc(n * sizeof(polynomial));
    pt = polys[0];
    for (i = 0; i < n; i++) {
        pt[i];
        pt[i].coeffs = malloc(2 * sizeof(complex));
        pt[i].coeffs = -z[i];
        *(pt[i].coeffs + 1) = 1;
        pt[i].degree = 1;
    }
    
    for (i = n, k = 0, b = (n & 1); i > 2; i = (i + 1) / 2, k++, b = i & 1) {
        if (b) {
            polys[k + 1] = malloc((i / 2 + 1) * sizeof(polynomial));
            polys[k + 1][i / 2] = polys[k][2 * j];
        } else {
            polys[k + 1] = malloc((i / 2) * sizeof(polynomial));
        }
        
        for (j = 0; j < i / 2; j++) {
            polys[k + 1][j] = pmul(polys[k][2 * j], polys[k][2 * j + 1]);
        }
    }
    output = malloc(n * sizeof(complex));
    evalpmr(output, p, polys, z, log2_int(round_up(n)), n, 0);
    
    return output;
}

/*

Calculates polynomial evaluation recursively.
Inputs:
Polynomial p - the polynomial function to be evaluated.
complex *z - The list of complex numbers on which p will be evaluated.
int layer - The layer of recursion.
int size - The size of *z.
int index - The index on which the output will be written.
Output: *output - p(z1), p(z2), ..., p(zl), where l is the length of z.
 */
void pevalmr(complex *output, polynomial *p, polynomial **polys, complex *z, unsigned int layer, int size, int index) {
    polynomial *p0, *p1;
    int m;
    if (size == 1) {
        *output = evaluate_poly(p, *z);
    } else {
        m = 1 << log2_int(size - 1);
        p0 = *(pdiv(p, &polys[layer - 1][2 * index]) + 1);
        pevalmr(output, p0, polys, z, layer - 1, m, 2 * index);
        if (polys[layer - 1][2 * index + 1].coeffs != NULL) {
            p1 = *(pdiv(p, &polys[layer - 1][2 * index + 1]) + 1);
            pevalmr(output + m, p1, polys, z + m, layer - 1, m, 2 * index + 1);
        }
    }
}

// Calculates p * q.
polynomial pmul(polynomial p, polynomial q) {
    complex *pf, *qf, *pc, *qc, *rc;
    int i, j, pd, qd, rd;
    polynomial r;
    
    pd = p.degree;
    qd = q.degree;
    pc = p.coeffs;
    qc = q.coeffs;
    
    rd = pd + qd + 1;
    rd = round_up(rd);
    pf = malloc(rd * sizeof(complex));
    qf = malloc(rd * sizeof(complex));
    rc = malloc(rd * sizeof(complex));
    if (pf == NULL || qf == NULL || rc == NULL) {
        fprintf(stderr, "Cannot allocate memory.");
        exit(1);
    }
    
    // Pad with zeroes.
    
    for (i = 0; i <= pd; i++) {
        pf[i] = pc[i];
    }
    for (i = 0; i <= qd; i++) {
        qf[i] = qc[i];
    }
    
    // Do in-place transformation.
    fft_it(pf, rd, 0);
    fft_it(qf, rd, 0);
    
    
    
    // Do pointwise multiplication.
    for (i = 0; i < rd; i++) {
        rc[i] = pf[i] * qf[i];
    }
    
    // Do inverse FFT.
    fft_it(rc, rd, 1);
    
    pset(&r, rc, rd);
    
    return r;
}

// Calculates p(x) * x^power
polynomial pmul_bp(polynomial p, size_t power) {
    int i, pd;
    polynomial r;
    complex *pc, *coeffs;
    
    pc = p.coeffs;
    pd = p.degree;
    coeffs = malloc((power + pd + 1) * sizeof(complex));
    if (coeffs == NULL) {
        fprintf(stderr, "Cannot allocate memory to quotient and remainder.\n");
        exit(1);
    }
    
    for (i = 0; i < power; i++) {
        coeffs[i] = 0;
    }
    for (i = 0; i <= pd; i++) {
        coeffs[power + i] = pc[i];
    }
    
    pset(&r, coeffs, power + p.degree + 1);
    
    return r;
}

// Calculates p(x) * x^power and stores it in p(x).
void pmul_bp_ip(polynomial *p, size_t power) {
    int i, pd;
    complex *pc, *coeffs, temp;
    
    pc = p->coeffs;
    pd = p->degree;
    
    pc = realloc(pc, (power + pd + 1) * sizeof(complex));
    if (pc == NULL) {
        fprintf(stderr, "Cannot allocate memory to quotient and remainder.\n");
        exit(1);
    }
    
    for (i = pd; i >= 0; i--) {
        temp = pc[i];
        pc[i] = pc[power + i];
        pc[power + i] = temp;
    }
    p->degree = power + pd;
    p->coeffs = pc;
}

// Calculates p * q and stores the result into p.
void pmul_ip(polynomial *p, polynomial q) {
    complex *pc, *qc, *qf;
    int i, j, pd, qd, rd;
    polynomial r;
    
    pd = p->degree;
    qd = q.degree;
    pc = p->coeffs;
    qc = q.coeffs;
    
    rd = pd + qd + 1;
    rd = round_up(rd);
    pc = realloc(pc, rd * sizeof(complex));
    qf = malloc(rd * sizeof(complex));
    if (pc == NULL || qc == NULL) {
        fprintf(stderr, "Cannot allocate memory.");
        exit(1);
    }
    arr_deep_copy(qf, qc, qd + 1);
    
    // Do in-place transformation.
    fft_it(pc, rd, 0);
    fft_it(qf, rd, 0);
    
    
    
    // Do pointwise multiplication.
    for (i = 0; i < rd; i++) {
        pc[i] *= qf[i];
    }
    
    // Do inverse FFT.
    fft_it(pc, rd, 1);
    // free(qf);
    
    p->coeffs = pc;
    p->degree = pd + qd;
    if (qc == NULL) {
        fprintf(stderr, "Cannot reallocate memory.");
        exit(1);
    }
}

// Calculates -p
polynomial pneg(polynomial p) {
    polynomial r;
    complex *coeffs = malloc((p.degree + 1) * sizeof(complex));
    int i;
    
    if (coeffs == NULL) {
        fprintf(stderr, "Cannot allocate memory.");
        exit(1);
    }
    for (i = 0; i <= p.degree; i++) {
        coeffs[i] = -p.coeffs[i];
    }
    pset(&r, coeffs, 1 + p.degree);
    
    return r;
}

// Sets p to zero.
void ps_zero(polynomial *p) {
    polynomial r;
    complex *coeffs = malloc(sizeof(complex));
    int i;
    
    if (coeffs == NULL) {
        fprintf(stderr, "Cannot allocate memory.");
        exit(1);
    }
    *coeffs = 0;

    pset(&r, coeffs, 1);


}

// Sets p to -p.
void pneg_ip(polynomial* p) {
    int idx, pd;
    complex *curr;
    
    pd = p->degree;
    
    for (curr = p->coeffs, idx = 0; idx <= pd; idx++, curr++) {
        *curr = -*curr;
    }
}

// Calculates the polynomial q such that p(x) * q(x) == 1 (mod x^degree).
polynomial precip(polynomial p, int degree) {
    polynomial r, p1, p2, q1, q2, tmp;
    complex *c, *pc;
    int m, pd;

    c = malloc(sizeof(complex));
    if (degree <= 0) {
        fprintf(stderr, "Degree cannot be zero or negative.");
        exit(1);
    } else if (degree == 1) {
        pc = p.coeffs;
        if (c == NULL) {
            fprintf(stderr, "Cannot allocate memory.");
            exit(1);
        }
        *c = 1 / *pc;
        pset(&r, c, 1);
        
        return r;
    }
    
    pc = p.coeffs;
    pd = p.degree;
    
    m = round_up(degree) / 2;
    q1 = precip(p, m);

    // set p1 and p2 according to p and m.
    switch (pd / m) {
        case 0:
            pset(&p1, pc, 1 + pd);
            ps_zero(&p2);
            break;
        case 1:
            pset(&p1, pc, m);
            pset(&p2, pc + m, pd - m + 1);
            break;
        default:
            pset(&p1, pc, m);
            pset(&p2, pc + m, m);
            break;
    }
    // set q2 = ((q1 (mod x^m)) * p1) mod x^m
    pset(&q2, q1.coeffs, m);
    pmul_ip(&q2, p1);
    q2 = *pdiv_bp(q2, m);
    
    // tmp = constant polynomial 1
    ps_zero(&tmp);
    *(tmp.coeffs) = 1;

    // r = q1 * (1 - (q1 * (p2 * q1 + q2) * x^m)
    // r = q1 - (q1 * (p2 * q1) + q2) * x^m
    pset_p(&r, p2);
    pmul_ip(&r, q1);
    padd_ip(&r, q2);
    pmul_bp_ip(&r, m);
    psubr_ip(&r, tmp);
    pmul_ip(&r, q1);
    pset(&r, r.coeffs, degree);
    return r;
}

// Computes x^n * p(x^-1), where n is the degree of the polynomial.
polynomial prev(polynomial p) {
    int idx, pd;
    complex *pc,*coeffs, *result;
    polynomial r;
    
    pc = p.coeffs;
    pd = p.degree;
    
    result = malloc((pd + 1) * sizeof(complex));
    
    for (coeffs = pc + pd, idx = 0; coeffs >= pc; coeffs--, idx++) {
        result[idx] = *coeffs;
    }
    
    r.coeffs = result;
    r.degree = pd;
    
    return r;
}

// Calculates x^d * p(x^-1), where d is the degree of p (the reverse of p) and stores the result onto p.
void prev_ip(polynomial *p) {
    int i, j, pd;
    complex *coeffs, *pc, *result, tmp;
    polynomial r;
    
    pc = p->coeffs;
    pd = p->degree;
    
    for (i = 0, j = pd; i < j; i++, j--) {
        tmp = pc[i];
        pc[i] = pc[j];
        pc[j] = tmp;
    }
}

// Sets the coefficients of the polynomial. If coeffs = {a0, a1, ...}, then p(x) = a0 + a1*x + ...
// In other words, the coefficients are little endian.
// Inputs:
// complex *coeffs - The coefficients of the polynomial,
// size_t length - The length of the coefficients.
// Output:
// polynomial *p - The polynomial with the specified coefficients.
void pset(polynomial *p, complex *coeffs, size_t length) {
    complex *coeff, *terms, *curr;
    int idx, size;
    
    coeff = coeffs + length - 1;
    // Iterate from term to term.
    // While terms has next element:
    for (size = length; size > 1 && cabs(*coeff--) < EPSILON; size--) ;
    
    terms = malloc(size * sizeof(complex));
    if (p == NULL || terms == NULL) {
        fprintf(stderr, "Cannot allocate memory for new coefficients.");
        exit(1);
    }
    for (idx = 0; idx < size; idx++) {
        terms[idx] = coeffs[idx];
    }
    
    p->coeffs = terms;
    p->degree = size - 1;
}

// Sets the coefficients of p to the coefficients of q.
void pset_p(polynomial *p, polynomial q) {
    complex *coeff, *terms;
    int idx, size;


    size = q.degree + 1;
    terms = malloc(size * sizeof(complex));
    if (p == NULL || terms == NULL) {
        fprintf(stderr, "Cannot allocate memory for new coefficients.");
        exit(1);
    }
    for (idx = 0; idx < size; idx++) {
        terms[idx] = q.coeffs[idx];
    }
    
    p->coeffs = terms;
    p->degree = size - 1;
}

// Compute p - q.
polynomial psub(polynomial p, polynomial q) {
    int idx, size, pd, qd, max;
    complex *result, *curr, *pc, *qc;
    polynomial r;
    
    pd = p.degree;
    pc = p.coeffs;
    qd = q.degree;
    qc = q.coeffs;
    
    max = MAX(pd, qd);
    result = malloc((1 + max) * sizeof(complex));
    
    if (result == NULL) {
        fprintf(stderr, "Cannot allocate memory to result.");
        exit(1);
    }
    for (idx = 0; idx <= pd; idx++) {
        result[idx] = p.coeffs[idx];
    }
    for (idx = 0; idx <= qd; idx++) {
        result[idx] -= q.coeffs[idx];
    }
    r.coeffs = result;
    r.degree = max;
    
    return r;
}

// Compute p - q and stores the result in p.
void psub_ip(polynomial *p, polynomial q) {
    int pd, qd, td, idx, rd;
    complex *pc, *qc, *curr;
    
    pd = p->degree;
    pc = p->coeffs;
    qd = q.degree;
    qc = q.coeffs;
    
    if (pd < qd) {
        pc = realloc(pc, (1 + qd) * sizeof(complex));
        if (pc == NULL) {
            fprintf(stderr, "Cannot reallocate memory for the new coefficients.");
            exit(1);
        }
        p->degree = qd;
    }
    for (idx = 0, curr = qc; idx <= qd; curr++, idx++) {
        pc[idx] -= *curr;
    }
    // Trim trailing zeros
    curr--;
    while (p->degree && cabs(*curr--) < EPSILON) {
        (p->degree)--;
    }
}

// Calculates q - p and stores the result in p.
void psubr_ip(polynomial *p, polynomial q) {
    int pd, qd, td, idx, rd;
    complex *pc, *qc, *curr;
    
    pd = p->degree;
    pc = p->coeffs;
    qd = q.degree;
    qc = q.coeffs;
    
    if (pd < qd) {
        pc = realloc(pc, (1 + qd) * sizeof(complex));
        if (pc == NULL) {
            fprintf(stderr, "Cannot reallocate memory for the new coefficients.");
            exit(1);
        }
        p->degree = qd;
    }
    for (idx = 0, curr = qc; idx <= qd; curr++, idx++) {
        pc[idx] = *curr - pc[idx];
    }
    // Trim trailing zeros
    curr--;
    while (p->degree && cabs(*curr--) < EPSILON) {
        (p->degree)--;
    }
}

// Reverses x, up to a given size.
unsigned int reverse(unsigned int x, unsigned int size) {
    int result = x, s = round_up(size);
    unsigned int mask = (unsigned int) (~0) >> (32 - s);
    while ((s >>= 1) > 0) {
      mask ^= (mask << s);
      result = ((result >> s) & mask) | ((result << s) & ~mask);
    }
    result >>= (round_up(size) - size);
    return result;
}

// Rounds up x to the nearest power of two.
unsigned int round_up(unsigned int x) {
    int j, s = 4 * sizeof(int), r = x - 1;
    for (int j = 1; j < s; j <<= 1) {
        r |= (r >> j);
    }
    return r + 1;
}

int main() {
    polynomial output;
    polynomial p[3];
    complex coeffs[3][4] = {{1, -2, 0, 0}, {1, 1, 1, 1}, {0, 0, 0}};
    complex z;
    pset(&p[0], coeffs[0], 2);
    pset(&p[1], coeffs[1], 4);
    pset(&p[2], coeffs[2], 2);
    display_polynomial(precip(p[0], 11), 0);
    return 0;
}