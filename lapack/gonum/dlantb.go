// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dlantb returns the given norm of an n×n triangular band matrix with k+1
// diagonals.
//
// When norm is lapack.MaxColumnSum, the length of work must be at least n.
func (impl Implementation) Dlantb(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, n, k int, a []float64, lda int, work []float64) float64 {
	switch {
	case norm != lapack.MaxAbs && norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius:
		panic(badNorm)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kdLT0)
	case lda < k+1:
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0
	}

	switch {
	case len(a) < (n-1)*lda+k+1:
		panic(shortAB)
	case len(work) < n && (norm == lapack.MaxColumnSum):
		panic(shortWork)
	}

	var value float64
	switch norm {
	case lapack.MaxAbs:
		if diag == blas.Unit {
			value = 1
			if uplo == blas.Upper {
				for i := 0; i < n; i++ {
					for j := 1; j < min(n-i, k+1); j++ {
						aij := math.Abs(a[i*lda+j])
						if aij > value || math.IsNaN(aij) {
							value = aij
						}
					}
				}
			} else {
				for i := 0; i < n; i++ {
					for j := max(0, k-i); j < k; j++ {
						aij := math.Abs(a[i*lda+j])
						if aij > value || math.IsNaN(aij) {
							value = aij
						}
					}
				}
			}
		} else {
			value = 0
			if uplo == blas.Upper {
				for i := 0; i < n; i++ {
					for j := 0; j < min(n-i, k+1); j++ {
						aij := math.Abs(a[i*lda+j])
						if aij > value || math.IsNaN(aij) {
							value = aij
						}
					}
				}
			} else {
				for i := 0; i < n; i++ {
					for j := max(0, k-i); j < k+1; j++ {
						aij := math.Abs(a[i*lda+j])
						if aij > value || math.IsNaN(aij) {
							value = aij
						}
					}
				}
			}
		}
	case lapack.MaxRowSum:
		var sum float64
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				if diag == blas.Unit {
					sum = 1
					for j := 1; j < min(n-i, k+1); j++ {
						sum += math.Abs(a[i*lda+j])
					}
				} else {
					sum = 0
					for j := 0; j < min(n-i, k+1); j++ {
						sum += math.Abs(a[i*lda+j])
					}
				}
				if sum > value || math.IsNaN(sum) {
					value = sum
				}
			}
		} else {
			for i := 0; i < n; i++ {
				if diag == blas.Unit {
					sum = 1
					for j := max(0, k-i); j < k; j++ {
						sum += math.Abs(a[i*lda+j])
					}
				} else {
					sum = 0
					for j := max(0, k-i); j < k+1; j++ {
						sum += math.Abs(a[i*lda+j])
					}
				}
				if sum > value || math.IsNaN(sum) {
					value = sum
				}
			}
		}
	case lapack.MaxColumnSum:
		work = work[:n]
		var sum float64
		if uplo == blas.Upper {
			if diag == blas.Unit {
				for i := range work {
					work[i] = 1
				}
				for i := 0; i < n; i++ {
					for j := 1; j < min(n-i, k+1); j++ {
						work[i+j] += math.Abs(a[i*lda+j])
					}
				}
			} else {
				for i := range work {
					work[i] = 0
				}
				for i := 0; i < n; i++ {
					for j := 0; j < min(n-i, k+1); j++ {
						work[i+j] += math.Abs(a[i*lda+j])
					}
				}
			}
		} else {
			if diag == blas.Unit {
				for i := range work {
					work[i] = 1
				}
				for i := 0; i < n; i++ {
					for j := max(0, k-i); j < k+1; j++ {
						work[i+j-k] += math.Abs(a[i*lda+j])
					}
				}
			} else {
				for i := range work {
					work[i] = 0
				}
				for i := 0; i < n; i++ {
					for j := max(0, k-i); j < k+1; j++ {
						work[i+j-k] += math.Abs(a[i*lda+j])
					}
				}
			}
		}
		value = 0
		for _, wi := range work {
			if wi > value || math.IsNaN(wi) {
				value = wi
			}
		}
	case lapack.Frobenius:
		scale := 0.0
		ssq := 1.0
		if uplo == blas.Upper {
			if kd > 0 {
				// Sum off-diagonals.
				for i := 0; i < n-1; i++ {
					ilen := min(n-i-1, kd)
					rowscale, rowssq := impl.Dlassq(ilen, ab[i*ldab+1:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
				ssq *= 2
			}
			// Sum diagonal.
			dscale, dssq := impl.Dlassq(n, ab, ldab, 0, 1)
			scale, ssq = impl.Dcombssq(scale, ssq, dscale, dssq)
		} else {
			if kd > 0 {
				// Sum off-diagonals.
				for i := 1; i < n; i++ {
					ilen := min(i, kd)
					rowscale, rowssq := impl.Dlassq(ilen, ab[i*ldab+kd-ilen:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
				ssq *= 2
			}
			// Sum diagonal.
			dscale, dssq := impl.Dlassq(n, ab[kd:], ldab, 0, 1)
			scale, ssq = impl.Dcombssq(scale, ssq, dscale, dssq)
		}
		value = scale * math.Sqrt(ssq)
	}
	return value
}
