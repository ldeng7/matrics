package matrics

import (
	"fmt"
	"unsafe"
)

type Vector struct {
	handler uintptr
	length  uint64
	Buf     []float32
}

func NewVector(length uint64) (*Vector, error) {
	vec := &Vector{length: length}
	var buf uintptr
	var code int
	mtNewVector.Call(
		uintptr(length),
		uintptr(unsafe.Pointer(&vec.handler)),
		uintptr(unsafe.Pointer(&buf)),
		uintptr(unsafe.Pointer(&code)),
	)
	if mtCodeSuccess != code {
		return nil, &mtErr{code, "failed to alloc"}
	}
	makeSlice(&vec.Buf, buf, int(length))
	return vec, nil
}

func (vec *Vector) Destroy() {
	mtVectorDestroy.Call(vec.handler)
	destroySlice(&vec.Buf)
}

// ind

func (vec *Vector) ind1s1(f MtFunc, vec1 *Vector, a float32, stream Stream) error {
	if vec.length != vec1.length {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	f.Call(vec.handler, vec1.handler, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

func (vec *Vector) ind1s2(f MtFunc, vec1 *Vector, a, b float32, stream Stream) error {
	if vec.length != vec1.length {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	f.Call(vec.handler, vec1.handler, uintptr(unsafe.Pointer(&a)), uintptr(unsafe.Pointer(&b)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

func (vec *Vector) ind2(f MtFunc, vec1, vec2 *Vector, stream Stream) error {
	if vec.length != vec1.length || vec.length != vec2.length {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	f.Call(vec.handler, vec1.handler, vec2.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

func (vec *Vector) AddScalar(a float32, vec1 *Vector, stream Stream) error {
	return vec.ind1s1(mtVectorAddScalar, vec1, a, stream)
}

func (vec *Vector) MulScalar(a float32, vec1 *Vector, stream Stream) error {
	return vec.ind1s1(mtVectorMulScalar, vec1, a, stream)
}

func (vec *Vector) MulAddScalar(a, b float32, vec1 *Vector, stream Stream) error {
	return vec.ind1s2(mtVectorMulAddScalar, vec1, a, b, stream)
}

func (vec *Vector) PowScalar(a float32, vec1 *Vector, stream Stream) error {
	return vec.ind1s1(mtVectorPowScalar, vec1, a, stream)
}

func (vec *Vector) PowMulScalar(a, b float32, vec1 *Vector, stream Stream) error {
	return vec.ind1s2(mtVectorPowMulScalar, vec1, a, b, stream)
}

func (vec *Vector) AddVector(vec1, vec2 *Vector, stream Stream) error {
	return vec.ind2(mtVectorAddVector, vec1, vec2, stream)
}

func (vec *Vector) SubVector(vec1, vec2 *Vector, stream Stream) error {
	return vec.ind2(mtVectorSubVector, vec1, vec2, stream)
}

func (vec *Vector) PatchMulVector(vec1, vec2 *Vector, stream Stream) error {
	return vec.ind2(mtVectorPatchMulVector, vec1, vec2, stream)
}

func (vec *Vector) TMulMatrix(mat *Matrix, vec1 *Vector, stream Stream) error {
	if vec.length != mat.height || mat.width != vec1.length {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	mtVectorTMulMatrix.Call(vec.handler, mat.handler, vec1.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

// acc

type VectorAccBuffer struct {
	handler uintptr
	vecLen  uint64
}

func (vec *Vector) NewAccBuffer() (*VectorAccBuffer, error) {
	buf := &VectorAccBuffer{vecLen: vec.length}
	var code int
	mtNewVectorAccBuffer.Call(vec.handler,
		uintptr(unsafe.Pointer(&buf.handler)), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return nil, &mtErr{code, "failed to alloc"}
	}
	return buf, nil
}

func (buf *VectorAccBuffer) Destroy() {
	mtBufferDestroy.Call(buf.handler)
}

func (vec *Vector) acc(f MtFunc, buf *VectorAccBuffer, stream Stream) (float32, error) {
	if vec.length != buf.vecLen {
		return 0, fmt.Errorf("unsuitable length")
	}
	var res float32
	var code int
	f.Call(vec.handler, buf.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&res)), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return 0, &mtErr{code, ""}
	}
	return res, nil
}

func (vec *Vector) acc1(f MtFunc, vec1 *Vector, buf *VectorAccBuffer, stream Stream) (float32, error) {
	if vec.length != vec1.length || vec.length != buf.vecLen {
		return 0, fmt.Errorf("unsuitable length")
	}
	var res float32
	var code int
	f.Call(vec.handler, vec1.handler, buf.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&res)), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return 0, &mtErr{code, ""}
	}
	return res, nil
}

func (vec *Vector) Sum(buf *VectorAccBuffer, stream Stream) (float32, error) {
	return vec.acc(mtVectorSum, buf, stream)
}

func (vec *Vector) SquareSum(buf *VectorAccBuffer, stream Stream) (float32, error) {
	return vec.acc(mtVectorSquareSum, buf, stream)
}

func (vec *Vector) Min(buf *VectorAccBuffer, stream Stream) (float32, error) {
	return vec.acc(mtVectorMin, buf, stream)
}

func (vec *Vector) Max(buf *VectorAccBuffer, stream Stream) (float32, error) {
	return vec.acc(mtVectorMax, buf, stream)
}

func (vec *Vector) Dot(vec1 *Vector, buf *VectorAccBuffer, stream Stream) (float32, error) {
	return vec.acc1(mtVectorDot, vec1, buf, stream)
}

func (vec *Vector) SumSquareSum(vec1 *Vector, buf *VectorAccBuffer, stream Stream) (float32, error) {
	return vec.acc1(mtVectorSumSquareSum, vec1, buf, stream)
}

func (vec *Vector) DiffSquareSum(vec1 *Vector, buf *VectorAccBuffer, stream Stream) (float32, error) {
	return vec.acc1(mtVectorDiffSquareSum, vec1, buf, stream)
}
