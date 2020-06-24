package matrics

import (
	"fmt"
	"unsafe"
)

type Matrix struct {
	handler uintptr
	width   uint64
	height  uint64
	Buf     []float32
}

func NewMatrix(w, h uint64) (*Matrix, error) {
	mat := &Matrix{width: w, height: h}
	var buf uintptr
	var code int
	mtNewMatrix.Call(
		uintptr(w), uintptr(h),
		uintptr(unsafe.Pointer(&mat.handler)),
		uintptr(unsafe.Pointer(&buf)),
		uintptr(unsafe.Pointer(&code)),
	)
	if mtCodeSuccess != code {
		return nil, &mtErr{code, "failed to alloc"}
	}
	makeSlice(&mat.Buf, buf, int(w*h))
	return mat, nil
}

func (mat *Matrix) Destroy() {
	mtMatrixDestroy.Call(mat.handler)
	destroySlice(&mat.Buf)
}

func (mat *Matrix) T(mat1 *Matrix, stream Stream) error {
	if mat.width != mat1.height || mat.height != mat1.width {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	mtMatrixT.Call(mat.handler, mat1.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

func (mat *Matrix) ind1s1(f MtFunc, mat1 *Matrix, a float32, stream Stream) error {
	if mat.width != mat1.width || mat.height != mat1.height {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	f.Call(mat.handler, mat1.handler, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

func (mat *Matrix) ind1s2(f MtFunc, mat1 *Matrix, a, b float32, stream Stream) error {
	if mat.width != mat1.width || mat.height != mat1.height {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	f.Call(mat.handler, mat1.handler, uintptr(unsafe.Pointer(&a)), uintptr(unsafe.Pointer(&b)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

func (mat *Matrix) ind2(f MtFunc, mat1, mat2 *Matrix, stream Stream) error {
	if mat.width != mat1.width || mat.height != mat1.height ||
		mat.width != mat2.width || mat.height != mat2.height {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	f.Call(mat.handler, mat1.handler, mat2.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

func (mat *Matrix) AddScalar(a float32, mat1 *Matrix, stream Stream) error {
	return mat.ind1s1(mtMatrixAddScalar, mat1, a, stream)
}

func (mat *Matrix) MulScalar(a float32, mat1 *Matrix, stream Stream) error {
	return mat.ind1s1(mtMatrixMulScalar, mat1, a, stream)
}

func (mat *Matrix) MulAddScalar(a, b float32, mat1 *Matrix, stream Stream) error {
	return mat.ind1s2(mtMatrixMulScalar, mat1, a, b, stream)
}

func (mat *Matrix) AddMatrix(mat1, mat2 *Matrix, stream Stream) error {
	return mat.ind2(mtMatrixAddMatrix, mat1, mat2, stream)
}

func (mat *Matrix) SubMatrix(mat1, mat2 *Matrix, stream Stream) error {
	return mat.ind2(mtMatrixSubMatrix, mat1, mat2, stream)
}

func (mat *Matrix) MulMatrix(mat1, mat2 *Matrix, stream Stream) error {
	if mat.width != mat1.height || mat.height != mat2.height || mat1.width != mat2.width {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	mtMatrixMulMatrix.Call(mat.handler, mat1.handler, mat2.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

func (mat *Matrix) MulVector(vec, vec1 *Vector, stream Stream) error {
	if mat.width != vec.length || mat.height != vec1.length {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	mtMatrixMulVector.Call(mat.handler, vec.handler, vec1.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}

type MatrixWideBuffer struct {
	handler uintptr
	matW    uint64
	matH    uint64
}

func (mat *Matrix) NewMatrixWideBuffer() (*MatrixWideBuffer, error) {
	buf := &MatrixWideBuffer{matW: mat.width, matH: mat.height}
	var code int
	mtNewMatrixWideBuffer.Call(mat.handler,
		uintptr(unsafe.Pointer(&buf.handler)), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return nil, &mtErr{code, "failed to alloc"}
	}
	return buf, nil
}

func (buf *MatrixWideBuffer) Destroy() {
	mtBufferDestroy.Call(buf.handler)
}

func (mat *Matrix) WideMulVector(vec, vec1 *Vector, buf *MatrixWideBuffer, stream Stream) error {
	if mat.width != vec.length || mat.height != vec1.length ||
		mat.width != buf.matW || mat.height != buf.matH {
		return fmt.Errorf("unsuitable length")
	}
	var code int
	mtMatrixWideMulVector.Call(mat.handler, vec.handler, vec1.handler, buf.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return &mtErr{code, ""}
	}
	return nil
}
