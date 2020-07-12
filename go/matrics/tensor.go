package matrics

import (
	"errors"
	"reflect"
	"unsafe"
)

var unsuitableShapeErr error = errors.New("unsuitable shape")
var invalidDimErr error = errors.New("invalid dimension")
var unsuitableBufferErr error = errors.New("unsuitable buffer")

func makeSlice(sl *[]float32, buf uintptr, length int) {
	h := (*reflect.SliceHeader)(unsafe.Pointer(sl))
	h.Data, h.Len, h.Cap = buf, length, length
}

func DestroySlice(sl *[]float32) {
	h := (*reflect.SliceHeader)(unsafe.Pointer(sl))
	h.Data, h.Len, h.Cap = 0, 0, 0
}

const TENSOR_DIM_MAX = 3

type TensorShape = [TENSOR_DIM_MAX + 1]uint32

type Tensor struct {
	handle            tensorHandle
	n, nx, ny, nz, nw uint32
	shape             TensorShape
	Data              []float32
}

func NewTensor(shape []uint32) (*Tensor, error) {
	if len(shape) == 0 || len(shape) > TENSOR_DIM_MAX+1 {
		return nil, invalidDimErr
	}
	t := &Tensor{n: 1, shape: [...]uint32{1, 1, 1, 1}}
	for _, n := range shape {
		t.n *= n
	}
	if t.n == 0 {
		return nil, unsuitableShapeErr
	}
	copy(t.shape[:], shape)
	t.nx, t.ny, t.nz, t.nw = t.shape[0], t.shape[1], t.shape[2], t.shape[3]

	var buf uintptr
	var code int
	t.mtNew(&buf, &code)
	if mtCodeSuccess != code {
		return nil, &mtError{code, "failed to alloc"}
	}
	makeSlice(&t.Data, buf, int(t.n))
	return t, nil
}

func (t *Tensor) Shape() TensorShape {
	return t.shape
}

func (t *Tensor) AvailableDimLen() uint {
	for i := TENSOR_DIM_MAX; i >= 0; i++ {
		if t.shape[i] > 1 {
			return uint(i + 1)
		}
	}
	return 0
}

func NewFlattenTensor(shape TensorShape, startDim, endDim uint, collapse bool) (*Tensor, error) {
	if startDim >= endDim || startDim > TENSOR_DIM_MAX || endDim > TENSOR_DIM_MAX {
		return nil, invalidDimErr
	}
	for i := startDim + 1; i <= endDim; i++ {
		shape[startDim] *= shape[i]
	}
	if collapse {
		i := startDim + 1
		for j := endDim + 1; j <= TENSOR_DIM_MAX; i, j = i+1, j+1 {
			shape[i] = shape[j]
		}
		for ; i <= TENSOR_DIM_MAX; i++ {
			shape[i] = 1
		}
	} else {
		for i := startDim + 1; i <= endDim; i++ {
			shape[i] = 1
		}
	}
	return NewTensor(shape[:])
}

func NewConcatTensor(shapes []TensorShape, dim uint) (*Tensor, error) {
	if dim > TENSOR_DIM_MAX {
		return nil, invalidDimErr
	}
	shape := TensorShape{}
	for i := uint(0); i <= TENSOR_DIM_MAX; i++ {
		if i != dim {
			shape[i] = shapes[0][i]
			for j := 1; j < len(shapes); j++ {
				if shape[i] != shapes[j][i] {
					return nil, unsuitableShapeErr
				}
			}
		} else {
			for _, s := range shapes {
				shape[i] += s[i]
			}
		}
	}
	return NewTensor(shape[:])
}

func (t *Tensor) ConcatFrom(ts []*Tensor, dim uint) error {
	l := len(ts)
	strides := make([]uint32, l)
	var strideBase uint32 = 1
	for i := uint(0); i < dim; i++ {
		strideBase *= ts[0].shape[i]
	}
	for i, t := range ts {
		strides[i] = strideBase * t.shape[dim]
	}
	offsets := make([]uint32, l)
	offsets[0] = 0
	for i := 1; i < l; i++ {
		offsets[i] = offsets[i-1] + strides[i-1]
	}

	nRound := ts[0].n / strides[0]
	var strideRound, offsetRound uint32 = offsets[l-1] + strides[l-1], 0
	for i := uint32(0); i < nRound; i, offsetRound = i+1, offsetRound+strideRound {
		for j, tIn := range ts {
			sj := strides[j]
			offsetOut, offsetIn := offsetRound+offsets[j], sj*i
			copy(t.Data[offsetOut:offsetOut+sj], tIn.Data[offsetIn:offsetIn+sj])
		}
	}
	return nil
}
