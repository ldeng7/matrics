package matrics

import (
	"errors"
	"reflect"
	"unsafe"
)

var unsuitableShapeErr error = errors.New("unsuitable shape")

func makeSlice(sl *[]float32, buf uintptr, length int) {
	h := (*reflect.SliceHeader)(unsafe.Pointer(sl))
	h.Data, h.Len, h.Cap = buf, length, length
}

func DestroySlice(sl *[]float32) {
	h := (*reflect.SliceHeader)(unsafe.Pointer(sl))
	h.Data, h.Len, h.Cap = 0, 0, 0
}

const TENSOR_DIM_LEN_MAX = 4

type Tensor struct {
	handler        uintptr
	n              uint64
	nd             int
	nx, ny, nz, nw uint64
	Data           []float32
}

func NewTensor(shape []uint64) (*Tensor, error) {
	nd := len(shape)
	if nd == 0 || nd > TENSOR_DIM_LEN_MAX {
		return nil, errors.New("invalid dimension length")
	}
	t := &Tensor{nd: nd, nx: 1, ny: 1, nz: 1, nw: 1}
	for i, n := range shape {
		if n == 0 {
			return nil, errors.New("invalid shape")
		}
		switch i {
		case 0:
			t.nx = n
		case 1:
			t.ny = n
		case 2:
			t.nz = n
		case 3:
			t.nw = n
		}
	}
	t.n = t.nx * t.ny * t.nz * t.nw

	var buf uintptr
	var code int
	mtNewTensor.Call(
		uintptr(t.nx), uintptr(t.ny), uintptr(t.nz), uintptr(t.nw),
		uintptr(unsafe.Pointer(&t.handler)),
		uintptr(unsafe.Pointer(&buf)),
		uintptr(unsafe.Pointer(&code)),
	)
	if mtCodeSuccess != code {
		return nil, &mtError{code, "failed to alloc"}
	}
	makeSlice(&t.Data, buf, int(t.n))
	return t, nil
}

func (t *Tensor) Destroy() {
	DestroySlice(&t.Data)
	mtTensorDestroy.Call(t.handler)
}

func (t *Tensor) Len() uint64 {
	return t.n
}

func (t *Tensor) DimLen() int {
	return t.nd
}

func (t *Tensor) Shape() [TENSOR_DIM_LEN_MAX]uint64 {
	return [...]uint64{t.nx, t.ny, t.nz, t.nw}
}

func (t *Tensor) ForEach(f func(e float32) float32) {
	sl := t.Data
	for i, e := range sl {
		sl[i] = f(e)
	}
	DestroySlice(&sl)
}

const (
	ActivationTypeNone uint32 = iota
	ActivationTypeReLU
	ActivationTypeLReLU
	ActivationTypeELU
	ActivationTypeSwish
)

type Activation struct {
	Typ uint32
	Arg float32
}

var ActivationNone *Activation = &Activation{ActivationTypeNone, 0}
var ActivationReLU *Activation = &Activation{ActivationTypeReLU, 0}
