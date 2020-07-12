package matrics

import (
	"unsafe"
)

type tensorHandle = uintptr

func (t *Tensor) mtNew(buf *uintptr, code *int) {
	mtCNewTensor.Call(
		uintptr(t.nx), uintptr(t.ny), uintptr(t.nz), uintptr(t.nw),
		uintptr(unsafe.Pointer(&t.handle)),
		uintptr(unsafe.Pointer(buf)),
		uintptr(unsafe.Pointer(code)),
	)
}

func (t *Tensor) Destroy() {
	DestroySlice(&t.Data)
	if 0 != t.handle {
		mtCTensorDestroy.Call(t.handle)
	}
	t.handle = 0
}
