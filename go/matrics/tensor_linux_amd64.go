package matrics

/*
#cgo CFLAGS: -I ../../libmatrics/include
#cgo LDFLAGS: -lmatrics
#include "libmatrics.h"
*/
import "C"
import (
	"unsafe"
)

type tensorHandle = *C.MtTensor

func (t *Tensor) mtNew(buf *uintptr, code *int) {
	C.mtNewTensor(
		C.uint32(t.nx), C.uint32(t.ny), C.uint32(t.nz), C.uint32(t.nw),
		&t.handle,
		(**C.float)(unsafe.Pointer(buf)),
		(*C.int)(unsafe.Pointer(code)),
	)
}

func (t *Tensor) Destroy() {
	DestroySlice(&t.Data)
	if nil != t.handle {
		C.mtTensorDestroy(t.handle)
	}
	t.handle = nil
}
