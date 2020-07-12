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

// lib

func Init() (err error) {
	return nil
}

func Uninit() {}

// types

func mtNewStream(stream *Stream, code *int) {
	C.mtNewStream((*C.MtStream)(unsafe.Pointer(stream)), (*C.int)(unsafe.Pointer(code)))
}

func (s Stream) Destroy() {
	C.mtStreamDestroy(C.MtStream(s))
}

type bufferHandle = C.MtBuffer

func (buf *Buffer) Destroy() {
	C.mtBufferDestroy(buf.handle)
}
