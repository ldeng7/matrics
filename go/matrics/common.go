package matrics

import (
	"fmt"
	"reflect"
	"syscall"
	"unsafe"
)

// lib

type MtFunc = *syscall.Proc

var mtLib *syscall.DLL
var mtNewStream MtFunc
var mtStreamDestroy MtFunc
var mtBufferDestroy MtFunc

var mtNewVector MtFunc
var mtVectorDestroy MtFunc

var mtVectorAddScalar MtFunc
var mtVectorMulScalar MtFunc
var mtVectorMulAddScalar MtFunc
var mtVectorPowScalar MtFunc
var mtVectorPowMulScalar MtFunc
var mtVectorAddVector MtFunc
var mtVectorSubVector MtFunc
var mtVectorPatchMulVector MtFunc
var mtVectorTMulMatrix MtFunc

var mtNewVectorAccBuffer MtFunc
var mtVectorSum MtFunc
var mtVectorSquareSum MtFunc
var mtVectorMin MtFunc
var mtVectorMax MtFunc
var mtVectorDot MtFunc
var mtVectorSumSquareSum MtFunc
var mtVectorDiffSquareSum MtFunc

var mtNewMatrix MtFunc
var mtMatrixDestroy MtFunc

var mtMatrixT MtFunc
var mtMatrixAddScalar MtFunc
var mtMatrixMulScalar MtFunc
var mtMatrixMulAddScalar MtFunc
var mtMatrixAddMatrix MtFunc
var mtMatrixSubMatrix MtFunc
var mtMatrixMulMatrix MtFunc
var mtMatrixMulVector MtFunc

var mtNewMatrixWideBuffer MtFunc
var mtMatrixWideMulVector MtFunc

var mtFuncTable = map[string]*MtFunc{
	"mtNewStream":     &mtNewStream,
	"mtStreamDestroy": &mtStreamDestroy,
	"mtBufferDestroy": &mtBufferDestroy,

	"mtNewVector":     &mtNewVector,
	"mtVectorDestroy": &mtVectorDestroy,

	"mtVectorAddScalar":      &mtVectorAddScalar,
	"mtVectorMulScalar":      &mtVectorMulScalar,
	"mtVectorMulAddScalar":   &mtVectorMulAddScalar,
	"mtVectorPowScalar":      &mtVectorPowScalar,
	"mtVectorPowMulScalar":   &mtVectorPowMulScalar,
	"mtVectorAddVector":      &mtVectorAddVector,
	"mtVectorSubVector":      &mtVectorSubVector,
	"mtVectorPatchMulVector": &mtVectorPatchMulVector,
	"mtVectorTMulMatrix":     &mtVectorTMulMatrix,

	"mtNewVectorAccBuffer":  &mtNewVectorAccBuffer,
	"mtVectorSum":           &mtVectorSum,
	"mtVectorSquareSum":     &mtVectorSquareSum,
	"mtVectorMin":           &mtVectorMin,
	"mtVectorMax":           &mtVectorMax,
	"mtVectorDot":           &mtVectorDot,
	"mtVectorSumSquareSum":  &mtVectorSumSquareSum,
	"mtVectorDiffSquareSum": &mtVectorDiffSquareSum,

	"mtNewMatrix":     &mtNewMatrix,
	"mtMatrixDestroy": &mtMatrixDestroy,

	"mtMatrixT":            &mtMatrixT,
	"mtMatrixAddScalar":    &mtMatrixAddScalar,
	"mtMatrixMulScalar":    &mtMatrixMulScalar,
	"mtMatrixMulAddScalar": &mtMatrixMulAddScalar,
	"mtMatrixAddMatrix":    &mtMatrixAddMatrix,
	"mtMatrixSubMatrix":    &mtMatrixSubMatrix,
	"mtMatrixMulMatrix":    &mtMatrixMulMatrix,
	"mtMatrixMulVector":    &mtMatrixMulVector,

	"mtNewMatrixWideBuffer": &mtNewMatrixWideBuffer,
	"mtMatrixWideMulVector": &mtMatrixWideMulVector,
}

func LoadLib() (err error) {
	ReleaseLib()
	if mtLib, err = syscall.LoadDLL("libmt.dll"); nil != err {
		err = fmt.Errorf("failed to load DLL: %s", err.Error())
		return
	}
	defer func() {
		if nil != err {
			ReleaseLib()
		}
	}()

	for k, f := range mtFuncTable {
		var proc MtFunc
		if proc, err = mtLib.FindProc(k); nil == err {
			*f = proc
		} else {
			err = fmt.Errorf("failed to load func: %s", k)
			return
		}
	}
	return
}

func ReleaseLib() {
	if nil != mtLib {
		mtLib.Release()
	}
}

// types

const mtCodeSuccess = 0

type mtErr struct {
	code int
	msg  string
}

func (le *mtErr) Error() string {
	return fmt.Sprintf("%s(cuda code: %d)", le.msg, le.code)
}

type Stream uintptr

func NewStream() (Stream, error) {
	var stream Stream
	var code int
	mtNewStream.Call(
		uintptr(unsafe.Pointer(&stream)),
		uintptr(unsafe.Pointer(&code)),
	)
	if mtCodeSuccess != code {
		return 0, &mtErr{code, ""}
	}
	return stream, nil
}

func (s Stream) Destroy() {
	mtStreamDestroy.Call(uintptr(s))
}

// utils

func makeSlice(sl *[]float32, buf uintptr, length int) {
	h := (*reflect.SliceHeader)(unsafe.Pointer(sl))
	h.Data, h.Len, h.Cap = buf, length, length
}

func destroySlice(sl *[]float32) {
	h := (*reflect.SliceHeader)(unsafe.Pointer(sl))
	h.Data, h.Len, h.Cap = 0, 0, 0
}
