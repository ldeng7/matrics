package matrics

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

// lib

type mtFunc = *windows.Proc

var mtLib *windows.DLL
var mtCNewStream mtFunc
var mtCStreamDestroy mtFunc
var mtCNewBuffer mtFunc
var mtCBufferDestroy mtFunc
var mtCNewTensor mtFunc
var mtCTensorDestroy mtFunc

var mtCVecAddScalar mtFunc
var mtCVecMulScalar mtFunc
var mtCVecMulAddScalar mtFunc
var mtCVecPowScalar mtFunc
var mtCVecPowMulScalar mtFunc
var mtCVecAddVec mtFunc
var mtCVecSubVec mtFunc
var mtCVecPatchMulVec mtFunc

var mtCNewVecAccBuffer mtFunc
var mtCVecSum mtFunc
var mtCVecSquareSum mtFunc
var mtCVecMin mtFunc
var mtCVecMax mtFunc
var mtCVecDot mtFunc
var mtCVecSumSquareSum mtFunc
var mtCVecDiffSquareSum mtFunc

var mtCMatT mtFunc
var mtCMatAddScalar mtFunc
var mtCMatMulScalar mtFunc
var mtCMatMulAddScalar mtFunc
var mtCMatPowScalar mtFunc
var mtCMatPowMulScalar mtFunc
var mtCMatAddMat mtFunc
var mtCMatSubMat mtFunc
var mtCMatMulMat mtFunc
var mtCVecTMulMat mtFunc
var mtCMatMulVec mtFunc

var mtCMatMulMatAddVecTAct mtFunc
var mtCVecTMulMatAddVecTAct mtFunc
var mtCCubBatchConv2d mtFunc
var mtCCubBatchPool2d mtFunc

var mtFuncTable = map[string]*mtFunc{
	"mtNewStream":     &mtCNewStream,
	"mtStreamDestroy": &mtCStreamDestroy,
	"mtNewBuffer":     &mtCNewBuffer,
	"mtBufferDestroy": &mtCBufferDestroy,
	"mtNewTensor":     &mtCNewTensor,
	"mtTensorDestroy": &mtCTensorDestroy,

	"mtVecAddScalar":    &mtCVecAddScalar,
	"mtVecMulScalar":    &mtCVecMulScalar,
	"mtVecMulAddScalar": &mtCVecMulAddScalar,
	"mtVecPowScalar":    &mtCVecPowScalar,
	"mtVecPowMulScalar": &mtCVecPowMulScalar,
	"mtVecAddVec":       &mtCVecAddVec,
	"mtVecSubVec":       &mtCVecSubVec,
	"mtVecPatchMulVec":  &mtCVecPatchMulVec,

	"mtNewVecAccBuffer":  &mtCNewVecAccBuffer,
	"mtVecSum":           &mtCVecSum,
	"mtVecSquareSum":     &mtCVecSquareSum,
	"mtVecMin":           &mtCVecMin,
	"mtVecMax":           &mtCVecMax,
	"mtVecDot":           &mtCVecDot,
	"mtVecSumSquareSum":  &mtCVecSumSquareSum,
	"mtVecDiffSquareSum": &mtCVecDiffSquareSum,

	"mtMatT":            &mtCMatT,
	"mtMatAddScalar":    &mtCMatAddScalar,
	"mtMatMulScalar":    &mtCMatMulScalar,
	"mtMatMulAddScalar": &mtCMatMulAddScalar,
	"mtMatPowScalar":    &mtCMatPowScalar,
	"mtMatPowMulScalar": &mtCMatPowMulScalar,
	"mtMatAddMat":       &mtCMatAddMat,
	"mtMatSubMat":       &mtCMatSubMat,
	"mtMatMulMat":       &mtCMatMulMat,
	"mtVecTMulMat":      &mtCVecTMulMat,
	"mtMatMulVec":       &mtCMatMulVec,

	"mtMatMulMatAddVecTAct":  &mtCMatMulMatAddVecTAct,
	"mtVecTMulMatAddVecTAct": &mtCVecTMulMatAddVecTAct,
	"mtCubBatchConv2d":       &mtCCubBatchConv2d,
	"mtCubBatchPool2d":       &mtCCubBatchPool2d,
}

func Init() (err error) {
	if nil != mtLib {
		return nil
	}
	defer func() {
		if nil != err {
			Uninit()
		}
	}()
	if mtLib, err = windows.LoadDLL("C:\\Windows\\libmatrics.dll"); nil != err {
		return fmt.Errorf("failed to load DLL: %s", err.Error())
	}
	for k, f := range mtFuncTable {
		var proc mtFunc
		if proc, err = mtLib.FindProc(k); nil == err {
			*f = proc
		} else {
			return fmt.Errorf("failed to load func: %s", k)
		}
	}
	return nil
}

func Uninit() {
	if nil != mtLib {
		mtLib.Release()
	}
}

// types

func mtNewStream(stream *Stream, code *int) {
	mtCNewStream.Call(uintptr(unsafe.Pointer(stream)), uintptr(unsafe.Pointer(code)))
}

func (s Stream) Destroy() {
	mtCStreamDestroy.Call(uintptr(s))
}

type bufferHandle = uintptr

func (buf *Buffer) Destroy() {
	mtCBufferDestroy.Call(buf.handle)
}
