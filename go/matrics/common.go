package matrics

import (
	"fmt"
	"syscall"
	"unsafe"
)

// lib

type mtFunc = *syscall.Proc

var mtLib *syscall.DLL
var mtNewStream mtFunc
var mtStreamDestroy mtFunc
var mtNewTensor mtFunc
var mtTensorDestroy mtFunc
var mtBufferDestroy mtFunc

var mtVecAddScalar mtFunc
var mtVecMulScalar mtFunc
var mtVecMulAddScalar mtFunc
var mtVecPowScalar mtFunc
var mtVecPowMulScalar mtFunc
var mtVecAddVec mtFunc
var mtVecSubVec mtFunc
var mtVecPatchMulVec mtFunc

var mtNewVecAccBuffer mtFunc
var mtVecSum mtFunc
var mtVecSquareSum mtFunc
var mtVecMin mtFunc
var mtVecMax mtFunc
var mtVecDot mtFunc
var mtVecSumSquareSum mtFunc
var mtVecDiffSquareSum mtFunc

var mtMatT mtFunc
var mtMatAddScalar mtFunc
var mtMatMulScalar mtFunc
var mtMatMulAddScalar mtFunc
var mtMatPowScalar mtFunc
var mtMatPowMulScalar mtFunc
var mtMatAddMat mtFunc
var mtMatSubMat mtFunc

var mtMatMulMat mtFunc
var mtVecTMulMat mtFunc
var mtMatMulVec mtFunc
var mtMatMulMatAddVecTAct mtFunc
var mtVecTMulMatAddVecTAct mtFunc
var mtNewMatFlatBuffer mtFunc
var mtMatFlatMulVec mtFunc

var mtCubConv2d mtFunc
var mtTesConv2d mtFunc
var mtCubCnnPool mtFunc
var mtTesCnnPool mtFunc

var mtFuncTable = map[string]*mtFunc{
	"mtNewStream":     &mtNewStream,
	"mtStreamDestroy": &mtStreamDestroy,
	"mtNewTensor":     &mtNewTensor,
	"mtTensorDestroy": &mtTensorDestroy,
	"mtBufferDestroy": &mtBufferDestroy,

	"mtVecAddScalar":    &mtVecAddScalar,
	"mtVecMulScalar":    &mtVecMulScalar,
	"mtVecMulAddScalar": &mtVecMulAddScalar,
	"mtVecPowScalar":    &mtVecPowScalar,
	"mtVecPowMulScalar": &mtVecPowMulScalar,
	"mtVecAddVec":       &mtVecAddVec,
	"mtVecSubVec":       &mtVecSubVec,
	"mtVecPatchMulVec":  &mtVecPatchMulVec,

	"mtNewVecAccBuffer":  &mtNewVecAccBuffer,
	"mtVecSum":           &mtVecSum,
	"mtVecSquareSum":     &mtVecSquareSum,
	"mtVecMin":           &mtVecMin,
	"mtVecMax":           &mtVecMax,
	"mtVecDot":           &mtVecDot,
	"mtVecSumSquareSum":  &mtVecSumSquareSum,
	"mtVecDiffSquareSum": &mtVecDiffSquareSum,

	"mtMatT":            &mtMatT,
	"mtMatAddScalar":    &mtMatAddScalar,
	"mtMatMulScalar":    &mtMatMulScalar,
	"mtMatMulAddScalar": &mtMatMulAddScalar,
	"mtMatPowScalar":    &mtMatPowScalar,
	"mtMatPowMulScalar": &mtMatPowMulScalar,
	"mtMatAddMat":       &mtMatAddMat,
	"mtMatSubMat":       &mtMatSubMat,

	"mtMatMulMat":            &mtMatMulMat,
	"mtVecTMulMat":           &mtVecTMulMat,
	"mtMatMulVec":            &mtMatMulVec,
	"mtMatMulMatAddVecTAct":  &mtMatMulMatAddVecTAct,
	"mtVecTMulMatAddVecTAct": &mtVecTMulMatAddVecTAct,
	"mtNewMatFlatBuffer":     &mtNewMatFlatBuffer,
	"mtMatFlatMulVec":        &mtMatFlatMulVec,

	"mtCubConv2d":  &mtCubConv2d,
	"mtTesConv2d":  &mtTesConv2d,
	"mtCubCnnPool": &mtCubCnnPool,
	"mtTesCnnPool": &mtTesCnnPool,
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
	if mtLib, err = syscall.LoadDLL("libmt.dll"); nil != err {
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

const mtCodeSuccess = 0

type mtError struct {
	code int
	msg  string
}

func (e *mtError) Error() string {
	return fmt.Sprintf("%s(cuda code: %d)", e.msg, e.code)
}

func newMtErr(code int, msg string) error {
	if mtCodeSuccess != code {
		return &mtError{code, msg}
	}
	return nil
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
		return 0, &mtError{code, ""}
	}
	return stream, nil
}

func (s Stream) Destroy() {
	mtStreamDestroy.Call(uintptr(s))
}
