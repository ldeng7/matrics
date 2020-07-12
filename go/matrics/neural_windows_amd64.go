package matrics

import (
	"unsafe"
)

func (tx *Tensor) mtMatMulMatAddVecTAct(ty, tz, tw *Tensor, act *Activation, stream Stream, code *int) {
	mtCMatMulMatAddVecTAct.Call(tx.handle, ty.handle, tz.handle, tw.handle, uintptr(unsafe.Pointer(act)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecTMulMatAddVecTAct(ty, tz, tw *Tensor, act *Activation, stream Stream, code *int) {
	mtCVecTMulMatAddVecTAct.Call(tx.handle, ty.handle, tz.handle, tw.handle, uintptr(unsafe.Pointer(act)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtCubBatchConv2d(ty, tz, tw *Tensor, conf *Conv2dConf, stream Stream, code *int) {
	mtCCubBatchConv2d.Call(tx.handle, ty.handle, tz.handle, tw.handle, uintptr(unsafe.Pointer(conf)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtCubBatchPool2d(ty *Tensor, conf *Pool2dConf, stream Stream, code *int) {
	mtCCubBatchPool2d.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(conf)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}
