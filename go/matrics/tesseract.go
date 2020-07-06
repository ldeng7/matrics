package matrics

import (
	"unsafe"
)

const (
	CnnPadingTypeSame uint8 = iota
	CnnPadingTypeValid
)

type CnnConf struct {
	StrideY uint8
	StrideZ uint8
	Padding uint8
}

func (c *CnnConf) OutputSize(nyIn, nzIn, nyCore, nzCore uint64) (uint64, uint64) {
	sy, sz := uint64(c.StrideY), uint64(c.StrideZ)
	switch c.Padding {
	case CnnPadingTypeSame:
		return (nyIn + sy - 1) / sy, (nzIn + sz - 1) / sz
	case CnnPadingTypeValid:
		return (nyIn-nyCore)/sy + 1, (nzIn-nzCore)/sz + 1
	}
	return 0, 0
}

type Conv2dConf struct {
	Act Activation
	Cnn CnnConf
}

func (tx *Tensor) CubConv2d(ty, tz, tw *Tensor, conf *Conv2dConf, stream Stream) error {
	if tx.nx != ty.nx || ty.nw != tz.nx || ty.nw != tw.nx {
		return unsuitableShapeErr
	} else if nyOut, nzOut := conf.Cnn.OutputSize(tx.ny, tx.nz, ty.ny, ty.nz); tw.ny != nyOut || tw.nz != nzOut {
		return unsuitableShapeErr
	}
	var code int
	mtCubConv2d.Call(tx.handler, ty.handler, tz.handler, tw.handler, uintptr(unsafe.Pointer(conf)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) TesConv2d(ty, tz, tw *Tensor, conf *Conv2dConf, stream Stream) error {
	if tx.nx != ty.nx || tx.nw != tw.nw || ty.nw != tz.nx || ty.nw != tw.nx {
		return unsuitableShapeErr
	} else if nyOut, nzOut := conf.Cnn.OutputSize(tx.ny, tx.nz, ty.ny, ty.nz); tw.ny != nyOut || tw.nz != nzOut {
		return unsuitableShapeErr
	}
	var code int
	mtTesConv2d.Call(tx.handler, ty.handler, tz.handler, tw.handler, uintptr(unsafe.Pointer(conf)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

const (
	CnnPoolTypeMax uint8 = iota
	CnnPoolTypeAvg
)

type CnnPoolConf struct {
	Typ   uint8
	CoreY uint8
	CoreZ uint8
	Cnn   CnnConf
}

func (tx *Tensor) CubCnnPool(ty *Tensor, conf *CnnPoolConf, stream Stream) error {
	if tx.nx != ty.nx {
		return unsuitableShapeErr
	}
	nyOut, nzOut := conf.Cnn.OutputSize(tx.ny, tx.nz, uint64(conf.CoreY), uint64(conf.CoreZ))
	if ty.ny != nyOut || ty.nz != nzOut {
		return unsuitableShapeErr
	}
	var code int
	mtCubCnnPool.Call(tx.handler, ty.handler, uintptr(unsafe.Pointer(conf)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) TesCnnPool(ty *Tensor, conf *CnnPoolConf, stream Stream) error {
	if tx.nx != ty.nx || tx.nw != ty.nw {
		return unsuitableShapeErr
	}
	nyOut, nzOut := conf.Cnn.OutputSize(tx.ny, tx.nz, uint64(conf.CoreY), uint64(conf.CoreZ))
	if ty.ny != nyOut || ty.nz != nzOut {
		return unsuitableShapeErr
	}
	var code int
	mtTesCnnPool.Call(tx.handler, ty.handler, uintptr(unsafe.Pointer(conf)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}
