package matrics

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

var ActivationNone = Activation{ActivationTypeNone, 0}
var ActivationReLU = Activation{ActivationTypeReLU, 0}

func (tx *Tensor) MatMulMatAddVecTAct(ty, tz, tw *Tensor, act *Activation, stream Stream) error {
	if tx.nx != ty.ny || tx.ny != tw.ny || ty.nx != tw.nx || ty.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatMulMatAddVecTAct(ty, tz, tw, act, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) VecTMulMatAddVecTAct(ty, tz, tw *Tensor, act *Activation, stream Stream) error {
	if tx.nx != ty.ny || ty.nx != tw.nx || ty.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecTMulMatAddVecTAct(ty, tz, tw, act, stream, &code)
	return newMtErr(code, "")
}

type CnnPaddingType uint8
type PoolType uint8

const (
	CnnPaddingTypeSame CnnPaddingType = iota
	CnnPaddingTypeValid
)
const (
	PoolTypeMax PoolType = iota
	PoolTypeAvg
)

type Cnn2dConf struct {
	PaddingType CnnPaddingType
	StrideX     uint8
	StrideY     uint8
}

func (c *Cnn2dConf) OutputSize(nxIn, nyIn, nxCore, nyCore uint32) (uint32, uint32) {
	sx, sy := uint32(c.StrideX), uint32(c.StrideY)
	if sx == 0 || sy == 0 {
		return 0, 0
	}
	switch c.PaddingType {
	case CnnPaddingTypeSame:
		return (nxIn + sx - 1) / sx, (nyIn + sy - 1) / sy
	case CnnPaddingTypeValid:
		if nxCore == 0 || nyCore == 0 || nxIn < nxCore || nyIn < nyCore {
			return 0, 0
		}
		return (nxIn-nxCore)/sx + 1, (nyIn-nyCore)/sy + 1
	}
	return 0, 0
}

type Conv2dConf struct {
	Act Activation
	Cnn Cnn2dConf
}

type Pool2dConf struct {
	Typ   PoolType
	CoreX uint8
	CoreY uint8
	Cnn   Cnn2dConf
}

func (tx *Tensor) CubBatchConv2d(ty, tz, tw *Tensor, conf *Conv2dConf, stream Stream) error {
	if tx.nz != ty.nz || tx.nw != tw.nw || ty.nw != tz.nx || ty.nw != tw.nz {
		return unsuitableShapeErr
	} else if nxOut, nyOut := conf.Cnn.OutputSize(tx.nx, tx.ny, ty.nx, ty.ny); tw.nx != nxOut || tw.ny != nyOut {
		return unsuitableShapeErr
	}
	var code int
	tx.mtCubBatchConv2d(ty, tz, tw, conf, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) CubBatchPool2d(ty *Tensor, conf *Pool2dConf, stream Stream) error {
	if tx.nz != ty.nz || tx.nw != ty.nw {
		return unsuitableShapeErr
	}
	nxOut, nyOut := conf.Cnn.OutputSize(tx.nx, tx.ny, uint32(conf.CoreX), uint32(conf.CoreY))
	if ty.nx != nxOut || ty.ny != nyOut {
		return unsuitableShapeErr
	}
	var code int
	tx.mtCubBatchPool2d(ty, conf, stream, &code)
	return newMtErr(code, "")
}
