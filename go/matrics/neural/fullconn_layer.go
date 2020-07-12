package neural

import (
	"github.com/ldeng7/matrics/go/matrics"
)

type FullConnLayer struct {
	*baseLayer
	nyIn, nxOut uint32
	act         *matrics.Activation
	tw, tb      *matrics.Tensor
}

func NewFullConnLayer(name string, ps *ParameterSource,
	nxIn, nyIn, nxOut uint32, act *matrics.Activation) (_ *FullConnLayer, err error) {
	l := &FullConnLayer{
		&baseLayer{nil, name, []matrics.TensorShape{{nxIn, nyIn, 1, 1}}, []*matrics.Tensor{nil}, []*matrics.Tensor{nil}},
		nyIn, nxOut, act, nil, nil,
	}
	l.self = l
	defer func() {
		if nil != err {
			l.Destroy()
		}
	}()

	if l.tsOut[0], err = matrics.NewTensor([]uint32{nxOut, nyIn}); nil != err {
		return nil, err
	} else if l.tw, err = matrics.NewTensor([]uint32{nxOut, nxIn}); nil != err {
		return nil, err
	} else if err = ps.read(l.tw); nil != err {
		return nil, err
	} else if l.tb, err = matrics.NewTensor([]uint32{nxOut}); nil != err {
		return nil, err
	} else if err = ps.read(l.tb); nil != err {
		return nil, err
	}
	return l, nil
}

func (l *FullConnLayer) Destory() {
	if nil == l.self {
		return
	}
	if nil != l.tw {
		l.tw.Destroy()
	}
	if nil != l.tb {
		l.tb.Destroy()
	}
	l.baseLayer.Destroy()
}

func (l *FullConnLayer) Run(stream matrics.Stream) error {
	if err := l.beforeRun(); nil != err {
		return err
	}
	tIn, tOut := l.tsIn[0], l.tsOut[0]
	if tIn.AvailableDimLen() <= 1 {
		return tIn.VecTMulMatAddVecTAct(l.tw, l.tb, tOut, l.act, stream)
	}
	return tIn.MatMulMatAddVecTAct(l.tw, l.tb, tOut, l.act, stream)
}

func (l *FullConnLayer) NewSubsequentFullConnLayerAndConnect(name string, ps *ParameterSource,
	nxOut uint32, act *matrics.Activation) (*FullConnLayer, error) {
	ls, err := NewFullConnLayer(name, ps, l.nxOut, l.nyIn, nxOut, act)
	if nil != err {
		return nil, err
	}
	ls.SetInput(l.tsOut[0], 0)
	return ls, nil
}
