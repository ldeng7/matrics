package neural

import (
	"errors"
	"fmt"

	"github.com/ldeng7/matrics/go/matrics"
)

type FullConnLayer struct {
	*baseLayer
	nxIn, nyIn uint64
	nxOut      uint64
	act        *matrics.Activation
	tw, tb     *matrics.Tensor
}

func NewFullConnLayer(name string, ps *ParameterSource,
	nxIn, nyIn, nxOut uint64, act *matrics.Activation) (_ *FullConnLayer, err error) {
	l := &FullConnLayer{
		&baseLayer{name: name, tsIn: []*matrics.Tensor{nil}, tsOut: []*matrics.Tensor{nil}},
		nxIn, nyIn, nxOut, act, nil, nil,
	}
	l.self = l
	defer func() {
		if nil != err {
			l.Destroy()
		}
	}()

	shapeOut := make([]uint64, 1, 2)
	shapeOut[0] = nxOut
	if nyIn != 1 {
		shapeOut = append(shapeOut, nyIn)
	}
	if l.tsOut[0], err = matrics.NewTensor(shapeOut); nil != err {
		return nil, err
	} else if l.tw, err = matrics.NewTensor([]uint64{nxOut, nxIn}); nil != err {
		return nil, err
	} else if err = ps.read(l.tw); nil != err {
		return nil, err
	} else if l.tb, err = matrics.NewTensor([]uint64{nxOut}); nil != err {
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

func (l *FullConnLayer) SetInput(t *matrics.Tensor, port uint) error {
	switch port {
	case 0:
		if nd := t.DimLen(); nd != 1 && nd != 2 {
			return unsuitableDimLenErr
		} else if shape := t.Shape(); shape[0] != l.nxIn || shape[1] != l.nyIn {
			return unsuitableShapeErr
		}
		l.tsIn[0] = t
		return nil
	}
	return invalidPortErr
}

func (l *FullConnLayer) Run(stream matrics.Stream) error {
	tIn, tOut := l.tsIn[0], l.tsOut[0]
	if nil == tIn {
		return layerNilInputTensorErr
	}
	if tIn.DimLen() == 1 {
		if err := tIn.VecTMulMatAddVecTAct(l.tw, l.tb, tOut, l.act, stream); nil != err {
			return err
		}
	} else {
		if err := tIn.MatMulMatAddVecTAct(l.tw, l.tb, tOut, l.act, stream); nil != err {
			return err
		}
	}
	return nil
}

func (l *FullConnLayer) NewSubsequentFullConnLayerAndConnect(name string, ps *ParameterSource,
	nxOut uint64, act *matrics.Activation) (*FullConnLayer, error) {
	ls, err := NewFullConnLayer(name, ps, l.nxOut, l.nyIn, nxOut, act)
	if nil != err {
		return nil, err
	}
	ls.SetInput(l.tsOut[0], 0)
	return ls, nil
}

type FullConnChainLayer struct {
	*baseLayer
	innerLayers []*FullConnLayer
}

func NewFullConnChainLayer(name string, ps *ParameterSource,
	nxIns []uint64, nyIn, nxOut uint64, act interface{}) (_ *FullConnChainLayer, err error) {
	n := len(nxIns)
	if n <= 1 {
		return nil, errors.New("invalid nxIns length")
	}

	l := &FullConnChainLayer{
		&baseLayer{name: name, tsIn: []*matrics.Tensor{nil}},
		make([]*FullConnLayer, n),
	}
	l.self = l
	defer func() {
		if nil != err {
			l.Destroy()
		}
	}()

	l.innerLayers = make([]*FullConnLayer, n)
	layerName := func(i int) string {
		return fmt.Sprintf("%s.fullconn%d", name, i)
	}
	for i := 0; i < n-1; i++ {
		var a *matrics.Activation
		if a, err = activationAt(act, i); nil != err {
			return nil, err
		}
		l.innerLayers[i], err = NewFullConnLayer(layerName(i), ps, nxIns[i], nyIn, nxIns[i+1], a)
		if nil != err {
			return nil, fmt.Errorf("failed to new layer %d: %s", i, err.Error())
		}
	}
	l.innerLayers[n-1], err = NewFullConnLayer(layerName(n-1), ps, nxIns[n-1], nyIn, nxOut, matrics.ActivationNone)
	if nil != err {
		return nil, fmt.Errorf("failed to new layer %d: %s", n-1, err.Error())
	}

	for i := 1; i < n; i++ {
		l.innerLayers[i].SetInput(l.innerLayers[i-1].Outputs()[0], 0)
	}
	return l, nil
}

func (l *FullConnChainLayer) Destroy() {
	if nil == l.self {
		return
	}
	for i := len(l.innerLayers) - 1; i >= 0; i-- {
		if innerLayer := l.innerLayers[i]; nil != innerLayer {
			innerLayer.Destroy()
		}
	}
	l.baseLayer.Destroy()
}

func (l *FullConnChainLayer) SetInput(t *matrics.Tensor, port uint) error {
	return l.innerLayers[0].SetInput(t, port)
}

func (l *FullConnChainLayer) Outputs() []*matrics.Tensor {
	return l.innerLayers[len(l.innerLayers)-1].Outputs()
}

func (l *FullConnChainLayer) Run(stream matrics.Stream) error {
	for _, innerLayer := range l.innerLayers {
		if err := innerLayer.Run(stream); nil != err {
			return err
		}
	}
	return nil
}

func (l *FullConnChainLayer) ChainOutputs() [][]*matrics.Tensor {
	o := make([][]*matrics.Tensor, len(l.innerLayers))
	for i, innerLayer := range l.innerLayers {
		o[i] = innerLayer.Outputs()
	}
	return o
}
