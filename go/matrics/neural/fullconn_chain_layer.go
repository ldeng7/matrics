package neural

import (
	"errors"
	"fmt"

	"github.com/ldeng7/matrics/go/matrics"
)

type FullConnChainLayer struct {
	*baseLayer
	innerLayers []*FullConnLayer
}

func activationAt(act interface{}, i int) (*matrics.Activation, error) {
	if nil != act {
		switch a := act.(type) {
		case *matrics.Activation:
			return a, nil
		case []*matrics.Activation:
			if i < len(a) {
				return a[i], nil
			}
			return nil, errors.New("invalid length of []Activation")
		}
		return nil, errors.New("invalid type of activation")
	}
	return nil, nil
}

func NewFullConnChainLayer(name string, ps *ParameterSource,
	nxIns []uint32, nyIn, nxOut uint32, act interface{}) (_ *FullConnChainLayer, err error) {
	n := len(nxIns)
	if n <= 1 {
		return nil, errors.New("invalid nxIns length")
	}

	l := &FullConnChainLayer{
		&baseLayer{name: name},
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
		a, err := activationAt(act, i)
		if nil != err {
			return nil, err
		}
		l.innerLayers[i], err = NewFullConnLayer(layerName(i), ps, nxIns[i], nyIn, nxIns[i+1], a)
		if nil != err {
			return nil, fmt.Errorf("failed to new layer %d: %s", i, err.Error())
		}
	}
	l.innerLayers[n-1], err = NewFullConnLayer(layerName(n-1), ps, nxIns[n-1], nyIn, nxOut, &matrics.ActivationNone)
	if nil != err {
		return nil, fmt.Errorf("failed to new layer %d: %s", n-1, err.Error())
	}

	for i := 1; i < n; i++ {
		l.innerLayers[i].SetInput(l.innerLayers[i-1].tsOut[0], 0)
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
	return l.innerLayers[len(l.innerLayers)-1].tsOut
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
		o[i] = innerLayer.tsOut
	}
	return o
}
