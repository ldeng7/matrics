package neural

import (
	"github.com/ldeng7/matrics/go/matrics"
)

type FlattenLayer struct {
	*baseLayer
	startDim, endDim uint
}

func NewFlattenLayer(name string, shapeIn matrics.TensorShape,
	startDim uint, endDim uint, collapse bool) (_ *FlattenLayer, err error) {
	l := &FlattenLayer{
		&baseLayer{nil, name, []matrics.TensorShape{shapeIn}, []*matrics.Tensor{nil}, []*matrics.Tensor{nil}},
		startDim, endDim,
	}
	l.self = l

	if l.tsOut[0], err = matrics.NewFlattenTensor(shapeIn, startDim, endDim, collapse); nil != err {
		return nil, err
	}
	return l, nil
}

func (l *FlattenLayer) Run(stream matrics.Stream) error {
	if err := l.beforeRun(); nil != err {
		return err
	}
	copy(l.tsOut[0].Data, l.tsIn[0].Data)
	return nil
}
