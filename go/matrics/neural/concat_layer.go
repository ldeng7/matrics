package neural

import (
	"github.com/ldeng7/matrics/go/matrics"
)

type ConcatLayer struct {
	*baseLayer
	dim uint
}

func NewConcatLayer(name string, shapesIn []matrics.TensorShape, dim uint) (_ *ConcatLayer, err error) {
	l := &ConcatLayer{
		&baseLayer{nil, name, shapesIn, make([]*matrics.Tensor, len(shapesIn)), []*matrics.Tensor{nil}},
		dim,
	}
	l.self = l

	if l.tsOut[0], err = matrics.NewConcatTensor(shapesIn, dim); nil != err {
		return nil, err
	}
	return l, nil
}
func (l *ConcatLayer) Run(stream matrics.Stream) error {
	if err := l.beforeRun(); nil != err {
		return err
	}
	return l.tsOut[0].ConcatFrom(l.tsIn, l.dim)
}
