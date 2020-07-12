package neural

import (
	"fmt"

	"github.com/ldeng7/matrics/go/matrics"
)

type inputLayer struct {
	*baseLayer
}

func newInputLayer(shapes [][]uint32) (*inputLayer, error) {
	var err error
	l := &inputLayer{&baseLayer{name: "input"}}
	l.self = l

	l.tsOut = make([]*matrics.Tensor, len(shapes))
	for i, shape := range shapes {
		if l.tsOut[i], err = matrics.NewTensor(shape); nil != err {
			l.Destroy()
			return nil, fmt.Errorf("failed to new input tensor #%d: %s", i, err.Error())
		}
	}
	return l, nil
}

func (l *inputLayer) SetInput(t *matrics.Tensor, port uint) error { return nil }
func (l *inputLayer) Run(stream matrics.Stream) error             { return nil }
