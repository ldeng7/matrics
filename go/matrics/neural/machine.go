package neural

import (
	"github.com/ldeng7/matrics/go/matrics"
)

type Machine struct {
	layers     []ILayer
	inputLayer *inputLayer
}

func NewMachine(inputShapes [][]uint32) (*Machine, error) {
	var err error
	m := &Machine{layers: make([]ILayer, 0, 16)}
	if m.inputLayer, err = newInputLayer(inputShapes); nil != err {
		return nil, err
	}
	return m, nil
}

func (m *Machine) Destroy() {
	for i := len(m.layers) - 1; i >= 0; i-- {
		if layer := m.layers[i]; nil != layer {
			layer.Destroy()
		}
	}
	if nil != m.inputLayer {
		m.inputLayer.Destroy()
	}
}

func (m *Machine) AddLayer(layer ...ILayer) {
	m.layers = append(m.layers, layer...)
}

func (m *Machine) Run(stream matrics.Stream) error {
	for _, layer := range m.layers {
		if err := layer.Run(stream); nil != err {
			return err
		}
	}
	return nil
}

func (m *Machine) InputLayer() ILayer {
	return m.inputLayer
}

func (m *Machine) Inputs() []*matrics.Tensor {
	return m.inputLayer.tsOut
}
