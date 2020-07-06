package neural

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/ldeng7/matrics/go/matrics"
)

type ParameterSource struct {
	Reader    io.Reader
	ByteOrder binary.ByteOrder
}

func (ps *ParameterSource) read(t *matrics.Tensor) error {
	sl := t.Data
	defer matrics.DestroySlice(&sl)
	slu := make([]uint32, len(sl))
	if err := binary.Read(ps.Reader, ps.ByteOrder, slu); nil != err {
		return err
	}
	for i, u := range slu {
		sl[i] = math.Float32frombits(u)
	}
	return nil
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

type ILayer interface {
	Destroy()
	Name() string
	SetInput(t *matrics.Tensor, port uint) error
	Outputs() []*matrics.Tensor
	Connect(src ILayer, srcPort, destPort uint) error
	Run(matrics.Stream) error
}

type layerAccpetError struct {
	srcLayer  ILayer
	destLayer ILayer
	msg       string
}

func (e *layerAccpetError) Error() string {
	return fmt.Sprintf("layer %s accepting %s: %s", e.destLayer.Name(), e.srcLayer.Name(), e.msg)
}

var invalidPortErr error = errors.New("invalid port")
var unsuitableDimLenErr error = errors.New("unsuitable dimension length")
var unsuitableShapeErr error = errors.New("unsuitable shape")
var layerNilInputTensorErr error = errors.New("nil input tensor")

type baseLayer struct {
	self  ILayer
	name  string
	tsIn  []*matrics.Tensor
	tsOut []*matrics.Tensor
}

func (l *baseLayer) Destroy() {
	if nil == l.self {
		return
	}
	for _, t := range l.tsOut {
		if nil != t {
			t.Destroy()
		}
	}
	l.tsIn, l.tsOut = nil, nil
	l.self = nil
}

func (l *baseLayer) Name() string {
	return l.name
}

func (l *baseLayer) Outputs() []*matrics.Tensor {
	return l.tsOut
}

func (l *baseLayer) Connect(src ILayer, srcPort, destPort uint) error {
	ts := src.Outputs()
	if srcPort >= uint(len(ts)) {
		return &layerAccpetError{src, l.self, "invalid port"}
	}
	if err := l.self.SetInput(ts[srcPort], destPort); nil != err {
		return &layerAccpetError{src, l.self, err.Error()}
	}
	return nil
}

type inputLayer struct {
	*baseLayer
}

func newInputLayer(shapes [][]uint64) (*inputLayer, error) {
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

type Machine struct {
	layers     []ILayer
	inputLayer *inputLayer
}

func NewMachine(inputShapes [][]uint64) (*Machine, error) {
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

func (m *Machine) AddLayer(layer ILayer) {
	m.layers = append(m.layers, layer)
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
