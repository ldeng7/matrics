package neural

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/ldeng7/matrics/go/matrics"
)

var invalidPortErr error = errors.New("invalid port")
var unsuitableDimLenErr error = errors.New("unsuitable dimension length")
var unsuitableShapeErr error = errors.New("unsuitable shape")
var layerNilInputTensorErr error = errors.New("nil input tensor")

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

type baseLayer struct {
	self     ILayer
	name     string
	shapesIn []matrics.TensorShape
	tsIn     []*matrics.Tensor
	tsOut    []*matrics.Tensor
}

func (l *baseLayer) Destroy() {
	if nil == l.self {
		return
	}
	for i, _ := range l.tsIn {
		l.tsIn[i] = nil
	}
	for i, _ := range l.tsOut {
		if nil != l.tsOut[i] {
			l.tsOut[i].Destroy()
			l.tsOut[i] = nil
		}
	}
	l.self = nil
}

func (l *baseLayer) Name() string {
	return l.name
}

func (l *baseLayer) SetInput(t *matrics.Tensor, port uint) error {
	if port >= uint(len(l.tsIn)) {
		return invalidPortErr
	}
	if t.Shape() != l.shapesIn[port] {
		return unsuitableShapeErr
	}
	l.tsIn[port] = t
	return nil
}

func (l *baseLayer) Outputs() []*matrics.Tensor {
	return l.tsOut
}

func (l *baseLayer) Connect(srcLayer ILayer, srcPort, destPort uint) error {
	ts := srcLayer.Outputs()
	if srcPort >= uint(len(ts)) {
		return &layerAccpetError{srcLayer, l.self, invalidPortErr.Error()}
	}
	if err := l.self.SetInput(ts[srcPort], destPort); nil != err {
		return &layerAccpetError{srcLayer, l.self, err.Error()}
	}
	return nil
}

func (l *baseLayer) beforeRun() error {
	for _, tIn := range l.tsIn {
		if nil == tIn {
			return layerNilInputTensorErr
		}
	}
	return nil
}

func (l *baseLayer) cubBatchNewSubsequentConv2dLayerAndConnect(name string, ps *ParameterSource,
	nCore [3]uint32, conf *matrics.Conv2dConf) (*Conv2dLayer, error) {
	ls, err := NewConv2dLayer(name, ps, l.tsOut[0].Shape(), nCore, conf)
	if nil != err {
		return nil, err
	}
	ls.SetInput(l.tsOut[0], 0)
	return ls, nil
}

func (l *baseLayer) cubBatchNewSubsequentPool2dLayerAndConnect(name string,
	conf *matrics.Pool2dConf) (*Pool2dLayer, error) {
	ls, err := NewPool2dLayer(name, l.tsOut[0].Shape(), conf)
	if nil != err {
		return nil, err
	}
	ls.SetInput(l.tsOut[0], 0)
	return ls, nil
}

func (l *baseLayer) cubBatchNewSubsequentFlattenLayerAndConnect(name string) (*FlattenLayer, error) {
	lf, err := NewFlattenLayer(name, l.tsOut[0].Shape(), 0, 2, true)
	if nil != err {
		return nil, err
	}
	lf.SetInput(l.tsOut[0], 0)
	return lf, nil
}

func (l *baseLayer) cubBatchNewSubsequentFullConnLayerAndConnect(name string, ps *ParameterSource,
	nxOut uint32, act *matrics.Activation) (*FlattenLayer, *FullConnLayer, error) {
	lf, err := l.cubBatchNewSubsequentFlattenLayerAndConnect("flatten-pre-" + name)
	if nil != err {
		return nil, nil, err
	}
	shape := lf.tsOut[0].Shape()
	ls, err := NewFullConnLayer(name, ps, shape[0], shape[1], nxOut, act)
	if nil != err {
		lf.Destroy()
		return nil, nil, err
	}
	ls.SetInput(lf.tsOut[0], 0)
	return lf, ls, nil
}

func (l *baseLayer) cubBatchNewSubsequentFullConnChainLayerAndConnect(name string, ps *ParameterSource,
	nxIns []uint32, nxOut uint32, act interface{}) (*FlattenLayer, *FullConnChainLayer, error) {
	lf, err := l.cubBatchNewSubsequentFlattenLayerAndConnect("flatten-pre-" + name)
	if nil != err {
		return nil, nil, err
	}
	shape := lf.tsOut[0].Shape()
	ls, err := NewFullConnChainLayer(name, ps, append([]uint32{shape[0]}, nxIns...), shape[1], nxOut, act)
	if nil != err {
		lf.Destroy()
		return nil, nil, err
	}
	ls.SetInput(lf.tsOut[0], 0)
	return lf, ls, nil
}
