package neural

import (
	"github.com/ldeng7/matrics/go/matrics"
)

type Pool2dLayer struct {
	*baseLayer
	conf *matrics.Pool2dConf
}

func NewPool2dLayer(name string, shapeIn matrics.TensorShape, conf *matrics.Pool2dConf) (_ *Pool2dLayer, err error) {
	l := &Pool2dLayer{
		&baseLayer{nil, name, []matrics.TensorShape{shapeIn}, []*matrics.Tensor{nil}, []*matrics.Tensor{nil}},
		conf,
	}
	l.self = l

	shapeOut := []uint32{0, 0, shapeIn[2], shapeIn[3]}
	shapeOut[0], shapeOut[1] = conf.Cnn.OutputSize(shapeIn[0], shapeIn[1], uint32(conf.CoreX), uint32(conf.CoreY))
	if l.tsOut[0], err = matrics.NewTensor(shapeOut); nil != err {
		return nil, err
	}
	return l, nil
}

func (l *Pool2dLayer) Run(stream matrics.Stream) error {
	if err := l.beforeRun(); nil != err {
		return err
	}
	return l.tsIn[0].CubBatchPool2d(l.tsOut[0], l.conf, stream)
}

func (l *Pool2dLayer) NewSubsequentConv2dLayerAndConnect(name string, ps *ParameterSource,
	nCore [3]uint32, conf *matrics.Conv2dConf) (*Conv2dLayer, error) {
	return l.cubBatchNewSubsequentConv2dLayerAndConnect(name, ps, nCore, conf)
}

func (l *Pool2dLayer) NewSubsequentFullConnLayerAndConnect(name string, ps *ParameterSource,
	nxOut uint32, act *matrics.Activation) (*FlattenLayer, *FullConnLayer, error) {
	return l.cubBatchNewSubsequentFullConnLayerAndConnect(name, ps, nxOut, act)
}

func (l *Pool2dLayer) NewSubsequentFullConnChainLayerAndConnect(name string, ps *ParameterSource,
	nxIns []uint32, nxOut uint32, act interface{}) (*FlattenLayer, *FullConnChainLayer, error) {
	return l.cubBatchNewSubsequentFullConnChainLayerAndConnect(name, ps, nxIns, nxOut, act)
}
