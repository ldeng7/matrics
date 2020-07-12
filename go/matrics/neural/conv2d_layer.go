package neural

import (
	"github.com/ldeng7/matrics/go/matrics"
)

type Conv2dLayer struct {
	*baseLayer
	conf   *matrics.Conv2dConf
	tw, tb *matrics.Tensor
}

// input: [width, height, depth, batch]
// core: [width, height, depth_out]
func NewConv2dLayer(name string, ps *ParameterSource,
	shapeIn matrics.TensorShape, nCore [3]uint32, conf *matrics.Conv2dConf) (_ *Conv2dLayer, err error) {
	l := &Conv2dLayer{
		&baseLayer{nil, name, []matrics.TensorShape{shapeIn}, []*matrics.Tensor{nil}, []*matrics.Tensor{nil}},
		conf, nil, nil,
	}
	l.self = l
	defer func() {
		if nil != err {
			l.Destroy()
		}
	}()

	shapeOut := []uint32{0, 0, nCore[2], shapeIn[3]}
	shapeOut[0], shapeOut[1] = conf.Cnn.OutputSize(shapeIn[0], shapeIn[1], nCore[0], nCore[1])
	if l.tsOut[0], err = matrics.NewTensor(shapeOut); nil != err {
		return nil, err
	} else if l.tw, err = matrics.NewTensor([]uint32{nCore[0], nCore[1], shapeIn[2], nCore[2]}); nil != err {
		return nil, err
	} else if err = ps.read(l.tw); nil != err {
		return nil, err
	} else if l.tb, err = matrics.NewTensor([]uint32{nCore[2]}); nil != err {
		return nil, err
	} else if err = ps.read(l.tb); nil != err {
		return nil, err
	}
	return l, nil
}

func (l *Conv2dLayer) Destory() {
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

func (l *Conv2dLayer) Run(stream matrics.Stream) error {
	if err := l.beforeRun(); nil != err {
		return err
	}
	return l.tsIn[0].CubBatchConv2d(l.tw, l.tb, l.tsOut[0], l.conf, stream)
}

func (l *Conv2dLayer) NewSubsequentConv2dLayerAndConnect(name string, ps *ParameterSource,
	nCore [3]uint32, conf *matrics.Conv2dConf) (*Conv2dLayer, error) {
	return l.cubBatchNewSubsequentConv2dLayerAndConnect(name, ps, nCore, conf)
}

func (l *Conv2dLayer) NewSubsequentPool2dLayerAndConnect(name string,
	conf *matrics.Pool2dConf) (*Pool2dLayer, error) {
	return l.cubBatchNewSubsequentPool2dLayerAndConnect(name, conf)
}

func (l *Conv2dLayer) NewSubsequentFullConnLayerAndConnect(name string, ps *ParameterSource,
	nxOut uint32, act *matrics.Activation) (*FlattenLayer, *FullConnLayer, error) {
	return l.cubBatchNewSubsequentFullConnLayerAndConnect(name, ps, nxOut, act)
}

func (l *Conv2dLayer) NewSubsequentFullConnChainLayerAndConnect(name string, ps *ParameterSource,
	nxIns []uint32, nxOut uint32, act interface{}) (*FlattenLayer, *FullConnChainLayer, error) {
	return l.cubBatchNewSubsequentFullConnChainLayerAndConnect(name, ps, nxIns, nxOut, act)
}
