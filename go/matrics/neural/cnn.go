package neural

import (
	"github.com/ldeng7/matrics/go/matrics"
)

type Conv2dLayer struct {
	*baseLayer
	nIn    [4]uint64
	conf   *matrics.Conv2dConf
	tw, tb *matrics.Tensor
}

// input: [depth, width, height, batch]
// core: [width, height, depth_out]
func NewConv2dLayerLayer(name string, ps *ParameterSource,
	nIn [4]uint64, nCore [3]uint64, conf *matrics.Conv2dConf) (_ *Conv2dLayer, err error) {
	l := &Conv2dLayer{
		&baseLayer{name: name, tsIn: []*matrics.Tensor{nil}, tsOut: []*matrics.Tensor{nil}},
		nIn, conf, nil, nil,
	}
	l.self = l
	defer func() {
		if nil != err {
			l.Destroy()
		}
	}()

	shapeOut := make([]uint64, 3, 4)
	shapeOut[0] = nCore[2]
	shapeOut[1], shapeOut[2] = conf.Cnn.OutputSize(nIn[1], nIn[2], nCore[0], nCore[1])
	if nIn[3] != 1 {
		shapeOut = append(shapeOut, nIn[3])
	}
	if l.tsOut[0], err = matrics.NewTensor(shapeOut); nil != err {
		return nil, err
	} else if l.tw, err = matrics.NewTensor([]uint64{nIn[0], nCore[0], nCore[1], nCore[2]}); nil != err {
		return nil, err
	} else if err = ps.read(l.tw); nil != err {
		return nil, err
	} else if l.tb, err = matrics.NewTensor([]uint64{nCore[2]}); nil != err {
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

func (l *Conv2dLayer) SetInput(t *matrics.Tensor, port uint) error {
	switch port {
	case 0:
		if nd := t.DimLen(); nd != 3 && nd != 4 {
			return unsuitableDimLenErr
		} else if t.Shape() != l.nIn {
			return unsuitableShapeErr
		}
		l.tsIn[0] = t
		return nil
	}
	return invalidPortErr
}

func (l *Conv2dLayer) Run(stream matrics.Stream) error {
	tIn, tOut := l.tsIn[0], l.tsOut[0]
	if nil == tIn {
		return layerNilInputTensorErr
	}
	if tIn.DimLen() == 3 {
		if err := tIn.CubConv2d(l.tw, l.tb, tOut, l.conf, stream); nil != err {
			return err
		}
	} else {
		if err := tIn.TesConv2d(l.tw, l.tb, tOut, l.conf, stream); nil != err {
			return err
		}
	}
	return nil
}

func (l *Conv2dLayer) NewSubsequentConv2dLayerAndConnect(name string, ps *ParameterSource,
	nCore [3]uint64, conf *matrics.Conv2dConf) (*Conv2dLayer, error) {
	ls, err := NewConv2dLayerLayer(name, ps, l.Outputs()[0].Shape(), nCore, conf)
	if nil != err {
		return nil, err
	}
	ls.SetInput(l.tsOut[0], 0)
	return ls, nil
}

type CnnPoolLayer struct {
	*baseLayer
	nIn   [4]uint64
	nCore [2]uint64
	conf  *matrics.CnnPoolConf
}

func NewCnnPoolLayerLayer(name string, nIn [4]uint64, nCore [2]uint64, conf *matrics.CnnPoolConf) (_ *CnnPoolLayer, err error) {
	l := &CnnPoolLayer{
		&baseLayer{name: name, tsIn: []*matrics.Tensor{nil}, tsOut: []*matrics.Tensor{nil}},
		nIn, nCore, conf,
	}
	l.self = l

	shapeOut := make([]uint64, 3, 4)
	shapeOut[0] = nIn[0]
	shapeOut[1], shapeOut[2] = conf.Cnn.OutputSize(nIn[1], nIn[2], nCore[0], nCore[1])
	if nIn[3] != 1 {
		shapeOut = append(shapeOut, nIn[3])
	}
	if l.tsOut[0], err = matrics.NewTensor(shapeOut); nil != err {
		return nil, err
	}
	return l, nil
}

func (l *CnnPoolLayer) SetInput(t *matrics.Tensor, port uint) error {
	switch port {
	case 0:
		if nd := t.DimLen(); nd != 3 && nd != 4 {
			return unsuitableDimLenErr
		} else if t.Shape() != l.nIn {
			return unsuitableShapeErr
		}
		l.tsIn[0] = t
		return nil
	}
	return invalidPortErr
}

func (l *CnnPoolLayer) Run(stream matrics.Stream) error {
	tIn, tOut := l.tsIn[0], l.tsOut[0]
	if nil == tIn {
		return layerNilInputTensorErr
	}
	if tIn.DimLen() == 3 {
		if err := tIn.CubCnnPool(tOut, l.conf, stream); nil != err {
			return err
		}
	} else {
		if err := tIn.TesCnnPool(tOut, l.conf, stream); nil != err {
			return err
		}
	}
	return nil
}
