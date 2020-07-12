package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ldeng7/matrics/go/matrics"
	"github.com/ldeng7/matrics/go/matrics/neural"
)

func runMachine(stream matrics.Stream) error {
	m, err := neural.NewMachine([][]uint32{{3, 2}})
	if nil != err {
		return err
	}
	defer m.Destroy()

	ss := []float32{0.3, 0.4, 0.5, 0.6,
		0.7, 0.8, 0.9, 1,
		1.1, 1.2, 1.3, 1.4,
		1, -10, 1.2, -5,
		2.1, 2.2, 2.3, 2.4, 2.5,
		2.6, 2.7, 2.8, 2.9, 3,
		3.1, 3.2, 3.3, 3.4, 3.5,
		3.6, 3.7, 3.8, 3.9, 4,
		1, -100, 1, -100, 1,
	}
	us := []uint32{}
	for _, s := range ss {
		us = append(us, math.Float32bits(s))
	}
	buf := bytes.NewBuffer(nil)
	if err := binary.Write(buf, binary.BigEndian, us); nil != err {
		return err
	}
	bs := buf.Bytes()

	ps := &neural.ParameterSource{bytes.NewReader(bs), binary.BigEndian}
	act := &matrics.Activation{matrics.ActivationTypeReLU, 0}
	l, err := neural.NewFullConnChainLayer("1", ps, []uint32{3, 4}, 2, 5, act)
	if nil != err {
		return err
	}
	m.AddLayer(l)

	if err := l.Connect(m.InputLayer(), 0, 0); nil != err {
		return err
	}

	sl := m.Inputs()[0].Data
	sl[0], sl[1], sl[2], sl[3], sl[4], sl[5] = 0.5, 1, 1.5, 2, 2.5, 3
	defer matrics.DestroySlice(&sl)
	if err := m.Run(stream); nil != err {
		return err
	}
	fmt.Printf("%+v\n", l.Outputs()[0].Data)
	return nil
}

func runCubConv2d(stream matrics.Stream) error {
	var wc, wi, hi uint32 = 7, 22, 22
	stride, pad := uint8(2), matrics.CnnPaddingTypeSame
	x, err := matrics.NewTensor([]uint32{wi, hi, 1, 1})
	if nil != err {
		return err
	}
	defer x.Destroy()
	y, err := matrics.NewTensor([]uint32{wc, wc, 1, 1})
	if nil != err {
		return err
	}
	defer y.Destroy()
	z, err := matrics.NewTensor([]uint32{1})
	if nil != err {
		return err
	}
	defer z.Destroy()

	conf := &matrics.Conv2dConf{
		matrics.ActivationNone,
		matrics.Cnn2dConf{pad, stride, stride},
	}
	wx, wy := conf.Cnn.OutputSize(wi, hi, wc, wc)
	w, err := matrics.NewTensor([]uint32{wx, wy, 1, 1})
	if nil != err {
		return err
	}
	defer w.Destroy()

	for i := uint32(0); i < wi*hi; i++ {
		x.Data[i] = 1 + float32(i)
	}
	for i := uint32(0); i < wc*wc; i++ {
		y.Data[i] = 1 + float32(i)
	}
	for i := 0; i < 1; i++ {
		z.Data[i] = 0
	}

	if err := x.CubBatchConv2d(y, z, w, conf, stream); nil != err {
		return err
	}
	fmt.Printf("%+v\n", w.Data[:wx])
	fmt.Printf("%+v\n", w.Data[len(w.Data)-int(wx):])
	return nil
}

func runConcat(stream matrics.Stream) error {
	x, _ := matrics.NewTensor([]uint32{2, 2, 2, 1})
	for i := 0; i < 2*2*2; i++ {
		x.Data[i] = float32(i + 1)
	}
	y, _ := matrics.NewTensor([]uint32{2, 2, 2, 1})
	for i := 0; i < 2*2*2; i++ {
		y.Data[i] = float32(i + 9)
	}
	z, _ := matrics.NewConcatTensor([][4]uint32{x.Shape(), y.Shape()}, 3)
	ts := []*matrics.Tensor{x, y}

	z.ConcatFrom(ts, 3)
	fmt.Printf("%+v\n", z.Data[:])
	return nil
}

func main() {
	if err := matrics.Init(); nil != err {
		println(err.Error())
		return
	}
	defer matrics.Uninit()

	stream, err := matrics.NewStream()
	if nil != err {
		println(err.Error())
		return
	}
	defer stream.Destroy()

	if err := runCubConv2d(stream); nil != err {
		println(err.Error())
	}
}
