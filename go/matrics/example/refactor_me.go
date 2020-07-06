package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ldeng7/matrics/go/matrics"
	"github.com/ldeng7/matrics/go/matrics/neural"
)

const (
	N        = 20240
	W, H, W1 = 256 * 256, 10, 12
)

/*
func runMatMulVec(stream matrics.Stream) error {
	x, err := matrics.NewTensor(W, H, 1, 1)
	if nil != err {
		return err
	}
	defer x.Destroy()
	y, err := matrics.NewTensor(W, 1, 1, 1)
	if nil != err {
		return err
	}
	defer y.Destroy()
	z, err := matrics.NewTensor(H, 1, 1, 1)
	if nil != err {
		return err
	}
	defer z.Destroy()
	buf, err := x.NewMatFlatBuffer()
	if nil != err {
		return err
	}
	defer buf.Destroy()

	rand.Seed(time.Now().Unix())
	for i := 0; i < W*H; i++ {
		x.Data[i] = rand.Float32()
	}
	for i := 0; i < W; i++ {
		y.Data[i] = rand.Float32()
	}

	t := time.Now()
	for i := 0; i < 30; i++ {
		x.MatMulVec(y, z, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(z.Data[0], z.Data[1], z.Data[H-1])

	for i := 0; i < H; i++ {
		z.Data[i] = 0
	}
	t = time.Now()
	for i := 0; i < 30; i++ {
		if err := x.MatFlatMulVec(y, z, buf, stream); nil != err {
			return err
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(z.Data[0], z.Data[1], z.Data[H-1])
	return nil
}

func runVecTMulMat(stream matrics.Stream) error {
	x, err := matrics.NewTensor(W, 1, 1, 1)
	if nil != err {
		return err
	}
	defer x.Destroy()
	y, err := matrics.NewTensor(W1, W, 1, 1)
	if nil != err {
		return err
	}
	defer y.Destroy()
	z, err := matrics.NewTensor(W1, 1, 1, 1)
	if nil != err {
		return err
	}
	defer z.Destroy()

	rand.Seed(time.Now().Unix())
	for i := 0; i < W; i++ {
		x.Data[i] = rand.Float32()
	}
	for i := 0; i < W1*W; i++ {
		y.Data[i] = rand.Float32()
	}

	t := time.Now()
	for i := 0; i < 30; i++ {
		x.VecTMulMat(y, z, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(z.Data[0], z.Data[1], z.Data[W1-1])

		for i := 0; i < W1; i++ {
			z.Data[i] = 0
		}
		t = time.Now()
		for i := 0; i < 30; i++ {
			if err := x.MatFlatMulVec(y, z, buf, stream); nil != err {
				return err
			}
		}
		println(time.Now().Sub(t).Microseconds())
		println(z.Data[0], z.Data[1], z.Data[H-1])
	return nil
}

func runVecPowScalar(stream matrics.Stream) error {
	vx, err := matrics.NewTensor(N)
	if nil != err {
		return err
	}
	defer vx.Destroy()
	vy, err := matrics.NewTensor(N)
	if nil != err {
		return err
	}
	defer vy.Destroy()

	rand.Seed(time.Now().Unix())
	x, y := make([]float32, N), make([]float32, N)
	for i := 0; i < N; i++ {
		x[i] = rand.Float32()
	}
	copy(vx.Data, x)

	t := time.Now()
	for i := 0; i < 1; i++ {
		for j := 0; j < N; j++ {
			y[j] = float32(math.Pow(float64(x[j]), 1.732))
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(y[0], y[1], y[N-1])

	t = time.Now()
	for i := 0; i < 200; i++ {
		vx.PowScalar(1.732, vy, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(vy.Data[0], vy.Data[1], vy.Data[N-1])
	return nil
}

func runVecAddVec(stream matrics.Stream) error {
	vx, err := matrics.NewTensor(N)
	if nil != err {
		return err
	}
	defer vx.Destroy()
	vy, err := matrics.NewTensor(N)
	if nil != err {
		return err
	}
	defer vy.Destroy()

	rand.Seed(time.Now().Unix())
	x, y := make([]float32, N), make([]float32, N)
	for i := 0; i < N; i++ {
		x[i], y[i] = rand.Float32(), rand.Float32()
	}
	copy(vx.Data, x)
	copy(vy.Data, y)

	t := time.Now()
	for i := 0; i < 100; i++ {
		for j := 0; j < N; j++ {
			x[j] += y[j]
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(x[0], x[1], x[N-1])

	t = time.Now()
	for i := 0; i < 100; i++ {
		vx.AddVec(vy, vx, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(vx.Data[0], vx.Data[1], vx.Data[N-1])
	return nil
}

func runVecSum(stream matrics.Stream) error {
	vx, err := matrics.NewTensor(N)
	if nil != err {
		return err
	}
	defer vx.Destroy()
	buf, err := vx.NewAccBuffer()
	if nil != err {
		return err
	}
	defer buf.Destroy()

	rand.Seed(time.Now().Unix())
	x := make([]float32, N)
	for i := 0; i < N; i++ {
		x[i] = rand.Float32()
	}
	copy(vx.Data, x)

	var r float32
	t := time.Now()
	for i := 0; i < 100; i++ {
		r = 0
		for j := 0; j < N; j++ {
			r += x[j]
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(r)

	t = time.Now()
	for i := 0; i < 100; i++ {
		r, _ = vx.Sum(buf, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(r)
	return nil
}

func runVecDot(stream matrics.Stream) error {
	vx, err := matrics.NewTensor(N)
	if nil != err {
		return err
	}
	defer vx.Destroy()
	vy, err := matrics.NewTensor(N)
	if nil != err {
		return err
	}
	defer vy.Destroy()
	buf, err := vx.NewAccBuffer()
	if nil != err {
		return err
	}
	defer buf.Destroy()

	rand.Seed(time.Now().Unix())
	x, y := make([]float32, N), make([]float32, N)
	for i := 0; i < N; i++ {
		x[i], y[i] = rand.Float32(), rand.Float32()
	}
	copy(vx.Data, x)
	copy(vy.Data, y)

	var r float32
	t := time.Now()
	for i := 0; i < 100; i++ {
		r = 0
		for j := 0; j < N; j++ {
			r += x[j] * y[j]
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(r)

	t = time.Now()
	for i := 0; i < 100; i++ {
		r, _ = vx.Dot(vy, buf, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(r)
	return nil
}

func runMatMulMat(stream matrics.Stream) error {
	x, err := matrics.NewTensor(W, H)
	if nil != err {
		return err
	}
	defer x.Destroy()
	my, err := matrics.NewTensor(W1, W)
	if nil != err {
		return err
	}
	defer my.Destroy()
	mz, err := matrics.NewTensor(W1, H)
	if nil != err {
		return err
	}
	defer mz.Destroy()

	rand.Seed(time.Now().Unix())
	x, y, z := make([]float32, W*H), make([]float32, W1*W), make([]float32, W1*H)
	for i := 0; i < W*H; i++ {
		x[i] = rand.Float32()
	}
	for i := 0; i < W1*W; i++ {
		y[i] = rand.Float32()
	}
	copy(x.Data, x)
	copy(my.Data, y)

	t := time.Now()
	for i := 0; i < 3; i++ {
		for j := 0; j < H; j++ {
			for k := 0; k < W1; k++ {
				var s float32
				for h := 0; h < W; h++ {
					s += x[W*j+h] * y[W1*h+k]
				}
				z[W1*j+k] = s
			}
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(z[0], z[1], z[W1*H-2], z[W1*H-1])

	t = time.Now()
	for i := 0; i < 3; i++ {
		x.MulMat(my, mz, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(mz.Data[0], mz.Data[1], mz.Data[W1*H-2], mz.Data[W1*H-1])
	return nil
}

func runMatT(stream matrics.Stream) error {
	x, err := matrics.NewTensor(W, H)
	if nil != err {
		return err
	}
	defer x.Destroy()
	my, err := matrics.NewTensor(H, W)
	if nil != err {
		return err
	}
	defer my.Destroy()

	rand.Seed(time.Now().Unix())
	x, y := make([]float32, W*H), make([]float32, W*H)
	for i := 0; i < W*H; i++ {
		x[i] = rand.Float32()
	}
	copy(x.Data, x)

	t := time.Now()
	for i := 0; i < 10; i++ {
		for j := 0; j < H; j++ {
			for k := 0; k < W; k++ {
				y[H*k+j] = x[W*j+k]
			}
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(y[0], y[1], y[W], y[W*H-2], y[W*H-1])

	t = time.Now()
	for i := 0; i < 3; i++ {
		x.T(my, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(my.Data[0], my.Data[1], my.Data[W], my.Data[W*H-2], my.Data[W*H-1])
	return nil
}
*/

func runMachine(stream matrics.Stream) error {
	m, err := neural.NewMachine([][]uint64{{3, 2}})
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
	l, err := neural.NewFullConnChainLayer("1", ps, []uint64{3, 4}, 2, 5, act)
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

	if err := runMachine(stream); nil != err {
		println(err.Error())
	}
}
