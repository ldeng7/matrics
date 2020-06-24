package main

import (
	"math"
	"math/rand"
	"time"

	"github.com/ldeng7/matrics/go/matrics"
)

const (
	N        = 1 << 20
	W, H, W1 = 1024, 768, 896
)

func runMatrixMulVector(stream matrics.Stream) error {
	mx, err := matrics.NewMatrix(W, H)
	if nil != err {
		return err
	}
	defer mx.Destroy()
	vy, err := matrics.NewVector(W)
	if nil != err {
		return err
	}
	defer vy.Destroy()
	vz, err := matrics.NewVector(H)
	if nil != err {
		return err
	}
	defer vz.Destroy()
	buf, err := mx.NewMatrixWideBuffer()
	if nil != err {
		return err
	}
	defer buf.Destroy()

	rand.Seed(time.Now().Unix())
	for i := 0; i < W*H; i++ {
		mx.Buf[i] = rand.Float32()
	}
	for i := 0; i < W; i++ {
		vy.Buf[i] = rand.Float32()
	}

	t := time.Now()
	for i := 0; i < 30; i++ {
		mx.MulVector(vy, vz, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(vz.Buf[0], vz.Buf[1], vz.Buf[H-1])

	for i := 0; i < H; i++ {
		vz.Buf[i] = 0
	}
	t = time.Now()
	for i := 0; i < 30; i++ {
		if err := mx.WideMulVector(vy, vz, buf, stream); nil != err {
			return err
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(vz.Buf[0], vz.Buf[1], vz.Buf[H-1])
	return nil
}

func runVectorTMulMatrix(stream matrics.Stream) error {
	vx, err := matrics.NewVector(W)
	if nil != err {
		return err
	}
	defer vx.Destroy()
	my, err := matrics.NewMatrix(W1, W)
	if nil != err {
		return err
	}
	defer my.Destroy()
	vz, err := matrics.NewVector(W1)
	if nil != err {
		return err
	}
	defer vz.Destroy()

	rand.Seed(time.Now().Unix())
	x, y, z := make([]float32, W), make([]float32, W1*W), make([]float32, W1)
	for i := 0; i < W; i++ {
		x[i] = rand.Float32()
	}
	for i := 0; i < W1*W; i++ {
		y[i] = rand.Float32()
	}
	copy(vx.Buf, x)
	copy(my.Buf, y)

	t := time.Now()
	for i := 0; i < 3; i++ {
		for k := 0; k < W1; k++ {
			var s float32
			for h := 0; h < W; h++ {
				s += x[h] * y[W1*h+k]
			}
			z[k] = s
		}
	}
	println(time.Now().Sub(t).Microseconds())
	println(z[0], z[1], z[W1-1])

	t = time.Now()
	for i := 0; i < 3; i++ {
		vx.TMulMatrix(my, vz, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(vz.Buf[0], vz.Buf[1], vz.Buf[W1-1])
	return nil
}

func runVectorPowScalar(stream matrics.Stream) error {
	vx, err := matrics.NewVector(N)
	if nil != err {
		return err
	}
	defer vx.Destroy()
	vy, err := matrics.NewVector(N)
	if nil != err {
		return err
	}
	defer vy.Destroy()

	rand.Seed(time.Now().Unix())
	x, y := make([]float32, N), make([]float32, N)
	for i := 0; i < N; i++ {
		x[i] = rand.Float32()
	}
	copy(vx.Buf, x)

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
	println(vy.Buf[0], vy.Buf[1], vy.Buf[N-1])
	return nil
}

func runVectorAddVector(stream matrics.Stream) error {
	vx, err := matrics.NewVector(N)
	if nil != err {
		return err
	}
	defer vx.Destroy()
	vy, err := matrics.NewVector(N)
	if nil != err {
		return err
	}
	defer vy.Destroy()

	rand.Seed(time.Now().Unix())
	x, y := make([]float32, N), make([]float32, N)
	for i := 0; i < N; i++ {
		x[i], y[i] = rand.Float32(), rand.Float32()
	}
	copy(vx.Buf, x)
	copy(vy.Buf, y)

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
		vx.AddVector(vy, vx, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(vx.Buf[0], vx.Buf[1], vx.Buf[N-1])
	return nil
}

func runVectorSum(stream matrics.Stream) error {
	vx, err := matrics.NewVector(N)
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
	copy(vx.Buf, x)

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

func runVectorDot(stream matrics.Stream) error {
	vx, err := matrics.NewVector(N)
	if nil != err {
		return err
	}
	defer vx.Destroy()
	vy, err := matrics.NewVector(N)
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
	copy(vx.Buf, x)
	copy(vy.Buf, y)

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

func runMatrixMulMatrix(stream matrics.Stream) error {
	mx, err := matrics.NewMatrix(W, H)
	if nil != err {
		return err
	}
	defer mx.Destroy()
	my, err := matrics.NewMatrix(W1, W)
	if nil != err {
		return err
	}
	defer my.Destroy()
	mz, err := matrics.NewMatrix(W1, H)
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
	copy(mx.Buf, x)
	copy(my.Buf, y)

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
		mx.MulMatrix(my, mz, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(mz.Buf[0], mz.Buf[1], mz.Buf[W1*H-2], mz.Buf[W1*H-1])
	return nil
}

func runMatrixT(stream matrics.Stream) error {
	mx, err := matrics.NewMatrix(W, H)
	if nil != err {
		return err
	}
	defer mx.Destroy()
	my, err := matrics.NewMatrix(H, W)
	if nil != err {
		return err
	}
	defer my.Destroy()

	rand.Seed(time.Now().Unix())
	x, y := make([]float32, W*H), make([]float32, W*H)
	for i := 0; i < W*H; i++ {
		x[i] = rand.Float32()
	}
	copy(mx.Buf, x)

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
		mx.T(my, stream)
	}
	println(time.Now().Sub(t).Microseconds())
	println(my.Buf[0], my.Buf[1], my.Buf[W], my.Buf[W*H-2], my.Buf[W*H-1])
	return nil
}

func main() {
	if err := matrics.LoadLib(); nil != err {
		println(err.Error())
		return
	}
	defer matrics.ReleaseLib()
	stream, err := matrics.NewStream()
	if nil != err {
		println(err.Error())
		return
	}
	defer stream.Destroy()
	if err := runMatrixMulVector(stream); nil != err {
		println(err.Error())
	}
}
