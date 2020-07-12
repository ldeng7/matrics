package matrics

import (
	. "github.com/smartystreets/goconvey/convey"
)

func (c *testCtx) testTensorCubConv2d() {
	Convey("little case", func() {
		var wIn, hIn, dIn, dOut, wCore, hCore uint32 = 4, 3, 2, 3, 3, 2
		x, _ := NewTensor([]uint32{wIn, hIn, dIn})
		for i := uint32(0); i < wIn*hIn*dIn; i++ {
			x.Data[i] = 1.0 + float32(i)
		}
		y, _ := NewTensor([]uint32{wCore, hCore, dIn, dOut})
		for i := uint32(0); i < wCore*hCore*dIn*dOut; i++ {
			y.Data[i] = 1.0 + float32(i)
		}
		z, _ := NewTensor([]uint32{dOut})
		z.Data[0] = -500.0

		conf := &Conv2dConf{ActivationReLU, Cnn2dConf{}}
		strides := []uint8{1, 2, 3}
		paddings := []CnnPaddingType{CnnPaddingTypeSame, CnnPaddingTypeValid}
		expected := [][]float32{
			{
				202., 540., 618., 222., 426., 852., 930., 414., 0., 92., 122., 0.,
				1614., 2480., 2702., 1826., 2222., 3368., 3590., 2402., 1158., 1744., 1846., 1228.,
				2526., 3920., 4286., 2930., 3518., 5384., 5750., 3890., 1902., 2896., 3070., 2068.,
			}, {
				540., 618., 852., 930., 2480., 2702., 3368., 3590., 3920., 4286., 5384., 5750.,
			}, {
				540., 222., 92., 0., 2480., 1826., 1744., 1228., 3920., 2930., 2896., 2068.,
			}, {
				540., 2480., 3920.,
			}, {
				202., 222., 1614., 1826., 2526., 2930.,
			}, {
				540., 2480., 3920.,
			},
		}

		i := 0
		for _, stride := range strides {
			conf.Cnn.StrideX, conf.Cnn.StrideY = stride, stride
			for _, padding := range paddings {
				conf.Cnn.PaddingType = padding
				wOut, hOut := conf.Cnn.OutputSize(wIn, hIn, wCore, hCore)
				w, _ := NewTensor([]uint32{wOut, hOut, dOut})
				So(x.CubBatchConv2d(y, z, w, conf, c.stream), ShouldBeNil)
				So(w.Data, testShouldEqualsDataSlice, expected[i])
				i++
			}
		}
	})
	Convey("great case", func() {
		var wIn, hIn, dIn, dOut, wCore uint32 = 128, 128, 1, 2, 2
		x, _ := NewTensor([]uint32{wIn, hIn, dIn})
		defer x.Destroy()
		for i := uint32(0); i < wIn*hIn*dIn; i++ {
			x.Data[i] = 1.0 + float32(i)/1000
		}
		y, _ := NewTensor([]uint32{wCore, wCore, dIn, dOut})
		defer y.Destroy()
		for i := uint32(0); i < wCore*wCore*dIn*dOut; i++ {
			y.Data[i] = 1.0 + float32(i)/10
		}
		z, _ := NewTensor([]uint32{dOut})
		defer z.Destroy()

		conf := &Conv2dConf{ActivationNone, Cnn2dConf{CnnPaddingTypeSame, 1, 1}}
		wOut, hOut := conf.Cnn.OutputSize(wIn, hIn, wCore, wCore)
		w, _ := NewTensor([]uint32{wOut, hOut, dOut})
		defer w.Destroy()
		expected := []float32{4.9224, 4.9270, 2.6330, 5.5112, 2.9146, 36.2387, 36.5033, 17.3830,
			6.6256, 3.5858, 50.0439, 50.4093, 24.3362}
		So(x.CubBatchConv2d(y, z, w, conf, c.stream), ShouldBeNil)
		data := []float32{w.Data[0], w.Data[1], w.Data[127], w.Data[128], w.Data[255], w.Data[16256], w.Data[16382], w.Data[16383],
			w.Data[16384], w.Data[16511], w.Data[32640], w.Data[32766], w.Data[32767]}
		So(data, testShouldEqualsDataSlice, expected)
	})
}

func (c *testCtx) testNeural() {
	Convey("Tensor.CubConv2d", c.testTensorCubConv2d)
}
