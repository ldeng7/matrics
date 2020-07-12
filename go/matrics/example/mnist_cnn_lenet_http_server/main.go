// an example without error handling

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"sync"

	"github.com/ldeng7/matrics/go/matrics"
	"github.com/ldeng7/matrics/go/matrics/neural"
)

const (
	wConvIn    uint32 = 28
	wConvCore  uint32 = 5
	dConv1Core uint32 = 32
	dConv2Core uint32 = 64
	strideConv uint8  = 1
	wPoolCore  uint8  = 2
	stridePool uint8  = 2
	nFcIn2     uint32 = 512
	nFcOut     uint32 = 10
)

type Ctx struct {
	machine *neural.Machine
	stream  matrics.Stream
	input   *matrics.Tensor
	output  *matrics.Tensor
}

func NewCtx(server *Server) *Ctx {
	c := &Ctx{}
	c.machine, _ = neural.NewMachine([][]uint32{{wConvIn, wConvIn, 1}})
	c.stream, _ = matrics.NewStream()

	ps := &neural.ParameterSource{bytes.NewReader(server.paras), binary.LittleEndian}
	conv2dConf := &matrics.Conv2dConf{
		matrics.ActivationReLU,
		matrics.Cnn2dConf{matrics.CnnPaddingTypeSame, strideConv, strideConv},
	}
	pool2dConf := &matrics.Pool2dConf{
		matrics.PoolTypeMax, wPoolCore, wPoolCore,
		matrics.Cnn2dConf{matrics.CnnPaddingTypeSame, stridePool, stridePool},
	}
	c1, _ := neural.NewConv2dLayer("c1", ps, [4]uint32{wConvIn, wConvIn, 1, 1}, [3]uint32{wConvCore, wConvCore, dConv1Core}, conv2dConf)
	c1.Connect(c.machine.InputLayer(), 0, 0)
	p1, _ := c1.NewSubsequentPool2dLayerAndConnect("p1", pool2dConf)
	c2, _ := p1.NewSubsequentConv2dLayerAndConnect("c2", ps, [3]uint32{wConvCore, wConvCore, dConv2Core}, conv2dConf)
	p2, _ := c2.NewSubsequentPool2dLayerAndConnect("p2", pool2dConf)
	f, fcc, _ := p2.NewSubsequentFullConnChainLayerAndConnect("fcc", ps, []uint32{nFcIn2}, nFcOut, &matrics.ActivationReLU)
	c.machine.AddLayer(c1, p1, c2, p2, f, fcc)
	c.input, c.output = c.machine.Inputs()[0], fcc.Outputs()[0]

	return c
}

func (c *Ctx) Destroy() {
	c.stream.Destroy()
	c.machine.Destroy()
}

type Server struct {
	en    bool
	pool  *sync.Pool
	paras []byte
}

func NewServer() *Server {
	matrics.Init()
	s := &Server{
		en:   true,
		pool: &sync.Pool{},
	}
	s.paras, _ = ioutil.ReadFile("para.bin")
	s.pool.New = func() interface{} {
		if s.en {
			return NewCtx(s)
		}
		return nil
	}
	return s
}

func (s *Server) Destroy() {
	s.en = false
	for {
		c := s.pool.Get()
		if nil == c {
			break
		}
		ctx, _ := c.(*Ctx)
		ctx.Destroy()
	}
	matrics.Uninit()
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx, _ := s.pool.Get().(*Ctx)
	defer s.pool.Put(ctx)
	json.Unmarshal([]byte(r.URL.Query().Get("m")), &ctx.input.Data)
	ctx.machine.Run(ctx.stream)
	str, _ := json.Marshal(ctx.output.Data)
	w.Write(str)
}

func main() {
	server := NewServer()
	defer server.Destroy()
	http.ListenAndServe(":8080", server)
}
