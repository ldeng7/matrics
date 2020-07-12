package matrics

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func testShouldEqualsDataSlice(actual interface{}, expected ...interface{}) string {
	a, _ := actual.([]float32)
	e, _ := expected[0].([]float32)
	if len(a) != len(e) {
		return "unequal length"
	}
	var t float32 = 0.001
	if len(expected) >= 2 {
		t, _ = expected[1].(float32)
	}
	for i, f := range e {
		d := (f - a[i]) / f
		if d < 0 {
			d = -d
		}
		if d >= t {
			return fmt.Sprintf("the error ratio at index %d exceeded the threshold %f: actual %f, expected %f",
				i, t, a[i], f)
		}
	}
	return ""
}

type testCtx struct {
	stream Stream
}

func TestMatrics(t *testing.T) {
	err := Init()
	defer Uninit()
	stream, _ := NewStream()
	ctx := &testCtx{stream}
	Convey("init", t, func() {
		So(err, ShouldBeNil)
		So(stream, ShouldNotBeNil)
	})
	Convey("neural", t, ctx.testNeural)
}
