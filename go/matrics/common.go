package matrics

import (
	"fmt"
)

const mtCodeSuccess = 0

type mtError struct {
	code int
	msg  string
}

func (e *mtError) Error() string {
	return fmt.Sprintf("%s(cuda code: %d)", e.msg, e.code)
}

func newMtErr(code int, msg string) error {
	if mtCodeSuccess != code {
		return &mtError{code, msg}
	}
	return nil
}

type Stream uintptr

func NewStream() (Stream, error) {
	var stream Stream
	var code int
	mtNewStream(&stream, &code)
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return stream, nil
}

type Buffer struct {
	handle bufferHandle
	ref    uint32
}
