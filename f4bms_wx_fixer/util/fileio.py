import struct
import ctypes
import numpy as np

class BytesReader(object):
    def __init__(self, filename: str):
        self.filename = filename
        with open(self.filename, 'rb') as f:
            self.bytes = f.read()
        self.idx = 0

    def read_int8(self):
        return self.read('b')

    def read_uint8(self):
        return self.read('B')

    def read_int16(self):
        return self.read('h')

    def read_uint16(self):
        return self.read('H')

    def read_int32(self):
        return self.read('i')

    def read_uint32(self):
        return self.read('I')

    def read_int64(self):
        return self.read('q')

    def read_uint64(self):
        return self.read('Q')

    def read_f32(self):
        return self.read('f')

    def read_f64(self):
        return self.read('d')

    def read(self, fmt: str):
        x, = struct.unpack_from(fmt, self.bytes, self.idx)
        self.idx += struct.calcsize(fmt)
        return x

    def read_asciiz(self, num_bytes: int = 0):
        """
        Read an null-terminated ASCII string. If num_bytes
        is nonzero, advance the read pointer by that many bytes.
        """
        s = ctypes.create_string_buffer(self.bytes[self.idx:]).value
        if num_bytes == 0:
            self.idx += len(s)
        else:
            self.idx += num_bytes
        return s


class FileReader(object):
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.fd = open(self.filename, 'rb')
        return self

    def read_int8(self):
        return self.read('b')

    def read_uint8(self):
        return self.read('B')

    def read_int16(self):
        return self.read('h')

    def read_uint16(self):
        return self.read('H')

    def read_int32(self):
        return self.read('i')

    def read_uint32(self):
        return self.read('I')

    def read_int64(self):
        return self.read('q')

    def read_uint64(self):
        return self.read('Q')

    def read_f32(self):
        return self.read('f')

    def read_f64(self):
        return self.read('d')

    def read(self, fmt):
        num_bytes = struct.calcsize(fmt)
        b = self.fd.read(num_bytes)
        x, = struct.unpack_from(fmt, b)
        return x

    def read_np(self, dtype: np.dtype, num_elements: int):
        elemsize = np.dtype(dtype).itemsize
        b = self.fd.read(elemsize * num_elements)
        return np.frombuffer(b, dtype)

    def __exit__(self, type, value, traceback):
        self.fd.close()


class FileWriter(object):
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.fd = open(self.filename, 'wb')
        return self

    def write_int8(self, value):
        return self.write('b', value)

    def write_uint8(self, value):
        return self.write('B', value)

    def write_int16(self, value):
        return self.write('h', value)

    def write_uint16(self, value):
        return self.write('H', value)

    def write_int32(self, value):
        return self.write('i', value)

    def write_uint32(self, value):
        return self.write('I', value)

    def write_int64(self, value):
        return self.write('q', value)

    def write_uint64(self, value):
        return self.write('Q', value)

    def write_f32(self, value):
        return self.write('f', value)

    def write_f64(self, value):
        return self.write('d', value)

    def write(self, fmt, value):
        num_bytes = struct.calcsize(fmt)
        b = bytearray(num_bytes)
        struct.pack_into(fmt, b, 0, value)
        self.fd.write(b)

    # write a numpy array as binary to the file self.fd
    def write_np(self, arr):
        buf = arr.tobytes()
        self.fd.write(buf)

    def __exit__(self, type, value, traceback):
        self.fd.close()
