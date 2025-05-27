# This is an experimental feature.
# It is used to generate SME/SVE instructions by calling functions in the dynamic library.
# Only the add, sub, matmul, and transpose are supported.

import ctypes
import numpy
import os
import subprocess
import torch

CPU_INFO = subprocess.run(["lscpu"], stdout=subprocess.PIPE, text=True).stdout
HAVE_SME = CPU_INFO.find("sme") != -1
if HAVE_SME:
    TRITON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if (os.path.exists(os.path.join(TRITON_PATH, "_C/libsme_matrix.so"))):
        libsme_matrix = ctypes.CDLL(os.path.join(TRITON_PATH, "_C/libsme_matrix.so"))

# Note: SME does not have native add instruction for matrix type, we use SVE version.
def smeadd(x: torch.Tensor, y: torch.Tensor):
    assert HAVE_SME, "This device must support SME!"
    assert libsme_matrix, "libsme_matrix.so does not exist!"
    if (x.shape != y.shape):
        raise Exception("The shape of x must be equal to the shape of y!")
    if (x.dtype != y.dtype):
        raise Exception("The dtype of x must be equal to the dtype of y!")
    if (x.dtype == torch.float32):
        z = torch.zeros(x.shape, dtype=torch.float32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        libsme_matrix.add_float4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.float64):
        z = torch.zeros(x.shape, dtype=torch.float64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        libsme_matrix.add_float8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int8):
        z = torch.zeros(x.shape, dtype=torch.int8)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        libsme_matrix.add_int1(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int16):
        z = torch.zeros(x.shape, dtype=torch.int16)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        libsme_matrix.add_int2(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int32):
        z = torch.zeros(x.shape, dtype=torch.int32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        libsme_matrix.add_int4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int64):
        z = torch.zeros(x.shape, dtype=torch.int64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        libsme_matrix.add_int8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint8):
        z = torch.zeros(x.shape, dtype=torch.uint8)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        libsme_matrix.add_uint1(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif(x.dtype == torch.uint16):
        z = torch.zeros(x.shape, dtype=torch.uint16)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        libsme_matrix.add_uint2(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint32):
        z = torch.zeros(x.shape, dtype=torch.uint32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        libsme_matrix.add_uint4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint64):
        z = torch.zeros(x.shape, dtype=torch.uint64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        libsme_matrix.add_uint8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    else:
        raise Exception("The dtype must be fp32, fp64, int8, int16, int32, int64, uint8, uint16, uint32 or uint64!")
    z = torch.tensor(numpy.ctypeslib.as_array(ctypes_z, shape=x.shape))
    return z

# Note: SME does not have native FMOPA instruction for int/uint type, we use SVE version.
def smematmul(x: torch.Tensor, y: torch.Tensor):
    assert HAVE_SME, "This device must support SME!"
    assert libsme_matrix, "libsme_matrix.so does not exist!"
    if (x.shape[1] != y.shape[0]):
        raise Exception("The column of x must be equal to the row of y!")
    if (x.dtype != y.dtype):
        raise Exception("The dtype of x must be equal to the dtype of y!")
    if (x.dtype == torch.float32):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.float32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        libsme_matrix.matmul_float4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.float64):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.float64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        libsme_matrix.matmul_float8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.int8):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.int8)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        libsme_matrix.matmul_int1(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.int16):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.int16)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        libsme_matrix.matmul_int2(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.int32):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.int32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        libsme_matrix.matmul_int4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.int64):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.int64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        libsme_matrix.matmul_int8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.uint8):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.uint8)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        libsme_matrix.matmul_uint1(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.uint16):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.uint16)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        libsme_matrix.matmul_uint2(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.uint32):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.uint32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        libsme_matrix.matmul_uint4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    elif (x.dtype == torch.uint64):
        z = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.uint64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        libsme_matrix.matmul_uint8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1], y.shape[1])
    else:
        raise Exception("The dtype must be fp32, fp64, int8, int16, int32, int64, uint8, uint16, uint32 or uint64!")
    z = torch.tensor(numpy.ctypeslib.as_array(ctypes_z, shape=(x.shape[0], y.shape[1])))
    return z

# Note: SME does not have native sub instruction for matrix type, we use SVE version.
def smesub(x: torch.Tensor, y: torch.Tensor):
    assert HAVE_SME, "This device must support SME!"
    assert libsme_matrix, "libsme_matrix.so does not exist!"
    if (x.shape != y.shape):
        raise Exception("The shape of x must be equal to the shape of y!")
    if (x.dtype != y.dtype):
        raise Exception("The dtype of x must be equal to the dtype of y!")
    if (x.dtype == torch.float32):
        z = torch.zeros(x.shape, dtype=torch.float32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        libsme_matrix.sub_float4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.float64):
        z = torch.zeros(x.shape, dtype=torch.float64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        libsme_matrix.sub_float8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int8):
        z = torch.zeros(x.shape, dtype=torch.int8)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        libsme_matrix.sub_int1(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int16):
        z = torch.zeros(x.shape, dtype=torch.int16)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        libsme_matrix.sub_int2(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int32):
        z = torch.zeros(x.shape, dtype=torch.int32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        libsme_matrix.sub_int4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int64):
        z = torch.zeros(x.shape, dtype=torch.int64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        libsme_matrix.sub_int8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint8):
        z = torch.zeros(x.shape, dtype=torch.uint8)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)) 
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        libsme_matrix.sub_uint1(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint16):
        z = torch.zeros(x.shape, dtype=torch.uint16)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        libsme_matrix.sub_uint2(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint32):
        z = torch.zeros(x.shape, dtype=torch.uint32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        libsme_matrix.sub_uint4(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint64):
        z = torch.zeros(x.shape, dtype=torch.int64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        ctypes_y = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        libsme_matrix.sub_uint8(ctypes_z, ctypes_x, ctypes_y, x.shape[0], x.shape[1])
    else:
        raise Exception("The dtype must be fp32, fp64, int8, int16, int32, int64, uint8, uint16, uint32 or uint64!")
    z = torch.tensor(numpy.ctypeslib.as_array(ctypes_z, shape=x.shape))
    return z

def smetranspose(x: torch.Tensor):
    assert HAVE_SME, "This device must support SME!"
    assert libsme_matrix, "libsme_matrix.so does not exist!"
    if (x.dtype == torch.float32):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.float32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        libsme_matrix.transpose_float4(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.float64):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.float64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        libsme_matrix.transpose_float8(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int8):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.int8)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        libsme_matrix.transpose_int1(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int16):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.int16)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        libsme_matrix.transpose_int2(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int32):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.int32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        libsme_matrix.transpose_int4(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.int64):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.int64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        libsme_matrix.transpose_int8(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint8):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.uint8)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        libsme_matrix.transpose_int1(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint16):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.uint16)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        libsme_matrix.transpose_int2(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint32):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.uint32)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        libsme_matrix.transpose_int4(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    elif (x.dtype == torch.uint64):
        z = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.uint64)
        ctypes_z = z.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        ctypes_x = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        libsme_matrix.transpose_int8(ctypes_z, ctypes_x, x.shape[0], x.shape[1])
    else:
        raise Exception("The dtype must be fp32, fp64, int8, int16, int32, int64, uint8, uint16, uint32 or uint64!")
    z = torch.tensor(numpy.ctypeslib.as_array(ctypes_z, shape=(x.shape[1], x.shape[0])))
    return z

