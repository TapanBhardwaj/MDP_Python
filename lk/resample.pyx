from libcpp.list cimport list
from cpython cimport array
import array

cdef extern from "imResample.h":
    void _imResample( float *A, float *B, int ha, int hb, int wa, int wb, int d, float r );

# def list_test(array.array verts):
#     # cdef float[:] view = imResample(verts.data.as_floats)
#     imResample(verts.data.as_floats)

def imResample(float[::1] A, float[::1] B, ha, hb, wa, wb, d, r):
    _imResample(&A[0] , &B[0], ha, hb, wa, wb, d, r)