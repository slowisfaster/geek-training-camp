# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
#from libcpp.utility cimport pair

def hello():
    print("hello")


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


cpdef target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

    target_mean_v3_impl(result, y, x, nrow)
    return result

cdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1

    i=0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)

cpdef target_mean_v4(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[long] y = np.asfortranarray(data[y_name])
    cdef np.ndarray[long] x = np.asfortranarray(data[x_name])

    target_mean_v4_impl(result, y, x, nrow)
    return result

cdef void target_mean_v4_impl(double[:] result, long[:] y, long[:] x, const long nrow):
    cdef unordered_map[long, long] value_map
    cdef unordered_map[long, long] count_map

    cdef long i
    #for i in range(nrow):
    for i from 0<=i<nrow by 1:
        if value_map.count(i):
            value_map[x[i]] += y[i]
            count_map[x[i]] += 1
        else:
            value_map[x[i]] = y[i]
            count_map[x[i]] = 1

    #for i in range(nrow):
    for i from 0 <= i < nrow by 1:
        result[i] = (value_map[x[i]] - y[i])/(count_map[x[i]]-1)

cpdef target_mean_v5(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[long] y = np.asfortranarray(data[y_name])
    cdef np.ndarray[long] x = np.asfortranarray(data[x_name])

    target_mean_v5_impl(result, y, x, nrow)
    return result

cdef void target_mean_v5_impl(double[:] result, long[:] y, long[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    #for i in range(nrow):
    for i from 0<=i<nrow by 1:
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1

    #for i in range(nrow):
    for i from 0<=i<nrow by 1:
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)