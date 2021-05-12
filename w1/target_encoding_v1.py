# coding = 'utf-8'
import numpy as np
import pandas as pd
from time import time
import tm

# def recordtime(func, *args, **kwargs):
#     def wrapper(func, *args, **kwargs):
#         start = time()
#         func(*args, **kwargs)
#         end = time()
#         print(f"elapsed time {end-start}")
#     return wrapper

#@recordtime
#@profile
def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result

#@recordtime
#@profile
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



def main():
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    start = time()
    result_1 = target_mean_v1(data, 'y', 'x')
    end = time()
    print("v1: {}".format(end-start))
    result_2 = target_mean_v2(data, 'y', 'x')
    end2 = time()
    print("v2: {}".format(end2-end))
    result_3 = tm.target_mean_v3(data, 'y', 'x')
    end3 = time()
    print("v3: {}".format(end3 - end2))
    result_5 = tm.target_mean_v5(data, 'y', 'x')
    end4 = time()
    print("v5: {}".format(end4 - end3))
    result_4 = tm.target_mean_v4(data, 'y', 'x')
    end5 = time()
    print("v4: {}".format(end5 - end4))
    #diff = np.linalg.norm(result_1 - result_2)
    #print(diff)


if __name__ == '__main__':
    main()
