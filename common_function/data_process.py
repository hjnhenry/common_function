import numpy as np
import pandas as pd


def reduce_memory_usage(data, verbose=True, data_name='dataframe'):
    '''
    减少DataFrame所占内存:
    ------------------------------------------------------------
    入参结果如下:
        data: 数据集
            pd.DataFrame
        verbose: 是否打印信息
            bool,默认True
        data_name: 数据集的名字
            string,默认"dataframe",用于信息的打印
    ------------------------------------------------------------
    出参结果如下:
        data: 减少内存后的数据集
            pd.DataFrame
    '''
    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('%s 所占内存为: %.2f MB' % (data_name, start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('优化后所占内存为: %.2f MB' % (end_mem))
        print('减少了: %.1f%%' % (100 * (start_mem - end_mem) / start_mem))

    return data
