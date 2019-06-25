import numpy as np
import pandas as pd


def dataframe_compare_all(data1, data2, primary_key, sort_key=None, precision=0, verbose=0,
                          num_fill=-9, char_fill='Blank'):
    '''
    比较两个数据集共有索引、变量的值是否相同:
    ------------------------------------------------------------
    入参结果如下:
        data1、data2: 要对比的两个数据集
            pd.DataFrame
        primary_key: 主键变量名,根据该变量取交集
            string
        sort_key: 排序键
            string or list of string,默认None
        precision: 数值型差值精度
            int,默认0,精度0很容易将两个实际相同的值判断为不等
        verbose: 打印信息
            int,默认0,打印每个变量的信息,当!=0时只打印不相等变量的信息
    ------------------------------------------------------------
    出参结果如下:
        compare_all_df: 每个变量的一致率
            pd.DataFrame
    '''
    if sort_key is None:
        sort_key = primary_key
    column = list(set(data1.columns) & set(data2.columns))
    column.remove(primary_key)
    index = list(set(data1[primary_key]) & set(data2[primary_key]))

    print('第一个数据集的维度:', data1.shape)
    print('第二个数据集的维度:', data2.shape)
    print('两个数据集公共变量数:', len(column))
    print('两个数据集相同 %s 样本数:' % (primary_key), len(index))

    data1_new = data1.loc[data1[primary_key].isin(index), :].drop_duplicates().sort_values(sort_key).reset_index()
    data2_new = data2.loc[data2[primary_key].isin(index), :].drop_duplicates().sort_values(sort_key).reset_index()

    compare_all_df = pd.DataFrame(columns=['var', 'consistent_rate'])

    if len(data1_new) != len(data2_new):
        index_count_compare = (data1_new[primary_key].value_counts().sort_index() != data2_new[
            primary_key].value_counts().sort_index())
        print('两个数据集按照共同的 %s 取新数据集,去重后新数据集样本数不同,不同的 %s 有 %s'
              % (primary_key, primary_key, list(index_count_compare[index_count_compare is True].index)))
    else:
        sample_number = len(data1_new)
        consist = 1
        for var in column:
            if data1_new[var].dtypes != 'object':
                same_number = sum(abs(data1_new[var].fillna(num_fill) - data2_new[var].fillna(num_fill)) <= precision)
            else:
                same_number = sum(abs(data1_new[var].fillna(char_fill) == data2_new[var].fillna(char_fill)))

            consist_rate = same_number / sample_number

            if verbose == 0:
                print('变量 %s 一致率：%s' % (var, consist_rate))
                if consist_rate < 1:
                    consist = 0
            else:
                if consist_rate < 1:
                    print('变量 %s 一致率：%s' % (var, consist_rate))
                    consist = 0

            compare_all_df = compare_all_df.append({'var': var, 'consistent_rate': consist_rate}, ignore_index=True)

        if consist == 1:
            print('所有变量均一致')

    return compare_all_df


def dataframe_compare_sigle(data1, data2, primary_key, var, sort_key=None, precision=0, num_fill=-9, char_fill='Blank'):
    '''
    查看两个数据集选定变量不等样本的值比较:
    ------------------------------------------------------------
    入参结果如下:
        data1、data2: 要对比的两个数据集
            pd.DataFrame
        primary_key: 主键变量名,根据该变量取交集
            string
        var: 要查看的变量
            string
        sort_key: 排序键
            string or list of string,默认None
        precision: 数值型差值精度
            int,默认0,精度0很容易将两个实际相同的值判断为不等
    ------------------------------------------------------------
    出参结果如下:
        compare_sigle_df: 选定变量
            pd.DataFrame
    '''
    if sort_key is None:
        sort_key = primary_key
    index = list(set(data1[primary_key]) & set(data2[primary_key]))

    if isinstance(sort_key, list):
        var_all = list(set([primary_key]) | set(sort_key) | set([var]))
    else:
        var_all = list(set([primary_key]) | set([sort_key]) | set([var]))

    data1_new = data1.loc[data1[primary_key].isin(index), var_all].drop_duplicates().sort_values(sort_key).reset_index()
    data2_new = data2.loc[data2[primary_key].isin(index), var_all].drop_duplicates().sort_values(sort_key).reset_index()

    if data1_new[var].dtypes != 'object':
        compare_sigle_df = pd.merge(
            data1_new.loc[(abs(data1_new[var].fillna(num_fill) - data2_new[var].fillna(num_fill)) <= precision
                           ) is False, [primary_key, var]],
            data2_new.loc[(abs(data1_new[var].fillna(num_fill) - data2_new[var].fillna(num_fill)) <= precision
                           ) is False, [primary_key, var]],
            on=primary_key)
    else:
        compare_sigle_df = pd.merge(
            data1_new.loc[(abs(data1_new[var].fillna(char_fill) == data2_new[var].fillna(char_fill))) is False,
                          [primary_key, var]],
            data2_new.loc[(abs(data1_new[var].fillna(char_fill) == data2_new[var].fillna(char_fill))) is False,
                          [primary_key, var]],
            on=primary_key)

    return compare_sigle_df
