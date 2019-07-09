import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, chi2
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")


def chi_square_bin(col, labels, value_range='default'):
    '''
    卡方分箱,chi_square_bin方法出入参如下:
    ------------------------------------------------------------
    入参结果如下:
        col: 要分箱的变量,格式为pandas.Series,可对dataframe的数据集用变量名直接索引得到
        labels: 目标变量的数据,格式为pandas.Series
        value_range: 要分箱变量的取值范围处理
            str:"default",默认,不对要分箱变量的取值范围进行处理
                "max_min",将要分箱变量的可取值如果超过1000个,对改变了的范围进行最大最小标准化,并乘以1000后取整,
                    将可取值映射至0-1000的整型,限制在1000个以内,减少遍历所有可取值合并分组的计算复杂度
    ------------------------------------------------------------
    入参结果如下:
        cut_bin: 卡方分箱的分箱阈值列表
    '''
    def chi_square_value(arr):
        '''计算二维矩阵的卡方值'''
        row_sum = arr.sum(axis=1)
        col_sum = arr.sum(axis=0)
        total_sum = arr.sum()
        # 计算各位置的期望频数 C_i * R_j / N
        expect = ((np.ones(arr.shape) * col_sum / total_sum).T * row_sum).T
        square = (arr - expect) ** 2 / expect
        # 期望频数为0时，做除数没有意义，不计入卡方值期望
        square[expect == 0] = 0
        # 计算卡方值
        chi_square = square.sum()

        return chi_square

    col_max = col.max()
    col_min = col.min()
    if value_range == 'default':
        freq_tab = pd.crosstab(col, labels)
    elif value_range == 'max_min':
        if len(col.unique()) <= 1000:
            freq_tab = pd.crosstab(col, labels)
        else:
            freq_tab = pd.crosstab(col.apply(lambda x: int((x - col_min) * 1000 / (col_max - col_min))
                                             if pd.notnull(x) else x), labels)
    else:
        raise ValueError('请传入正确的value_range参数,value_range应为"default"或"max_min"')

    # 初始分组切分点，每个变量值都是切分点。每组中只包含一个变量值.分组区间是左闭右开的
    cut_bin = freq_tab.index.values
    # 转成numpy数组用于计算
    freq_tab = freq_tab.values
    # 以95%的置信度（自由度为类数目-1）设定阈值。
    threshold = chi2.isf(0.05, df=freq_tab.shape[-1] - 1)

    while True:
        min_value = None
        min_index = None
        # 变量被归为一档时，结束运行，否则后面代码报错，归为一档的变量计算的IV值为0
        if len(freq_tab) == 1:
            break
        # 依次取两组计算卡方值，并判断是否小于当前最小的卡方
        for index in range(len(freq_tab) - 1):
            chi_square = chi_square_value(freq_tab[index:index + 2])
            if (min_value is None) or (min_value > chi_square):
                min_value = chi_square
                min_index = index
        # 如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
        if min_value <= threshold:
            freq_tab[min_index] = freq_tab[min_index] + freq_tab[min_index + 1]
            freq_tab = np.delete(freq_tab, min_index + 1, axis=0)
            cut_bin = np.delete(cut_bin, min_index + 1, axis=0)
        # 如果最小卡方值大于阈值，并且组数大于10组，则合并最小卡方值的相邻两组，并继续循环
        elif (min_value > threshold) and len(cut_bin) >= 10:
            freq_tab[min_index] = freq_tab[min_index] + freq_tab[min_index + 1]
            freq_tab = np.delete(freq_tab, min_index + 1, axis=0)
            cut_bin = np.delete(cut_bin, min_index + 1, axis=0)
        else:
            break
    if (value_range == 'max_min') and (len(col.unique()) > 1000):
        cut_bin = cut_bin * (col_max - col_min) / 1000 + col_min

    cut_bin = [-np.inf] + sorted(list(cut_bin)) + [np.inf]

    return cut_bin


def best_ks_bin(col, labels):
    '''
    BEST-KS分箱,best_ks_bin方法出入参如下:
    ------------------------------------------------------------
    入参结果如下:
        col: 要分箱的变量,格式为pandas.Series,可对dataframe的数据集用变量名直接索引得到
        labels: 目标变量的数据,格式为pandas.Series
    ------------------------------------------------------------
    入参结果如下:
        cut_bin: 卡方分箱的分箱阈值列表
    '''
    # 最小的箱的占比阈值（样本数的0.05）
    limit = len(labels) / 20

    # 获取最大KS的变量值
    def get_max_ks_value(col_sample, labels_sample):
        # best-ks的第一个停止分箱条件：该箱对应类别全部为0或者1
        if len(set(labels)) == 1:
            return None

        cumsum_tab = (pd.crosstab(col_sample, labels_sample) / pd.Series(labels_sample).value_counts()).\
            sort_index().cumsum()
        ks_list = abs(cumsum_tab[1] - cumsum_tab[0])
        # best-ks的第二个停止分箱条件：最小的箱的占比低于设定的阈值（常用0.05）
        # 但直接按照最优best分箱很容易达到停止分箱条件,于是将ks值排序,ks可选择次优值
        ks_list_index = ks_list.nlargest(len(ks_list)).index.tolist()
        for value in ks_list_index:
            if len(col_sample[col_sample < value]) >= limit and len(col_sample[col_sample >= value]) >= limit:
                return value
        else:
            return None

    # cut_bin为分箱阈值
    cut_bin = []
    # travers_value保存了遍历过的数据范围
    travers_value = []
    while True:
        # early 为停止迭代的条件，如果没有新的阈值加入到cut_bin，early则保持为1
        early = 1
        # cut_bin为空的时候的遍历情况
        if len(cut_bin) == 0:
            max_ks_value = get_max_ks_value(col, labels)
            if max_ks_value is not None:
                cut_bin.append(max_ks_value)
                early = 0
        # cut_bin只有一个值的时候的遍历情况
        elif len(cut_bin) == 1:
            # 将遍历的值范围加入到travers_value，该值范围不再遍历，如果此次遍历max_ks_value不为空，会有一个新的值范围，否则也不需在遍历
            travers_value.append('(-inf,%s)' % cut_bin[0])
            max_ks_value = get_max_ks_value(col[col < cut_bin[0]], labels[col < cut_bin[0]])
            if max_ks_value is not None:
                cut_bin.append(max_ks_value)
                early = 0

            travers_value.append('[%s,inf)' % cut_bin[0])
            max_ks_value = get_max_ks_value(col[col >= cut_bin[0]], labels[col >= cut_bin[0]])
            if max_ks_value is not None:
                cut_bin.append(max_ks_value)
                early = 0
        # cut_bin有两个值及以上的时候的遍历情况
        else:
            cut_bin_tmp = cut_bin.copy()
            for index, value in enumerate(cut_bin_tmp):
                # cut_bin第一个值的时候的遍历情况
                if index == 0:
                    if '(-inf,%s)' % value not in travers_value:
                        travers_value.append('(-inf,%s)' % value)
                        max_ks_value = get_max_ks_value(col[col < value], labels[col < value])
                        if max_ks_value is not None:
                            cut_bin.append(max_ks_value)
                            early = 0
                # cut_bin其他值的变量情况
                else:
                    if '[%s,%s)' % (cut_bin_tmp[index - 1], value) not in travers_value:
                        travers_value.append('[%s,%s)' % (cut_bin_tmp[index - 1], value))
                        max_ks_value = get_max_ks_value(col[(col >= cut_bin_tmp[index - 1]) & (col < value)],
                                                        labels[(col >= cut_bin_tmp[index - 1]) & (col < value)])
                        if max_ks_value is not None:
                            cut_bin.append(max_ks_value)
                            early = 0
                    # cut_bin最后一个值的遍历情况
                    if index == len(cut_bin_tmp) - 1:
                        if '[%s,inf)' % value not in travers_value:
                            travers_value.append('[%s,inf)' % value)
                            max_ks_value = get_max_ks_value(col[col >= value], labels[col >= value])
                            if max_ks_value is not None:
                                cut_bin.append(max_ks_value)
                                early = 0
        cut_bin.sort()
        if early == 1:
            break

    cut_bin = [-np.inf] + cut_bin + [np.inf]
    return cut_bin


def get_psi_dataframe(feature, keep_cols=None, cut_bin_dict=None):
    '''
    计算各变量的psi值,get_psi_dataframe、方法出入参如下:
    ------------------------------------------------------------
    入参结果如下:
        feature: 数据集的特征空间
        keep_cols: 需计算max_psi的变量列表
        cut_bin_dict: 数值型变量要进行分箱的阈值字典,格式为{'col1':[value1,value2,...], 'col2':[value1,value2,...], ...}
    ------------------------------------------------------------
    入参结果如下:
        max_psi_series: 各变量的psi值
    '''
    def psi_count(data_expect, data_real):
        '''计算psi值'''
        value_list = set(data_expect.unique()) | set(data_real.unique())
        psi = 0
        len_expect = len(data_expect)
        len_real = len(data_real)
        for value in value_list:
            # 判断是否某类是否为0，避免出现无穷小值和无穷大值
            if sum(data_expect == value) == 0:
                expect_rate = 1 / len_expect
            else:
                expect_rate = sum(data_expect == value) / len_expect
            if sum(data_real == value) == 0:
                real_rate = 1 / len_real
            else:
                real_rate = sum(data_real == value) / len_real
            psi += (real_rate - expect_rate) * np.log(real_rate / expect_rate)
        return psi

    # 取期望月份列表和实际月份列表
    feature['DATE'] = feature['DATE'].apply(lambda x: x[:7])
    date_expect = sorted(feature['DATE'].unique())
    if len(date_expect) < 2:
        raise ValueError('"DATE"变量的月份跨度需≥2个月,否则无法计算psi值')
    # 如果第一个月和最后一个月的数据量太少,直接剔除,可能会验证影响psi值的计算
    if len(feature.loc[feature['DATE'] == date_expect[0]]) < (len(feature.loc[feature['DATE'] == date_expect[1]]) / 5):
        date_expect = date_expect[1:]
    if len(feature.loc[feature['DATE'] == date_expect[-1]]) < (
        len(feature.loc[feature['DATE'] == date_expect[-2]]) / 5):
        date_expect = date_expect[:-1]
    date_real = date_expect[1:]
    date_expect = date_expect[:-1]

    col_types = feature[keep_cols].dtypes
    categorical_feature = list(col_types[col_types == 'object'].index)
    numerical_feature = list(col_types[col_types != 'object'].index)

    psi_dataframe = pd.DataFrame(index=date_real, columns=keep_cols)

    # 遍历数值变量计算psi值
    for col in numerical_feature:
        cut_bin = cut_bin_dict[col]
        col_tmp = feature[[col, 'DATE']].copy()
        # 按照分箱阈值分箱,并将缺失值替换成Blank
        col_tmp[col] = pd.cut(col_tmp[col], cut_bin, right=False).cat.add_categories(['Blank']).fillna('Blank')
        # 计算各个时间段的psi值
        for expect, real in zip(date_expect, date_real):
            data_expect = col_tmp.loc[col_tmp['DATE'] == expect, col]
            data_real = col_tmp.loc[col_tmp['DATE'] == real, col]
            psi_dataframe.loc[real, col] = psi_count(data_expect, data_real)
    # 遍历类别变量计算psi值
    for col in categorical_feature:
        # 时间变量可能包含在categorical_feature中,如有时间变量需另外处理
        if col != 'DATE':
            col_tmp = feature[[col, 'DATE']].copy()
        else:
            col_tmp = feature[['DATE']].copy()
        # 缺失值替换成Blank
        col_tmp[col] = col_tmp[col].fillna('Blank')
        # 计算各个时间段的psi值
        for expect, real in zip(date_expect, date_real):
            data_expect = col_tmp.loc[col_tmp['DATE'] == expect, col]
            data_real = col_tmp.loc[col_tmp['DATE'] == real, col]
            psi_dataframe.loc[real, col] = psi_count(data_expect, data_real)

    return psi_dataframe


def get_iv_series(feature, labels, keep_cols=None, cut_bin_dict=None):
    '''
    计算各变量最大的iv值,get_iv_series方法出入参如下:
    ------------------------------------------------------------
    入参结果如下:
        feature: 数据集的特征空间
        labels: 数据集的输出空间
        keep_cols: 需计算iv值的变量列表
        cut_bin_dict: 数值型变量要进行分箱的阈值字典,格式为{'col1':[value1,value2,...], 'col2':[value1,value2,...], ...}
    ------------------------------------------------------------
    入参结果如下:
        iv_series: 各变量最大的psi值
    '''
    def iv_count(data_bad, data_good):
        '''计算iv值,计算逻辑和计算psi值相同'''
        value_list = set(data_bad.unique()) | set(data_good.unique())
        iv = 0
        len_bad = len(data_bad)
        len_good = len(data_good)
        for value in value_list:
            # 判断是否某类是否为0，避免出现无穷小值和无穷大值
            if sum(data_bad == value) == 0:
                bad_rate = 1 / len_bad
            else:
                bad_rate = sum(data_bad == value) / len_bad
            if sum(data_good == value) == 0:
                good_rate = 1 / len_good
            else:
                good_rate = sum(data_good == value) / len_good
            iv += (good_rate - bad_rate) * np.log(good_rate / bad_rate)
        return iv

    if keep_cols is None:
        keep_cols = sorted(list(feature.columns))
    col_types = feature[keep_cols].dtypes
    categorical_feature = list(col_types[col_types == 'object'].index)
    numerical_feature = list(col_types[col_types != 'object'].index)

    iv_series = pd.Series()

    # 遍历数值变量计算iv值
    for col in numerical_feature:
        cut_bin = cut_bin_dict[col]
        # 按照分箱阈值分箱,并将缺失值替换成Blank,区分好坏样本
        data_bad = pd.cut(feature[col], cut_bin, right=False).cat.add_categories(['Blank']).fillna('Blank')[labels == 1]
        data_good = pd.cut(feature[col], cut_bin, right=False
                           ).cat.add_categories(['Blank']).fillna('Blank')[labels == 0]
        iv_series[col] = iv_count(data_bad, data_good)
    # 遍历类别变量计算iv值
    for col in categorical_feature:
        # 将缺失值替换成Blank,区分好坏样本
        data_bad = feature[col].fillna('Blank')[labels == 1]
        data_good = feature[col].fillna('Blank')[labels == 0]
        iv_series[col] = iv_count(data_bad, data_good)

    return iv_series


def get_index_values(feature, labels=None, method='missing', bin_method='chi_square_map'):
    '''
    计算指定的指标值,get_index_values方法出入参如下:
    ------------------------------------------------------------
    入参结果如下:
        feature: 数据集的特征空间
        labels: 数据集的输出空间,默认None,计算psi、iv、importance时才需要传入该数据
        method: 需计算的指标值,目前支持缺失率"missing"、同质比"homogeny"、类别变量最大计数类别"category"、
            类别变量最大计数类别(包括缺失值)"category_miss"、相关性"collinear"、psi值"psi"、iv值"iv"、特征重要度"importance"
    ------------------------------------------------------------
    入参结果如下:
        iv_series: 各变量最大的psi值
    '''
    if method == 'missing':
        missing_series = feature.isnull().sum() / feature.shape[0]
        index_values = pd.DataFrame(missing_series)
        index_values.reset_index(inplace=True)
        index_values.columns = ['feature', 'missing_rate']

    elif method == 'homogeny':
        homogeny_series = feature.apply(lambda x: x.value_counts().max()) / feature.shape[0]
        index_values = pd.DataFrame(homogeny_series)
        index_values.reset_index(inplace=True)
        index_values.columns = ['feature', 'homogeny_rate']

    elif method == 'category':
        col_types = feature.dtypes
        categorical_feature = list(col_types[col_types == 'object'].index)
        category_series1 = feature[categorical_feature].apply(lambda x: x.value_counts().max())
        category_series2 = feature[categorical_feature].apply(lambda x: x.value_counts().max()) / feature.shape[0]
        index_values = pd.DataFrame({'category_max_cnt': category_series1, 'category_max_rate': category_series2})
        index_values.reset_index(inplace=True)
        index_values.columns = ['feature', 'category_max_cnt', 'category_max_rate']

    elif method == 'category_miss':
        col_types = feature.dtypes
        categorical_feature = list(col_types[col_types == 'object'].index)
        category_series1 = feature[categorical_feature].fillna('Blank').apply(lambda x: x.value_counts().max())
        category_series2 = feature[categorical_feature].fillna('Blank').apply(
            lambda x: x.value_counts().max()) / feature.shape[0]
        index_values = pd.DataFrame({'category_max_cnt': category_series1, 'category_max_rate': category_series2})
        index_values.reset_index(inplace=True)
        index_values.columns = ['feature', 'category_max_cnt', 'category_max_rate']

    elif method == 'collinear':
        index_values = feature.corr()

    elif method in ['psi', 'iv']:
        if labels is None:
            raise ValueError('请确保方法传入labels数据集')
        col_types = feature.dtypes
        cut_bin_dict = {}
        for col in list(col_types[col_types != 'object'].index):
            if bin_method == 'chi_square':
                cut_bin_dict[col] = chi_square_bin(feature[col], labels, value_range='default')
            elif bin_method == 'chi_square_map':
                cut_bin_dict[col] = chi_square_bin(feature[col], labels, value_range='max_min')
            elif bin_method == 'best_ks':
                cut_bin_dict[col] = best_ks_bin(feature[col], labels)
            else:
                raise ValueError('请传入正确的bin_method参数,bin_method应为分箱方法"chi_square"或"chi_square_map"或"best_ks"')

        if method == 'psi':
            if 'DATE' not in feature.columns:
                raise ValueError('请确保数据集有"DATE"变量,计算psi值需用到"DATE"变量')
            index_values = get_psi_dataframe(feature, list(feature.columns), cut_bin_dict=cut_bin_dict)

        else:
            iv_series = get_iv_series(feature, labels, list(feature.columns), cut_bin_dict=cut_bin_dict)
            index_values = pd.DataFrame(iv_series)
            index_values.reset_index(inplace=True)
            index_values.columns = ['feature', 'iv_value']

    elif method == 'importance':
        if labels is None:
            raise ValueError('请确保方法传入labels数据集')
        features = feature.copy()
        col_types = features.dtypes
        categorical_feature = list(col_types[col_types == 'object'].index)
        feature_names = list(features.columns)
        features[categorical_feature] = features[categorical_feature].astype('category')
        feature_importance_values = np.zeros(len(feature_names))
        # 设置每次迭代的随机种子，保证每次迭代不同，且结果可复现
        np.random.seed(1234)
        random_states = np.random.randint(1, 1000, size=10)

        # 特征重要度由十次训练结果取平均
        for i in range(10):
            model = LGBMClassifier(
                verbose=-1,
                random_state=random_states[i])
            eval_metric = 'auc'

            train_features, valid_features, train_labels, valid_labels = train_test_split(
                features,
                labels,
                test_size=0.15,
                random_state=random_states[i])
            # 训练提前停止模型,根据验证集eval_set的eval_metric结果,当有early_stopping_rounds次迭代分数没有提高时停止训练
            model.fit(
                train_features,
                train_labels,
                feature_name=feature_names,
                categorical_feature=categorical_feature,
                eval_metric=eval_metric,
                eval_set=[(valid_features, valid_labels)],
                early_stopping_rounds=100,
                verbose=False)

            feature_importance_values += model.feature_importances_ / 10

        index_values = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance_values
        })
        # 根据重要性对特性进行排序,并将特性重要性归一化,使其总和为1
        index_values = index_values.sort_values(
            'importance', ascending=False).reset_index(drop=True)
        index_values['importance'] = index_values['importance'] / index_values['importance'].sum()

    else:
        raise ValueError('请传入正确的method参数,method应需计算的指标值名字')

    return index_values
