# common_function
个人常用的一些函数

## data_process.py
* reduce\_memory\_usage 减少DataFrame所占内存


## data_check.py
* dataframe\_compare\_all 比较两个数据集共有索引、变量的值是否相同
* dataframe\_compare\_sigle 查看两个数据集选定变量不等样本的值比较


## index_values.py
* get\_index\_values 计算指定的指标值,目前支持缺失率"missing"、同质比"homogeny"、类别变量最大计数类别"category"、类别变量最大计数类别(包括缺失值)"category_miss"、相关性"collinear"、psi值"psi"、iv值"iv"、特征重要度"importance"


## analysis_address.py
* get\_ip\_info\_freeapi 调用freeapi接口解析ip
* get\_ip\_info\_taobao 调用taobao接口解析ip
* get\_ip\_info\_ip2region 调用ip2region解析ip，需准备ip2region.db和ip2Region.py
* get\_ip\_info\_batch 调用ip2region批量解析ip，需准备ip2region.db和ip2Region.py