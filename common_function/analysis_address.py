import numpy as np
import pandas as pd
import requests
import json
from ip2Region import Ip2Region


def get_ip_info_freeapi(ip):
    '''
    调用 http://freeapi.ipip.net/ 接口解析ip:
    ------------------------------------------------------------
    入参结果如下:
        ip: ip地址
    '''
    url = 'http://freeapi.ipip.net/'
    response = requests.get(url + ip)

    info = response.text.replace('\"', '').replace('[', '').replace(']', '').replace(' ', '').split(",")

    print("****************************************")
    print("您查询的IP地址 %s 来源地是：" % ip)
    print("国家：%s" % (info[0]))
    print("省份：%s" % (info[1]))
    print("城市：%s" % (info[2]))
    print("区域：%s" % (info[3]))
    print("运营商：%s" % (info[4].replace('\n', '')))


def get_ip_info_taobao(ip):
    '''
    调用 ttp://ip.taobao.com/ 接口解析ip:
    ------------------------------------------------------------
    入参结果如下:
        ip: ip地址
    '''
    url = 'http://ip.taobao.com/service/getIpInfo.php?ip='

    while 1:
        response = requests.get(url + ip).text
        if response.find('"code":0') != -1:
            break

    info = json.loads(response)['data']

    print("****************************************")
    print("您查询的IP地址 %s 来源地是：" % ip)
    print("国家：%s" % (info['country']))
    print("省份：%s" % (info['region']))
    print("城市：%s" % (info['city']))
    print("运营商：%s" % (info['isp']))


def get_ip_info_ip2region(db_file, ip):
    '''
    调用 ip2region 解析ip:
    ------------------------------------------------------------
    入参结果如下:
        db_file: 解析ip的db文件,为https://github.com/lionsoul2014/ip2region的data/ip2region.db
        ip: ip地址
    '''
    searcher = Ip2Region(db_file)
    info = searcher.memorySearch(ip)["region"].decode('utf-8').split('|')

    print("****************************************")
    print("您查询的IP地址 %s 来源地是：" % ip)
    print("国家：%s" % (info[0]))
    print("区域：%s" % (info[1]))
    print("省份：%s" % (info[2]))
    print("城市：%s" % (info[3]))
    print("运营商：%s" % (info[4]))


def get_ip_info_batch(db_file, ip_data):
    '''
    调用 ip2region 批量解析ip:
    ------------------------------------------------------------
    入参结果如下:
        db_file: 解析ip的db文件,为https://github.com/lionsoul2014/ip2region的data/ip2region.db
        ip_data: ip地址组成的Series或DataFrame
    ------------------------------------------------------------
    出参结果如下:
        ip_info: ip解析后的Series
    '''
    searcher = Ip2Region(db_file)

    ip_info = pd.Series(index=ip_data.index)

    for index in ip_data.index:
        try:
            info = searcher.memorySearch(ip_data[index])['region'].decode('utf-8')
        except OSError:
            info = np.nan
        ip_info[index] = info

    return ip_info
