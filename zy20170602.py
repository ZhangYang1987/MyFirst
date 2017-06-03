# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 23:14:23 2017

@author: zy
"""

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import neurolab as nl
import matplotlib as plt
import xlrd as xlsrd

class Data_Wash():
    def xlsread(self,path):
        data=xlsrd.open_workbook(path)
        table=data.sheets()[1]
        aa=[]
            aa.append(table.row_values(i)) #table.row_values(i)获取整行的数
        frame=pd.DataFrame(data_array)
        abc=frame.duplicated([0])
        for i in range(1,table.ncols):
            abc&=frame.duplicated([i])
#        data_bool=pd.DataFrame(np.array([frame.duplicated([0]),\
#                                         frame.duplicated([1]),
#                                         frame.duplicated([2]),
#                                         frame.duplicated([3]),
#                                         frame.duplicated([4]),]).T)
        #data_bool.columns=list('abcde')
        
        #return data_bool
        return abc
#test=Data_Wash()
#s=test.xlsread('C:/Users/zy/Documents/天业13#机组数据(sql).xlsx')
#abc[abc==True].index
