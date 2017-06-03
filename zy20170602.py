# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 23:14:23 2017

@author: zy
"""

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as skimport neurolab as nl
import matplotlib.pyplot as plt #记住要用pyplot这个子模块
import xlrd as xlsrd

class Data_Wash():
    def xlsread(self,path):
        data=xlsrd.open_workbook(path)
        table=data.sheets()[1]
        aa=[]
        for i in range(table.nrows): #table.nrows获取行数
            aa.append(table.row_values(i)) #table.row_values(i)获取整行的数
        data_array=np.array(aa)
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
        frame_NoDupicated=frame.iloc[abc[abc==False].index]
        frame_Dupicated=frame.iloc[abc[abc==True].index]
        return frame,frame_NoDupicated,frame_Dupicated
test=Data_Wash()
data,data_nodup,data_dup=test.xlsread('C:/Users/zy/Documents/13#机组数据(sql).xlsx')

class Graph():
    def graph_line(self,chart_title,x_label,y_name,y_data):
        line_chart = pygal.Line()
        line_chart.title = chart_title
        line_chart.x_labels =x_label
        line_chart.add(y_name, y_data)
        line_chart.render_to_file('aaa.html')
       
    
    def graph_bar(self,chart_title,y_name,hist,bin_edges):
        histogram = pygal.Histogram()
        histogram.title=chart_title
        b=[]
        for i in range(len(hist)):
            a=(hist[i],bin_edges[i],bin_edges[i+1])
            b.append(a)
        histogram.add(y_name, b)
        histogram.render_to_file('hist.html')

chart_title='进汽量'
#x_label=map(str,list(data[0][1:100].index))
y_name='进汽量'
#y_data=list(data[0][1:100])
#
g=Graph()
#g.graph_line(chart_title,x_label,y_name,y_data)
hist,bin_edges=np.histogram(data_nodup[0],5)
g.graph_bar(chart_title,y_name,hist,bin_edges)



