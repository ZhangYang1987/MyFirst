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
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import sklearn as skimport neurolab as nl
import matplotlib.pyplot as plt #记住要用pyplot这个子模块

import xlrd as xlsrd
import pygal
import os
import math
from pylab import mpl
from matplotlib import gridspec 
mpl.rcParams['font.sans-serif'] = ['SimHei']
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
        
    def single_hist(self,x,n_bins,title):
        #fig,ax= plt.subplots()
        plt.hist(x,n_bins)
        plt.title(title)        
        plt.savefig("test.svg", format="svg")
        plt.show()
        
    def multi_hist(self,x,n_bins,labels,colors):
        num_ax=len(x.columns)
        nrows=2
        ncols=3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15,8))
        if num_ax<=3:
            nrows=1
            ncols=num_ax
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        elif num_ax==4:
            nrows=2
            ncols=2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        elif num_ax==5|6:
            nrows=2
            ncols=3
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        elif num_ax==7|8:
            nrows=2
            ncols=4
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)  
        a=[]
        for i in range(nrows):
            for j in range(ncols):
                a.append(axes[i,j])
#        ax0, ax1, ax2, ax3, ax4,ax5 = axes.flatten()
#        a=list([ax0,ax1,ax2,ax3,ax4])
        for i in range(num_ax):
            a[i].hist([x[i]],n_bins,color=colors[i])#https://stackoverflow.com/questions/19523563/python-typeerror-int-object-is-not-iterable
            a[i].set_title(labels[i])
        lab='异常数据','重复数据','有效数据'
        sizes=[0,1459,7211]
        explode = (0.1, 0.1, 0.1)
        a[5].pie(sizes, explode=explode, labels=lab, autopct='%1.1f%%',\
                 shadow=True, startangle=90,data=True)
        a[5].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig.tight_layout()
        plt.savefig("multi_hist_test.svg", format="svg")
        plt.show()



chart_title='进汽量'
#x_label=map(str,list(data[0][1:100].index))
y_name='进汽量'
#y_data=list(data[0][1:100])
#
g=Graph()
#g.single_hist(data_nodup[0],10,chart_title)
colors=['tomato','tan','peru','teal','olive']
labels=['进汽量','进汽温度','进汽压力','实发功率','抽汽量']
g.multi_hist(data_nodup,10,labels,colors)
#g.graph_line(chart_title,x_label,y_name,y_data)
#hist,bin_edges=np.histogram(data_nodup[0],10)
#g.graph_bar(chart_title,y_name,hist,bin_edges)
