# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 23:14:23 2017

@author: zy
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as sts
import sklearn as sk
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn import preprocessing
import sklearn.datasets as sk_dataset
import neurolab as nl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import xlrd as xlsrd
import pygal
import os
import math
from pylab import mpl
from matplotlib import gridspec 
mpl.rcParams['font.sans-serif'] = ['SimHei']


class Data_Wash():
    #从xls文件读取数据
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

    def preprocessing_for_tree(self,x,discretization,bins,labels):
        B_Stats=Base_Statistics()
        #discretization为x中需要离散化的指标
        #计算频率分布
        hist,bin_edges=B_Stats.histogram(discretization,bins)
        #计算每个段的频率
        density=list(hist/hist.sum())
        #为每个段设置一个标识符，这里以从0开始的整数为标识
        mark=list(range(len(bin_edges)-1))
        #设置target_names,为标签+出现频率
        target_names=[]
        for i in range(len(bin_edges)-1):
            target_names.append(labels[i]+","+str(density[i]))
        #计算target
        target=[]
        for index,item in enumerate(discretization):
            if item==bin_edges[bins]:
                target.append(mark[bins-1])
            for i in range(len(bin_edges)-1):
                if item>=bin_edges[i] and item<bin_edges[i+1]:
                    target.append(mark[i])
        #创建字典
#        n_dic=dict(data=data_nodup,feature_names=["主蒸汽流量","主蒸汽温度",\
#                                          "主蒸汽压力","机组实发功率","热网抽汽流量"],\
#                                        target=target,target_names=target_names)
        dataset=sk_dataset.base.Bunch(data=x,feature_names=["主蒸汽流量","主蒸汽温度",\
                                          "主蒸汽压力","机组实发功率","热网抽汽流量"],\
                                        target=target,target_names=target_names)
        return dataset
            
            



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

class Base_Statistics():
    
    #描述性统计
    def descriptive_statistics(self,x):
        
        #集中趋势的度量
        
        #均值
        mean=np.mean(x)
        
        #中位数
        median=np.median(x,axis=0)
        
        #众数
        mode=sts.mode(x)
        
        #上四分位数Q1
        Q1=x.quantile(0.25)
        
        
        #下四分位数Q1
        Q2=x.quantile(0.75)
        
        
        #离散趋势的度量
        
        #最大值
        mx=np.max(x)
        
        #最小值
        mn=np.min(x)
        
        #四分位差
        IQR=Q2-Q1
        
        #极差
        ptp=np.ptp(np.array(x),axis=0)
        
        #方差
        var=np.var(x)
        
        #标准差
        std=np.std(x)
        
        #离散系数
        cv=std/mean
        
        
        #峰度  
        kurtosis=sts.kurtosis(x)
        
        #偏度
        skewness=sts.skew(x)
        
        return mean,median,mode,Q1,Q2,mx,mn,IQR,ptp,var,std,cv,kurtosis,skewness
    
    
    #频率分布计算
    def histogram(self,x,bins):
        hist,bin_edges=np.histogram(x,bins=bins)
        return hist,bin_edges
    
    #F值和p值计算
    def f_res(self,x,y):
        F,p=f_regression(x,y)
        return F,p
    
    #PCA计算，包含了多自变量的F值和p值计算    
    def pca(self,x):
        pca=PCA(n_components='mle')#注意，如果选了mle自动选取特征数，且数据超过500条即会报错'>=' not supported between instances of 'str' and 'int'
        #pca=PCA(n_components=1)
        pca.fit(x)
        components=pca.components_
        explained_variance=pca.explained_variance_
        explained_variance_ratio=pca.explained_variance_ratio_
        n_components=pca.n_components_
        mean=pca.mean_
        noise_variance=pca.noise_variance_
        newData=pca.fit_transform(x)
        return components,explained_variance,explained_variance_ratio,n_components,\
                mean,noise_variance,newData
    
    #数据是否正态分布的检验
    def k_s(self,x):
        D,p=sts.kstest(x,'norm')
        return D,p
    
    #Z标准化,也叫标准差标准化
    def Z_score(self,x):
        scaler=preprocessing.StandardScaler().fit(x)
        mean=scaler.mean_
        std=scaler.std_
        scaler_trans=scaler.transform(x)
        return mean,std,scaler_trans
    
    #MinMax标准化，将数据归一到【0，1】
    def MinMax(self,x):
        minmax_scaler=preprocessing.MinMaxScaler().fit(x)
        scaler_trans=minmax_scaler.transform(x)
        return scaler_trans

class Save():
    def save_as_json(self,dic):
        import json
        a=json.dumps(dic)
        return a
    
    def save_as_dict(self,dic):
#        path="C:/Users/zy/python_practice/dict.txt"
        path="./dict.txt"
        file=open(path,'w')
        file.write(str(dic))
        file.close()

##原始数据区
#test=Data_Wash()
#data,data_nodup,data_dup=test.xlsread('C:/Users/zy/Documents/13#机组数据(sql).xlsx')
#
#
#    
###基础统计分析区
B_Stats=Base_Statistics()
mean,median,mode,Q1,Q2,mx,mn,IQR,ptp,var,std,cv,kurtosis,skewness\
=B_Stats.descriptive_statistics(data_nodup)
##F,p=B_Stats.f_res(data_nodup.iloc[:,1:5],data_nodup[0])
##components,explained_variance,explained_variance_ratio,n_components,\
##                mean,noise_variance,newData=B_Stats.pca(data_nodup.iloc[0:500,1:5])
##D,p=B_Stats.k_s(data_nodup[3])
##mean,std,scaler_trans=B_Stats.Z_score(data_nodup)
##mx_scaler=B_Stats.MinMax(data_nodup)


#决策树
###发电蒸汽单耗
#a=data_nodup[3]/(data_nodup[0]-data_nodup[4])
#
##计算频率分布
##hist,bin_edges=B_Stats.histogram(a,10)
#
##计算每个段的频率
##b=list(hist/hist.sum())
#
##为每个段设置标签
#c=["A+","A","A-","B+","B","B-","C+","C","C-","D"]
#
#dataset=test.preprocessing_for_tree(data_nodup,a,10,c)

#save=Save()
#save.save_as_dict(n_dic)
#a=save_json.save_as_json(n_dic)
#from sklearn.datasets import load_iris
#iris = load_iris()

#from sklearn import tree 
#from sklearn.tree import export_graphviz
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(dataset.data, dataset.target)
#with open("dataset.dot", 'w') as f:
#    f = tree.export_graphviz(clf, out_file=f)
#os.unlink('dataset.dot')
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#import pydotplus 
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = pydotplus.graph_from_dot_data(dot_data) 
#graph.write_pdf("dataset.pdf") 
#from IPython.display import Image  
#dot_data = tree.export_graphviz(clf, out_file=None,
#                                feature_names=dataset.feature_names,
#                                class_names=dataset.target_names, 
#                                filled=True, rounded=True, 
#                                special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data)  
#Image(graph.create_png())

#用决策树进行预测 
#x=np.array([867.567688,533.920044,14.061001,272.029327,27.75]).reshape(1,-1) 
#clf.predict_proba(x)


#随机森林回归将特征按照重要程度排序
#from sklearn.ensemble import RandomForestRegressor
#
#X=dataset.data
#Y=a
#names=dataset.feature_names
#rf=RandomForestRegressor()
#rf.fit(X,Y)
#print("Features sorted by their score:")
#print(sorted(zip(map(lambda x: round(x,4),rf.feature_importances_),names),reverse=True))

"""
##作图区
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
"""