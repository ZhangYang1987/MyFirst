# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:57:28 2017

@author: zy
"""

import sys
from PyQt5 import QtCore,uic
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtWebKitWidgets 
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import xlrd as xlsrd
import numpy as np
import pandas as pd
import scipy.stats as sts

import matplotlib.pyplot as plt

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

import time
import copy
import numpy as np
import random
import pandas as pd
import scipy as sp
import scipy.stats as sts
import sklearn as sk
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn import preprocessing
import sklearn.datasets as sk_dataset
import pywt #小波分析
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



qtCreatorFile = "mainwindow.ui" # Enter file here.


Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)



class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    data=pd.DataFrame([])
    feature_names=[]
    samples=pd.DataFrame([])
    sample_re=pd.DataFrame([])
    data_nodup=[]
    rf=RandomForestRegressor()
    fileName=""
    num_All=0
    num_valid=0
    num_duplicated=0
    num_invalid=0
    figure_type="222"
    
    x_train=[]
    y_train=[]
    layer_num=2
    neuron_num=3,1
    initial_W=[]
    lr=[]
    goal=0.01
    Epochs=500
    
    
    


    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        #self.read_xlsx.clicked.connect(self.read_xls)
        self.pushButton_Cal.clicked.connect(self.add_head)
#        self.comboBox_Y.currentIndexChanged.connect(self.Y_select)
#        self.comboBox_X.currentIndexChanged.connect(self.X_select)
        self.pushButton_R2_Cal.clicked.connect(self.pearsonr_cal)
        self.pushButton_select_Y.clicked.connect(self.randforest_select_Y)
        self.pushButton_select_X.clicked.connect(self.randforest_select_X)
        self.pushButton_select_X_2.clicked.connect(self.randforest_delete_X)
        self.pushButton_RandForest.clicked.connect(self.randforest_cal)
        self.read_xlsx.clicked.connect(self.read_xls)
        self.pushButton_figure.clicked.connect(self.figure)        
        self.radioButton_hist.toggled.connect(self.figure_type_get)
        self.radioButton_boxplot.toggled.connect(self.figure_type_get)
        self.actionOpen.triggered.connect(self.read_xls)
        self.treeWidget.itemClicked.connect(self.onClick)
        self.pushButton_neural.clicked.connect(self.neural_cal)
        self.pushButton_RandForestRegress.clicked.connect(self.RandForesRegress)
        self.pushButton_traindata.clicked.connect(self.train_data)
        self.pushButton_Sampling.clicked.connect(self.Sampling)
        
        
        
        
        
#        self.treeWidget.setHeaderLabel("系统菜单")
#        Item1=QtWidgets.QTreeWidgetItem(self.treeWidget)
#        Item1.setText(0,"数据预处理")
#        Item1_1=QtWidgets.QTreeWidgetItem(Item1)
#       
#        Item1_1.setText(0,"加权采样")
#        Item1.addChild(Item1_1)
#        

    def Sampling(self):
#        DR=Data_Read()
#        self.read_xls()
#        self.data=DR.xlsread(self.fileName)
        Sam=Sampling()
        Data_All_num,Field=Sam.Rand_Sampling(self.data)
        self.lineEdit_Data_All_num.setText(str(Data_All_num))
        self.listWidget_Field.addItems(Field)
        k=int(self.lineEdit_Sampling_num.text())
        
        if self.checkBox_Repetition.isChecked():
            self.samples_re=pd.DataFrame([self.data.iloc[random.randint(0,Data_All_num),:] \
                                                         for i in range(k)])
            self.textEdit_Sampling_Result.setText(str(self.samples_re.shape[0]))            
        else:
            s_k=random.sample(range(Data_All_num),k)
            try:
                self.samples=self.data.iloc[s_k,:]
                self.textEdit_Sampling_Result.setText(str(self.samples.shape[0]))
                print(self.samples)
            except:
                QtWidgets.QMessageBox.information(self,"提示框","采样失败")
                


    def onClick(self):
        txt=self.treeWidget.currentItem().text(0)
        tab_index=0
        if txt=="数据预处理":
            tab_index=0
        if txt=="特征处理":
            tab_index=1
        if txt=="统计分析":
            tab_index=2
        if txt=="回归":
            tab_index=3
        if txt=="神经网络回归":
            tab_index=3
            tab_regresssion_index=2
        if txt=="多项式回归":
            tab_index=3
            tab_regresssion_index=1
        if txt=="线性回归":
            tab_index=3
            tab_regresssion_index=0
        if txt=="随机森林回归":
            tab_index=3
            tab_regresssion_index=3
        self.tabWidget_2.setCurrentIndex(tab_index)
        self.tabWidget_Regression.setCurrentIndex(tab_regresssion_index)
        
        
        
    def read_xls(self):
#        self.fileDialog=QtWidgets.QFileDialog()
        self.fileName,filetype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                    "选取文件",
                                                                    "C:/Users/zy/python_practice/QT/zy20170626/zy20170626",
                                                                    "xlsxfile(*.xlsx);;xlsfile(*.xls)")
        if len(self.fileName)==0:
            QtWidgets.QMessageBox.information(self,"数据读取失败","未选择有效的数据文件")
        else:
            QtWidgets.QMessageBox.information(self,"提示框","数据已经加载，可以进行后续计算")
            DR=Data_Read()
            self.data=DR.xlsread(self.fileName)
            
        
        
    def figure_type_get(self):
        if self.radioButton_hist.isChecked():
            self.figure_type=self.radioButton_hist.text()
        elif self.radioButton_boxplot.isChecked():
            self.figure_type=self.radioButton_boxplot.text()
        else:
            self.figure_type=""
        

    def figure(self):
        graph=Graph()
        if (self.figure_type=="频率分布直方图"):
            labels=self.feature_names
            colors=['tomato','tan','peru','teal','olive']
            graph.multi_hist(self.data_nodup,10,labels,colors,self.num_invalid,self.num_duplicated,self.num_valid)
            self.view1=QtWebKitWidgets.QWebView()
            self.view1.setWindowTitle(self.figure_type)
            self.view1.load(QtCore.QUrl("file:///C:/Users/zy/python_practice/QT/zy20170626/zy20170626/multi_hist_test.svg"))
            self.view1.showNormal()
            
        elif (self.figure_type=="箱形图"):
            labels=self.feature_names
#            colors=['tomato','tan','peru','teal','olive']
            graph.boxplot(self.data_nodup,labels)
            self.view2=QtWebKitWidgets.QWebView()
            self.view2.setWindowTitle(self.figure_type)
            self.view2.load(QtCore.QUrl("file:///C:/Users/zy/python_practice/QT/zy20170626/zy20170626/boxplot.png"))
            self.view2.showNormal()

    def randforest_select_Y(self):
        self.lineEdit_select_Y.setText(self.listWidget_All.currentItem().text())
        
    def randforest_select_X(self):
        self.listWidget_select_X.addItem(self.listWidget_All.currentItem().text())
        
    def randforest_delete_X(self):
        self.listWidget_select_X.takeItem(self.listWidget_select_X.currentRow())
        
    def randforest_cal(self):
        DW=Data_Wash()
        lst=[]
        for i in range(self.listWidget_select_X.count()):
            lst.append(self.listWidget_select_X.item(i).text())
        x=self.data_nodup[lst]
        y=self.data_nodup[self.lineEdit_select_Y.text()]
        result_txt=DW.RandomForestRegressor(x,y)
        self.textEdit_Randforest_result.setText(result_txt)
        
    def train_data(self):
        DW=Data_Wash()        
        self.x_train,self.y_train=DW.train_data(self.data_nodup)
        
    def RandForesRegress(self):
        Re=Regression()
        x_predict=self.x_train.copy()
        y_predict=Re.RandForestRegress(x_train=self.x_train,y_train=self.y_train,\
                                       x_predict=x_predict)
#        Accuracy=metrics.accuracy_score(self.y_train,y_predict)
        
        mse=sum((self.y_train-y_predict)**2)/sum(self.y_train**2)*100
        self.textEdit_R2.setText(str(Accuracy))
    
    def pearsonr_cal(self):
        DW=Data_Wash()
        x=self.data_nodup[self.comboBox_X.currentText()]
        y=self.data_nodup[self.comboBox_Y.currentText()]
        r,p=DW.pearson(x,y)
        self.lineEdit_R2.setText(str('%.5f' % r))
        self.lineEdit_p_value.setText(str('%.5f' % p))
        
    def add_head(self):
     
        self.tableWidget=QtWidgets.QTableWidget(16,6)
       
        
        items=['有效数据','均值','标准差','最小值','最大值','上四分位数',\
               '中位数','下四分位数','离散系数','峰度','偏度',]
        
        DW=Data_Wash()
#        data,self.data_nodup,data_dup=DW.NoDup(self.data)
        data,self.data_nodup,data_dup=DW.NoDup(self.samples)
        self.num_All=data.shape[0]
        self.num_valid=self.data_nodup.shape[0]
        self.num_duplicated=data_dup.shape[0]
        self.num_invalid=0
        self.feature_names=self.data_nodup.columns
        self.lineEdit_num_All.setText(str(self.num_All))
        self.lineEdit_num_valid.setText(str(self.num_valid))
        self.lineEdit_num_duplicated.setText(str(self.num_duplicated))
        self.lineEdit_num_invalid.setText(str(self.num_invalid))
        
        
        for i in range(len(self.feature_names)):
            self.comboBox_X.addItem(self.feature_names[i])
            self.comboBox_Y.addItem(self.feature_names[i])
        qlist=["项目"]
        list(map(lambda x:qlist.append(x),self.feature_names))#因为在3.3里面，map(),filter()这些的返回值已经不再是list,而是iterators, 所以想要使用，只用将iterator 转换成list 即可
        self.tableWidget.setHorizontalHeaderLabels(qlist)    
        count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness=DW.descriptive_statistics(self.data_nodup)
        stats=[count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness]
        for i in range(len(items)):                   
            self.newItem = QtWidgets.QTableWidgetItem(items[i])
            self.tableWidget.setItem(i,0,self.newItem)
            for j in range(len(mean)):
                self.newItem = QtWidgets.QTableWidgetItem(str('%.5f' % stats[i][j]))
                self.tableWidget.setItem(i,j+1,self.newItem)
        height=0
        row=self.tableWidget.rowCount()
        for i in range(1,row):
            height+=self.tableWidget.rowHeight(i)*1.5
            
        width=0
        column=self.tableWidget.columnCount()
        for i in range(1,column):
            width+=self.tableWidget.columnWidth(i)*1.5
            
        self.tableWidget.resize(width,height)
        self.tableWidget.setAlternatingRowColors(1)
        self.tableWidget.setWindowTitle("描述性统计分析")
        self.tableWidget.show()
        self.listWidget_All.addItems(self.feature_names)

    def neural_cal(self):
        re=Regression()
        train_data=self.data_nodup
        input_train_minmax_scaler,output_train_minmax_scaler,net=\
        re.neural(train_data=train_data,layer_num=self.layer_num,\
                  neuron_num=self.neuron_num,initial_W=self.initial_W,\
                  lr=self.lr,goal=self.goal,epochs=self.Epochs)

class Sampling():
    
    def Rand_Sampling(self,data):        
        Data_All_num=data.shape[0]
        Field=data.columns
        return Data_All_num,Field

class Data_Read():
    def xlsread(self,path):
        data=xlsrd.open_workbook(path)
        table=data.sheets()[1]
        feature_name=table.row_values(0)
        row=[]
        for i in range(1,table.nrows): #table.nrows获取行数
            row.append(table.row_values(i)) #table.row_values(i)获取整行的数
        data_array=np.array(row)
        frame=pd.DataFrame(data_array)
        frame.columns=[feature_name]
        return frame
    
    
class Data_Wash():

        
    #从xls文件读取数据
    def NoDup(self,data):
        abc=data.duplicated(data.columns[0])
        for i in range(1,len(data.columns)):
            abc&=data.duplicated([data.columns[i]])
        frame_NoDupicated=data.loc[abc[abc==False].index]
        frame_Dupicated=data.loc[abc[abc==True].index]
        return data,frame_NoDupicated,frame_Dupicated
    
    def train_data(self,data):
        x_train=data.iloc[:,0:-1]
        y_train=data.iloc[:,-1]
        return x_train,y_train
    
    def descriptive_statistics(self,x):
        #DataFrame自带描述性统计的方法
        des=x.describe()
        count=des.loc['count']
        mean=des.loc['mean']
        std=des.loc['std']
        mn=des.loc['min']
        per25=des.loc['25%']
        per50=des.loc['50%']
        per75=des.loc['75%']
        mx=des.loc['max']        
        #离散系数
        cv=std/mean        
        #峰度  
        kurtosis=sts.kurtosis(x)        
        #偏度
        skewness=sts.skew(x)
        
        return count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness
#        return des,cv,kurtosis,skewness


    #计算皮尔森相关系数
    def pearson(self,x,y):        
        pearson=sts.pearsonr(x,y)
        return pearson
    
    
    def RandomForestRegressor(self,x,y):
        self.rf=RandomForestRegressor()
        self.rf.fit(x,y)
        names=x.columns
        result_txt="特征重要性排列如下：\n"+str(sorted(zip(map(lambda x: \
                                                     str('%.5f' %x),self.rf.feature_importances_),names),reverse=True))
        return result_txt
        
class Regression():
    
    
    def RandForestRegress(self,x_train,y_train,x_predict):
        self.rf=RandomForestRegressor()
        self.rf.fit(x_train,y_train)
        y_predict=self.rf.predict(x_predict)
        print(y_predict)
        return y_predict
        
        
        
    #区域划分
    def split(self,x,num_split):
        des=x.describe()
        count=des.loc['count'][0]
        x_num=len(x.columns)-1
        
        x_split=np.zeros([max(num_split),x_num])
        for i in range(x_num):
            x_split[0:num_split[i],i]=np.linspace(des.loc['min'][i],\
                   des.loc['max'][i],num_split[i])
#        x1_split=np.linspace(np.min(x1),np.max(x1),27)
#        x2_split=np.linspace(np.min(x2),np.max(x2),27)
        x_split_sum=np.zeros([max(num_split),x_num])
        
        x_k=np.zeros([max(num_split),x_num])
        
        for i in range(int(count)):
            for k in range(x_num):
                for j in range(num_split[k]):
                    if x_split[j,k]<x.iloc[i,k]<=x_split[j+1,k]:
                        x_split_sum[j,k]+=x.iloc[i,len(x.columns)-1]
                        x_k[j,k]+=1
                
        x_split_average=x_split_sum/x_k    
        
        return x_split,x_split_average
    
    
     #神经网络拟合
    def neural(self,train_data,layer_num,neuron_num,initial_W,lr,goal,epochs):
        x_num=len(train_data.columns)
        input_train=train_data.iloc[:,0:x_num-1]
        output_train=(train_data.iloc[:,x_num-1]).values.reshape(-1,1)
        input_train_minmax_scaler=preprocessing.MinMaxScaler().fit(input_train)
        input_train_scaler_trans=input_train_minmax_scaler.transform(input_train)
        output_train_minmax_scaler=preprocessing.MinMaxScaler().fit(output_train)
        output_train_scaler_trans=output_train_minmax_scaler.transform(output_train)
        
#        net_layer=list(neuron_num).copy()
#        net_layer=list.append(net_layer,1)
        net_layer=[5,3,1]
        inp=input_train_scaler_trans
        tar=output_train_scaler_trans
        a=[]
        for i in range(x_num-1):
            a.append([0,1])
##        net=nl.net.newff([[0,1],[0,1]],net_layer)
        net=nl.net.newff(a,net_layer)
#        error=net.train(inp,tar,epochs=500,show=100,goal=0.001)
        error=net.train(inp,tar,show=10)
##        error=nl.train.train_gdx(net,inp,tar,epochs=500,show=100,goal=0.001)
        out=net.sim(inp)
        out_inverse=output_train_minmax_scaler.inverse_transform(out)
##        test_input=input_train_minmax_scaler.transform([test])
##        test_out=output_train_minmax_scaler.inverse_transform(net.sim(test_input))
##        return out_inverse,test_out,error[-1]
        return input_train_minmax_scaler,output_train_minmax_scaler,net
    
     #多项式拟合
    def polyfit(self,data,n):
        x_num=len(data.columns)
        x=data.iloc[:,0:x_num-1]
        y=data.iloc[:,x_num-1]
        result=[]
        y_poly=[]
        R2=[]
        coefficient=[]
        y2_sum=sum(list(map(lambda x:x**2,y)))
        for i in range(x_num-1):
            z=np.polyfit(x[i],y,n,full=True)
            R2.append((y2_sum-z[1])/y2_sum)
            coefficient.append(z[0])
            y_poly.append(np.polyval(z[0],x[i]))
#           y_val=np.polyval(z[0],test[i])
            result.append(R2)
#            result.append(y_val)
#        return y_poly,result
        return R2,coefficient





        
class Graph():
    def multi_hist(self,x,n_bins,labels,colors,num_invalid,num_duplicated,num_valid):
        num_ax=len(x.columns)
        nrows=2
        ncols=3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10,6))
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

        for i in range(1,num_ax+1):
            a[i].hist([x.iloc[:,i-1]],n_bins,color=colors[i-1])#https://stackoverflow.com/questions/19523563/python-typeerror-int-object-is-not-iterable
            a[i].set_title(labels[i-1])
        lab='异常数据','重复数据','有效数据'
        sizes=[num_invalid,num_duplicated,num_valid]
        explode = (0.1, 0.1, 0.1)
        a[0].pie(sizes, explode=explode, labels=lab, autopct='%1.1f%%',\
                 shadow=True, startangle=90,data=True)
        a[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig.tight_layout()
        plt.savefig("multi_hist_test.svg", format="svg")
        plt.clf()
#        plt.show()   


    def boxplot(self,x,labels):
        minmax_scaler=preprocessing.MinMaxScaler().fit(x)
        scaler_trans=minmax_scaler.transform(x)
        plt.boxplot(np.array(scaler_trans),labels=labels,notch=True)        
        plt.savefig("boxplot.png", format="png",dpi=300)
        plt.clf()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())        
        