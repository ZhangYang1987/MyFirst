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

import xlrd as xlsrd
import numpy as np
import pandas as pd
import scipy.stats as sts

import matplotlib.pyplot as plt

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

qtCreatorFile = "mainwindow.ui" # Enter file here.


Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    feature_names=[]
    data_nodup=[]
    rf=RandomForestRegressor()
    fileName=""
    num_All=0
    num_valid=0
    num_duplicated=0
    num_invalid=0
    figure_type="222"

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
        
#            
        
    def read_xls(self):
#        self.fileDialog=QtWidgets.QFileDialog()
        self.fileName,filetype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                    "选取文件",
                                                                    "C:/Users/zy/python_practice/QT/zy20170626/zy20170626",
                                                                    "xlsxfile(*.xlsx);;xlsfile(*.xls)")
        
        QtWidgets.QMessageBox.about(self,"提示框","数据已经加载，可以进行后续计算")
        
    def figure_type_get(self):
        if self.radioButton_hist.isChecked():
            self.figure_type=self.radioButton_hist.text()
        else:
            self.figure_type=""

    def figure(self):
        graph=Graph()
        if (self.figure_type=="频率分布直方图"):
            labels=self.feature_names
            colors=['tomato','tan','peru','teal','olive']
            graph.multi_hist(self.data_nodup,10,labels,colors)
            self.view=QtWebKitWidgets.QWebView()
            self.view.setWindowTitle(self.figure_type)
            self.view.load(QtCore.QUrl("file:///C:/Users/zy/python_practice/QT/zy20170626/zy20170626/multi_hist_test.svg"))
            self.view.showNormal()

        
        
#    def X_select(self):
#        self.textEdit_X.setText(self.comboBox_X.currentText())
#    
#    def Y_select(self):
#        self.textEdit_Y.setText(self.comboBox_Y.currentText())
    
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
    
    def pearsonr_cal(self):
        DW=Data_Wash()
        x=self.data_nodup[self.comboBox_X.currentText()]
        y=self.data_nodup[self.comboBox_Y.currentText()]
        r,p=DW.pearson(x,y)
        self.textEdit_R2.setText(str('%.5f' % r))
        self.textEdit_p_value.setText(str('%.5f' % p))
        
    def add_head(self):
#        qList = QtCore.QStringList(['a','b','c']) Qt5已经不支持，因为Qt是C++实现的，所以有List等概念的对象。但对于PyQt，因为Python本身支持List等，所以就没有这个对象了
        
        self.tableWidget=QtWidgets.QTableWidget(16,6)
       
        
        items=['有效数据','均值','标准差','最小值','最大值','上四分位数',\
               '中位数','下四分位数','离散系数','峰度','偏度',]
        
        DW=Data_Wash()
        data,self.data_nodup,data_dup=DW.xlsread(self.fileName)
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
#        for i in range(len(self.feature_names)):
#            qlist.append(self.feature_names[i])
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

        

class Data_Wash():
    
    def xlsxread(self,path):
        data=xlsrd.open_workbook(path)
        table=data.sheets()[1]
        aa=table.row_values(0)
        return aa
        
    #从xls文件读取数据
    def xlsread(self,path):
        data=xlsrd.open_workbook(path)
        table=data.sheets()[1]
        feature_name=table.row_values(0)
        row=[]
        for i in range(1,table.nrows): #table.nrows获取行数
            row.append(table.row_values(i)) #table.row_values(i)获取整行的数
        data_array=np.array(row)
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
        frame_NoDupicated.columns=[feature_name]
        frame_Dupicated=frame.iloc[abc[abc==True].index]
        return frame,frame_NoDupicated,frame_Dupicated
    
    
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
        
        
class Graph():
    def multi_hist(self,x,n_bins,labels,colors):
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
        sizes=[0,1459,7211]
        explode = (0.1, 0.1, 0.1)
        a[0].pie(sizes, explode=explode, labels=lab, autopct='%1.1f%%',\
                 shadow=True, startangle=90,data=True)
        a[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig.tight_layout()
        plt.savefig("multi_hist_test.svg", format="svg")
#        plt.show()        
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())        
        