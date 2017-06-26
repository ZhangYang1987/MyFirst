# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:57:28 2017

@author: zy
"""

import sys
from PyQt5 import QtCore,uic
from PyQt5 import QtWidgets

import xlrd as xlsrd
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

import pygal
import os
import math
from pylab import mpl
from matplotlib import gridspec 
mpl.rcParams['font.sans-serif'] = ['SimHei']


qtCreatorFile = "mainwindow.ui" # Enter file here.


Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    feature_names=[]
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.read_xlsx.clicked.connect(self.read_xls)
        self.pushButton_Cal.clicked.connect(self.add_head)
        
    def read_xls(self):
        DW=Data_Wash()
#        price = int(self.price_box.toPlainText())
#        tax = (self.tax_rate.value())
#        total_price = price  + ((tax / 100) * price)
#        total_price_string = "The total price with tax is: " + str(total_price)
#        self.results_window.setText(total_price_string)
        self.feature_names=DW.xlsxread('13#机组数据(sql).xlsx')
        for i in range(len(self.feature_names)):
            self.comboBox.addItem(self.feature_names[i])    
        
        
        
    def add_head(self):
#        qList = QtCore.QStringList(['a','b','c']) Qt5已经不支持，因为Qt是C++实现的，所以有List等概念的对象。但对于PyQt，因为Python本身支持List等，所以就没有这个对象了
        qlist=[]
        for i in range(len(self.feature_names)):
            qlist.append(self.feature_names[i])
       
        self.tableWidget.setHorizontalHeaderLabels(qlist)
        
        
#        

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
        aa=[]
        for i in range(1,table.nrows): #table.nrows获取行数
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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())        
        