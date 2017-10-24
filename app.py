# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:22:34 2017

@author: zy
"""
import random
import scipy.stats as sts

import time
from flask import Flask, jsonify,render_template,url_for,redirect,session
from flask import request
from flask import json
from flask import abort
from flask import Response
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm as Form

#form wtforms import TextField
#from flask_wtf import TextField, StringField,SubmitField,TextAreaField,IntegerField
#from wtforms.validators import Required 
#from flask_wtf import TextField,StringField,SubmitField,TextAreaField,IntegerField
from wtforms import TextField, StringField,SubmitField,TextAreaField,IntegerField,BooleanField,\
RadioField,SelectField,SelectMultipleField,FloatField
from wtforms import fields
from wtforms.validators import DataRequired,ValidationError
from wtforms import validators

from sklearn.externals import joblib
from sklearn import linear_model
from sklearn import preprocessing

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import neurolab as nl
import neurolab.error as error 
import math
import pymysql

import random
from pyecharts import Scatter3D
from pyecharts.constants import DEFAULT_HOST
from pyecharts import Bar
from pyecharts import Pie
from pyecharts import Boxplot

app=Flask(__name__)
#app.config.from_object('config')
bootstrap=Bootstrap(app)


tasks=[
       {
            'id':1,
            'title':u'Buy gro',
            'description':u'Milk,Cheese,Pizza,Fruit'
        },
        {
            'id':2,
            'title':u'Learn Python',
            'description':u'Need to find '     
        }]

train=[
       {
        'x_train_name':['Steam_in_Q','Steam_in_T','Steam_in_P','Steam_out_Q','P'],
        'y_train_name':u'ss'
        }]

data_num=0
data=[]
samples=[]
hist_selected_column=""
Selected_1_default_item=0
Selected_2_default_item=0
x=[]
y=[]
x_linear_regression=[]
y_linear_regression=[]
choices=[]    
X_train=[]
X_test=[]
y_train=[]
y_test=[]
fea_import=[]
input_train_scaler_transform=[]
output_train_scaler_transform=[]


class Submit(Form):
#    user_email = StringField("email address",[validators.Email()])
#    api = StringField("api",[DataRequired()])
#    submit = SubmitField("Submit")
#    code = IntegerField("code example: 200",[DataRequired()])
#    alias = StringField("alias for api")
#    data = TextAreaField("json format",[DataRequired()])
    s_in_Q=StringField('主蒸汽流量',[DataRequired()],render_kw={\
                                "style":"width:280px"})
    s_in_T=StringField('主蒸汽温度',[DataRequired()],render_kw={\
                                "style":"width:280px"})
    s_in_P=StringField('主蒸汽压力',[DataRequired()],render_kw={\
                                "style":"width:280px"})
    s_out_Q=StringField('热网抽汽流量',[DataRequired()],render_kw={\
                                "style":"width:280px"})
    submit=SubmitField('提交',render_kw={"style":"class:btn btn-primary"})
    
class Get_Data_Submit(Form):
    submit=SubmitField('读取数据',render_kw={"style":"class:btn btn-primary"})
    
class Samping_Submit(Form):
    sampling_count=StringField('采样个数')
    sampling_ratio=StringField('采样比例')
    Repetition=BooleanField('放回采样')
    submit=SubmitField('采样',render_kw={"style":"class:btn btn-primary"})
    
    
class Missing_Process_Submit(Form):
    Missing_Process=RadioField('缺失值处理',choices=[(1,'删除缺失值'),(2,'删除包含缺失值的行'),(3,'删除所有字段都是缺失值的行'),\
                                                (4,'删除只针对xx字段包含缺失值的行')],default=1)
    submit=SubmitField('处理',render_kw={"style":"class:btn btn-primary"})

class Describe_Statistics_Submit(Form):
    submit=SubmitField('处理',render_kw={"style":"class:btn btn-primary"})
    
class Pearson_cal_Submit(Form):
    submit_pearson=SubmitField('处理',render_kw={"style":"class:btn btn-primary"})
        
class Statistics():
    def descriptive_statistics(self,data):
        des=data.describe().round(2)
        count=des.loc['count']
        mean=des.loc['mean']
        std=des.loc['std']
        mn=des.loc['min']
        per25=des.loc['25%']
        per50=des.loc['50%']
        per75=des.loc['75%']
        mx=des.loc['max']        
        #离散系数
        cv=(std/mean).round(2)        
        #峰度  
        kurtosis=sts.kurtosis(data).round(2)        
        #偏度
        skewness=sts.skew(data).round(2)
        
        return count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness
    
    def pearson(self,x,y):
        pearson=sts.pearsonr(x,y)
        return pearson
    
    def spearman(self,x,y):
        spearman=sts.spearmanr(x,y)
        return spearman
    
class Hist_Graph(Form):
    item=SelectField("选择列表",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=0)
    submit1=SubmitField("提交")    

class Selected_1(Form):
    global Selected_1_default_item
    item2=SelectField("因变量",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=Selected_1_default_item)
    submit2=SubmitField("提交")

class Selected_2(Form):
    global Selected_2_default_item
    item3=SelectField("自变量",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=Selected_2_default_item)
    submit3=SubmitField("提交")
class MultipleSelect(Form):
    item_multiple=SelectMultipleField("自变量",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=0)
    submit_multiple=SubmitField('提交')    
    
class NeuralNet(Form):
    layers_num=StringField('网络层数')
    neuron_num=StringField('隐层神经元数')
    error_goal=StringField('期望误差')
    epochs=StringField('最大迭代次数')
    submit_Neural=SubmitField('计算',render_kw={"style":"class:btn btn-primary"})

class RandomForest(Form):
    n_estimators=StringField('树的数目')
    max_features=StringField('最大选取特征数')
    max_depth=StringField('树的最大深度')
    min_samples_split=StringField('节点最小分裂数')
    min_samples_leaf=StringField('叶子节点上的最小样本数')
    submit_RF=SubmitField('计算',render_kw={"style":"class:btn btn-primary"})

def test_size_check(form,field):
        if field.data>=1 or field.data<0:
            raise ValidationError('请输入0-1之间的小数')
            
class Train_Test(Form):
    test_size=FloatField('测试集比例',[DataRequired(),test_size_check],render_kw={\
                                "placeholder":"一般选取0.2-0.4之间"})
    random_state=StringField('随机数种子',render_kw={\
                                "placeholder":"0"},default='0')
    submit_Train_Test=SubmitField('提交')
    
    
    
class Get_Data():
    def get_data(self,table_name):
         global data
         sql_columns_name=str("desc %s" %table_name)
         db=pymysql.connect("localhost","zy","Zy2225786!","test1")
         cursor=db.cursor()
         cursor.execute(sql_columns_name)
         columns_name =list(cursor.fetchall())
         sql_data_count=str("select count(*) from %s" %table_name)
         cursor.execute(sql_data_count)
         data_count=list(cursor.fetchall())
         
         sql_data=str("select * from %s" %table_name)
         cursor.execute(sql_data)
         data = pd.DataFrame(np.array(cursor.fetchall()))
         data.columns=[np.array(columns_name)[:,0]]
         db.close()
         return columns_name,data_count,data

#class Sclar

@app.route('/login', methods=['POST','GET'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username']=='admin' and request.form['password']=='123456':
            return redirect(url_for('home',username=request.form['username']))
        else:
            error = 'Invalid username/password'
    return render_template('login.html', error=error)

@app.route('/table', methods=['POST','GET'])
def table():
#    return resp      
    return render_template('table.html')

@app.route('/table2', methods=['POST','GET'])
def table2():
    result='56fg'
    result_json=json.dumps(result)
    resp = Response(result_json)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp      
#    return render_template('table.html')

@app.route('/testRF', methods=['POST','GET'])
def testRF():
    form=Submit()
    if form.validate_on_submit():
#        session['test']=form.s_in_Q.data+form.s_in_T.data
#        p=form.s_in_Q.data+form.s_in_T.data
        x1=float(form.s_in_Q.data)
        x2=float(form.s_in_T.data)
        x3=float(form.s_in_P.data)
        x4=float(form.s_out_Q.data)
        x_predict=np.c_[x1,x2,x3,x4]
        rf=joblib.load("train_model_RF.m")
        y_predict=rf.predict(x_predict)
        result=json.dumps(list(y_predict))
        session['result']=result
#        flash(form.s_in_Q.data+'|'+form.s_in_T.data)
#        return redirect(url_for('testRF'))
    return render_template('testRF.html', form=form,result=session.get('result'))

@app.route('/preprocess',methods=['post','get'])
def preprocess():
    form1=Get_Data_Submit()
    if form1.validate_on_submit():
        global data_num
        table_name="table1"
        GD=Get_Data()
        columns,data_count,data=GD.get_data(table_name)
        for i in range(len(columns)):
            session[('columns_name'+str(i))]=columns[i][0]
            session[('columns_type'+str(i))]=columns[i][1]
            session[('data_count'+str(i))]=data_count[0][0]

        
        data_num=data_count[0][0]
    form2=Samping_Submit()
    if form2.validate_on_submit(): 
        if form2.sampling_ratio.data!="":
            sampling_ratio=float(form2.sampling_ratio.data)
            form2.sampling_count.data=int(sampling_ratio*data_num)
        if form2.sampling_count.data!="":
            sampling_ratio=int(form2.sampling_count.data)/data_num
            form2.sampling_ratio.data=str(sampling_ratio)
            s_k=random.sample(range(data_num),int(form2.sampling_count.data))
            global samples
            samples=data.iloc[s_k,:]

        
    form3=Missing_Process_Submit()
#    if form3.validate_on_submit():
#        form4=Hist_Graph()
#        return render_template('statistics.html',form2=form4)

    return render_template('preprocess.html',form1=form1, form2=form2,form3=form3,\
                           columns_name0=session.get('columns_name0'),columns_name1=session.get('columns_name1'),\
                           columns_name2=session.get('columns_name2'),columns_name3=session.get('columns_name3'),\
                           columns_name4=session.get('columns_name4'),columns_name5=session.get('columns_name5'),\
                           columns_name6=session.get('columns_name6'),columns_type0=session.get('columns_type0'),\
                           columns_type1=session.get('columns_type1'),\
                           columns_type2=session.get('columns_type2'),columns_type3=session.get('columns_type3'),\
                           columns_type4=session.get('columns_type4'),columns_type5=session.get('columns_type5'),\
                           columns_type6=session.get('columns_type6'),\
                           data_count0=session.get('data_count0'),data_count1=session.get('data_count1'),\
                           data_count2=session.get('data_count2'),data_count3=session.get('data_count3'),\
                           data_count4=session.get('data_count4'),data_count5=session.get('data_count5'),\
                           data_count6=session.get('data_count6'))
    

@app.route('/statistics',methods=['post','get'])
   
def statistics():
    global choices
    global Selected_2_default_item
    global x
    global y
    form2=Hist_Graph()
    choices=list(zip(range(len(samples.columns)),samples.columns))
    form2.item.choices=choices

    if form2.submit1.data and form2.validate_on_submit():
        session['select_item']=str(samples.columns[form2.item.data])
        global hist_selected_column
        hist_selected_column=str(samples.columns[form2.item.data])
#        return render_template('statistics.html',form2=form2,select_item=session.get('select_item'))

    form3=Selected_1()
    form3.item2.choices=choices    
    
    form4=Selected_2()
    form4.item3.choices=choices 
    
    if form3.submit2.data and form3.validate_on_submit():
#        Selected_2_default_item=3
        session['Selected_X']=str(samples.columns[form3.item2.data])
        x=samples[str(samples.columns[form3.item2.data])]
#        print(x)
        
    if form4.submit3.data and form4.validate_on_submit():
#        Selected_1_default_item=4
        session['Selected_Y']=str(samples.columns[form4.item3.data])
        y=samples[str(samples.columns[form4.item3.data])]
    
    S=Statistics()
    
    form_pearson=Pearson_cal_Submit()
    if form_pearson.submit_pearson.data and form_pearson.validate_on_submit():
        spearman_R,spearman_p=S.spearman(x,y)
        pearson_R,pearson_p=S.pearson(x,y)
        session['pearson']=str(pearson_R)
    
    form1=Describe_Statistics_Submit()
    if form1.submit.data and form1.validate_on_submit():
        count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness=S.descriptive_statistics(samples)
#        Desc=pd.DataFrame([count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness])
        
        #特别注意游标cursorclass类型，如果想以字典的形式返回，请加后边这条语句，帮助非常大，查到想打人才找到原因
        db=pymysql.connect("localhost","zy","Zy2225786!","test1",cursorclass=pymysql.cursors.DictCursor)
#        sql_string=str("update desc set count=%f" %count)\+str(" ,mean=%f" %mean)+str(" ,std=%f" %std)\
#                    +str(" ,mn=%f" %mn)+str(" ,mx=%f" %mx)+str(" ,per25=%f" %per25)+str(" ,per50=%f" %per50)\
#                    +str(" ,per75=%f" %per75)+str(" ,cv=%f" %cv)+str(" ,kurtosis=%f" %kurtosis)\
#                    +str(" ,skewness=%f" %skewness)
#        sql_string=str("update desc set count=%f" %count)
        cursor=db.cursor()
        
#        columns_name =list(cursor.fetchall())
        for i in range(len(samples.columns)):
#            session[('columns'+str(i))]=samples.columns[i]
#            session['s1_'+str(i)]=count[i]
#            session['s2_'+str(i)]=mean[i]
#            session['s3_'+str(i)]=std[i]
#            session['s4_'+str(i)]=mn[i]
#            session['s5_'+str(i)]=mx[i]
#            session['s6_'+str(i)]=per25[i]
#            session['s7_'+str(i)]=per50[i]
#            session['s8_'+str(i)]=per75[i]
#            session['s9_'+str(i)]=cv[i]
#            session['s10_'+str(i)]=kurtosis[i]
#            session['s11_'+str(i)]=skewness[i]
            sql_string=str("update test1.desc set count=%f" %count[i])+str(",mean=%f" %mean[i])+str(",std=%f" %std[i])\
                    +str(",mn=%f" %mn[i])+str(",mx=%f" %mx[i])+str(",per25=%f" %per25[i])+str(",per50=%f" %per50[i])\
                    +str(",per75=%f" %per75[i])+str(",cv=%f" %cv[i])+str(",kurtosis=%f" %kurtosis[i])\
                    +str(",skewness=%f" %skewness[i])+str(" where feature='%s'" %samples.columns[i])  
                    
#            sql_string=str("UPDATE test1.desc SET count=%f" %count[i])+str(" WHERE feature='%s'" %samples.columns[i]) 
            cursor.execute(sql_string)
            db.commit()
        sql_string=str("select * from test1.desc")
        cursor.execute(sql_string)
        desc=cursor.fetchall()
        session['desc']=desc   
     
#    return render_template('statistics.html',form1=form1,form2=form2,form3=form3,form4=form4,form_pearson=form_pearson,\
#                           columns_name0=session.get('columns_name0'),columns_name1=session.get('columns_name1'),\
#                           columns_name2=session.get('columns_name2'),columns_name3=session.get('columns_name3'),\
#                           columns_name4=session.get('columns_name4'),\
#                           s1_0=session.get('s1_0'),s1_1=session.get('s1_1'),s1_2=session.get('s1_2'),s1_3=session.get('s1_3'),s1_4=session.get('s1_4'),\
#                           s2_0=session.get('s2_0'),s2_1=session.get('s2_1'),s2_2=session.get('s2_2'),s2_3=session.get('s2_3'),s2_4=session.get('s2_4'),\
#                           s3_0=session.get('s3_0'),s3_1=session.get('s3_1'),s3_2=session.get('s3_2'),s3_3=session.get('s3_3'),s3_4=session.get('s3_4'),\
#                           s4_0=session.get('s4_0'),s4_1=session.get('s4_1'),s4_2=session.get('s4_2'),s4_3=session.get('s4_3'),s4_4=session.get('s4_4'),\
#                           s5_0=session.get('s5_0'),s5_1=session.get('s5_1'),s5_2=session.get('s5_2'),s5_3=session.get('s5_3'),s5_4=session.get('s5_4'),\
#                           s6_0=session.get('s6_0'),s6_1=session.get('s6_1'),s6_2=session.get('s6_2'),s6_3=session.get('s6_3'),s6_4=session.get('s6_4'),\
#                           s7_0=session.get('s7_0'),s7_1=session.get('s7_1'),s7_2=session.get('s7_2'),s7_3=session.get('s7_3'),s7_4=session.get('s7_4'),\
#                           s8_0=session.get('s8_0'),s8_1=session.get('s8_1'),s8_2=session.get('s8_2'),s8_3=session.get('s8_3'),s8_4=session.get('s8_4'),\
#                           s9_0=session.get('s9_0'),s9_1=session.get('s9_1'),s9_2=session.get('s9_2'),s9_3=session.get('s9_3'),s9_4=session.get('s9_4'),\
#                           s10_0=session.get('s10_0'),s10_1=session.get('s10_1'),s10_2=session.get('s10_2'),s10_3=session.get('s10_3'),s10_4=session.get('s10_4'),\
#                           s11_0=session.get('s11_0'),s11_1=session.get('s11_1'),s11_2=session.get('s11_2'),s11_3=session.get('s11_3'),s11_4=session.get('s11_4'),\
#                           select_item=session.get('select_item'),Selected_X=session.get('Selected_X'),Selected_Y=session.get('Selected_Y'),\
#                           pearson=session.get('pearson'))
    
    return render_template('statistics.html',form1=form1,form2=form2,form3=form3,form4=form4,form_pearson=form_pearson,\
                           desc=session.get('desc'),\
                           select_item=session.get('select_item'),Selected_X=session.get('Selected_X'),Selected_Y=session.get('Selected_Y'),\
                           pearson=session.get('pearson'))


@app.route('/regression',methods=['post','get'])
def regression():
    global choices
    global x_linear_regression
    global y_linear_regression
    test_size=0
    random_state=0
    global X_train
    global X_test
    global y_train
    global y_test
    global fea_import
    global input_train_scaler_transform
    global output_train_scaler_transform
    form3=Selected_1()
#    form3.item2.description='因变量ww'
    form3.item2.choices=choices    
    
    form_multipleSelect=MultipleSelect()
    form_multipleSelect.item_multiple.choices=choices
    
    if form3.submit2.data and form3.validate_on_submit():
        session['single_selected']=str(samples.columns[form3.item2.data])
        y_linear_regression=samples[str(samples.columns[form3.item2.data])]
        line_reg=linear_model.LinearRegression()
        line_reg.fit(x_linear_regression,y_linear_regression)
        a,b,r=line_reg.coef_,line_reg.intercept_,line_reg._residues
        R=1-r/sum((y_linear_regression-y_linear_regression.mean())**2)
        session['a']=str(a)
        session['b']=str('%.3f' %b)
        session['r']=str(r)
        session['R']=str(R)

    if form_multipleSelect.submit_multiple.data and form_multipleSelect.validate_on_submit():
        session['multiple_selected']=str(list(samples.columns[form_multipleSelect.item_multiple.data]))
        x_linear_regression=samples[samples.columns[form_multipleSelect.item_multiple.data]]
    
    form_Train_Test=Train_Test()
#    form_Train_Test.test_size.data='0.3'
    form_Train_Test.random_state.description="如果填入整数则为固定随机数种子"
    if form_Train_Test.submit_Train_Test.data and form_Train_Test.validate_on_submit():
        test_size=float(form_Train_Test.test_size.data)
        random_state=int(form_Train_Test.random_state.data)
        X_train,X_test,y_train,y_test = train_test_split(x_linear_regression, \
                                                            y_linear_regression, test_size=test_size, random_state=random_state)
        session['train_sample_num']=str(len(X_train))
        session['test_sample_num']=str(len(X_test))
    
    form_neural=NeuralNet()
    form_neural.layers_num.data="2"
    form_neural.neuron_num.data="[3,1]"
    form_neural.neuron_num.description="请在[]内修改，保留[]"
    if form_neural.submit_Neural.data and form_neural.validate_on_submit():
        start_time=time.clock()

        
        Input=X_train
        Target=y_train.values.reshape(-1,1)
        
        input_train_minmax_scaler=preprocessing.MinMaxScaler().fit(Input)
        input_train_scaler_transform=input_train_minmax_scaler.transform(Input)
        
        output_train_minmax_scaler=preprocessing.MinMaxScaler().fit(Target)
        output_train_scaler_transform=output_train_minmax_scaler.transform(Target)
        
        minmax=[]
        for _ in range(len(Input.columns)):
            minmax.append([0,1])
#        minmax=list(zip(Input.min(),Input.max()))
#        print(minmax)
#        print(Target)
        net_layer=[3,1]
        net=nl.net.newff(minmax,net_layer)
        train=net.train(input_train_scaler_transform,output_train_scaler_transform,epochs=500,show=100,goal=0.01)
        out=output_train_minmax_scaler.inverse_transform(net.sim(input_train_scaler_transform))
#        SSEf=error.SSE()
#        SSE=SSEf(Target,out)
        SSE=sum((Target-out)**2)
        R_neuro=1-(SSE/sum((Target-Target.mean())**2))
        end_time=time.clock()
        time_span=str("%.3f" %(end_time-start_time))
        session['time_span']=time_span
        session['SSE']=str('%.3f' %SSE)
        session['R_neuro']=str('%.4f' %R_neuro)

        
    form_RF=RandomForest()
    if form_RF.submit_RF.data and form_RF.validate_on_submit():
        start_time=time.clock()
      
        rf=RandomForestRegressor()
        GBDT=GradientBoostingRegressor()
        rf.fit(X_train,y_train)
        names=X_train.columns
        feature_importances="特征重要性排列如下：\n"+str(sorted(zip(map(lambda x: \
                                                     str('%.5f' %x),rf.feature_importances_),names),reverse=True))
        fea_import=rf.feature_importances_
        
        GBDT.fit(X_train,y_train)

        R_RF_train=rf.score(X_train,y_train)
        R_RF_test=rf.score(X_test,y_test)
        R_GBDT_train=GBDT.score(X_train,y_train)
        R_GBDT_test=GBDT.score(X_test,y_test)
        end_time=time.clock()
        time_span_RF=str("%.3f" %(end_time-start_time))
        session['R_RF_train']=str('%.4f' %R_RF_train)
        session['time_span_RF']=time_span_RF
        session['R_GBDT_train']=str('%.4f' %R_GBDT_train)
        session['R_RF_test']=str('%.4f' %R_RF_test)
        session['R_GBDT_test']=str('%.4f' %R_GBDT_test)
        session['feature_importances']=feature_importances

    
    return render_template('regression.html',form3=form3,form_multipleSelect=form_multipleSelect,\
                           form_neural=form_neural,form_RF=form_RF,form_Train_Test=form_Train_Test,\
                           multiple_selected=session.get('multiple_selected'),\
                           single_selected=session.get('single_selected'),\
                           train_sample_num=session.get('train_sample_num'),test_sample_num=session.get('test_sample_num'),\
                           a=session.get('a'),b=session.get('b'),r=session.get('r'),R=session.get('R'),\
                           time_span=session.get('time_span'),SSE=session.get('SSE'),R_neuro=session.get('R_neuro'),\
                           R_RF_train=session.get('R_RF_train'),time_span_RF=session.get('time_span_RF'),\
                           R_GBDT_train=session.get('R_GBDT_train'),R_RF_test=session.get('R_RF_test'),\
                           R_GBDT_test=session.get('R_GBDT_test'),feature_importances=session.get('feature_importances'))

@app.route('/')
def index():
    return render_template('index.html')    

@app.route('/feature')
def feature():
    return render_template('feature.html')   

#@app.route('/3D')
#def hello():
#    s3d = scatter3d()
##    return render_template('pyecharts.html',
##                           myechart=s3d.render_embed(),
##                           host='C:\\Users\\zy\\Downloads\\jupyter-echarts-master\\jupyter-echarts-master\\echarts',
##                           script_list=s3d.get_js_dependencies())
##    return render_template('pyecharts.html',
##                           myechart=s3d.render_embed(),
##                           host=DEFAULT_HOST,
##                           script_list=s3d.get_js_dependencies())
#
#    return render_template('pyecharts.html',
#                           myechart=s3d.render_embed(),
#                           host='http://chfw.github.io/jupyter-echarts/echarts',
#                           script_list=s3d.get_js_dependencies())
#
#def scatter3d():
#    data = [generate_3d_random_point() for _ in range(80)]
#    range_color = [
#        '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
#        '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
#    scatter3D = Scatter3D("3D scattering plot demo", width=1200, height=600)
#    scatter3D.add("", data, is_visualmap=True, visual_range_color=range_color)
#    return scatter3D


#def generate_3d_random_point():
#    return [random.randint(0, 100),
#            random.randint(0, 100),
#            random.randint(0, 100)]

@app.route('/bar')
#def sss():
#    b=bar_render()
#    return render_template('pyecharts.html',
#                            myechart=b.render_embed(),
#                            host='http://chfw.github.io/jupyter-echarts/echarts',
#                            script_list=b.get_js_dependencies())

def bar():
    kk=bar_render()
    return render_template('bar.html')

def bar_render():
    attr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    v1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
    v2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
    bar = Bar("Bar chart", "precipitation and evaporation one year")
    bar.add("precipitation", attr, v1, mark_line=["average"], mark_point=["max", "min"])
    bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
    bar.render('C:/Users/zy/python_practice/flask/templates/bar.html')
    
#    return bar

@app.route('/hist')
def hist():
    hist_g()
    return render_template('hist.html')
def hist_g():
    a=samples[hist_selected_column]
    hist,bin_edges=np.histogram(a,bins='auto',density=False)
    attr=list(bin_edges[0:-1].round(1))
    v1=list(hist)
    result = [float(item) for item in v1]
#    v1=[23,44,21,34,56,77,88,32,33,44]
    bar=Bar("频率分布直方图")
    bar.add(hist_selected_column,attr,result)
    bar.render('C:/Users/zy/python_practice/flask/templates/hist.html')

@app.route('/pie')
def pie():
    pie_g()
    return render_template('pie.html')
def pie_g():
    attr=list(X_train.columns)
    v1=list(fea_import)
    pie=Pie("特征重要性")
    pie.add("",attr,v1,is_label_show=True)
    pie.render('C:/Users/zy/python_practice/flask/templates/pie.html')

@app.route('/box')
def box():
    box_g()
    return render_template('box.html')
def box_g():
    boxplot=Boxplot("箱形图")
    x_axis=list(X_train.columns)
    y_axis=input_train_scaler_transform.T
    print(y_axis)
    _yaxis=boxplot.prepare_data(y_axis)       # 转换数据
    boxplot.add("boxplot", x_axis, _yaxis)
    boxplot.render('C:/Users/zy/python_practice/flask/templates/box.html')


@app.route('/tasks',methods=['GET'])
def get_task():
    return jsonify({'tasks':tasks})

@app.route('/tasks/<int:task_id>',methods=['GET'])
def get_tasks(task_id):
    task=list(filter(lambda t:t['id']==task_id,tasks))
    if len(task)==0:
        abort(404)
    return jsonify({'tasks':task[0]})
    
@app.route('/tasks',methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(404)
    task={
            'id':tasks[-1]['id']+1,
            'title':request.json['title'],
            'description':request.json.get('description',"")
            }
    tasks.append(task)
    return jsonify({'task':task}),201

@app.route('/train',methods=['POST'])
def RandForestRegress():
    
    columns_str_x_train=request.json['columns_str_x_train']
    sql_x_train=str("select %s from table1 " %columns_str_x_train)
    columns_str_y_train=request.json['columns_str_y_train']
    sql_y_train=str("select %s from table1" %columns_str_y_train)
    
    db=pymysql.connect("localhost","zy","Zy2225786!","test1")
    cursor=db.cursor()
    cursor.execute(sql_x_train)
    x_train = pd.DataFrame(np.array(cursor.fetchall()))
    
    cursor.execute(sql_y_train)
    y_train=[]
    temp=cursor.fetchall()
    for (row,) in temp:y_train.append(row) #妈的鸡，查询出来带(,),这样一搞就没了，好神奇
    db.close()
    rf=RandomForestRegressor()
    rf.fit(x_train,y_train)
    x_predict=x_train.copy()
    y_predict=rf.predict(x_predict)
    mse=sum((y_train-y_predict)**2)
#    print(mse)
#直接**2会报错：unsupported operand type(s) for ** or pow(): 'list' and 'int'
    R=1-math.sqrt(mse/sum(np.array(y_train)**2)) 
    joblib.dump(rf, "train_model_RF.m")
    result_txt=list(map(lambda x:str('%.5f' %x),rf.feature_importances_))
    result={
            'R':R,
            'feature_importance':result_txt
            }
    return jsonify(result),201
        
        
        

@app.route('/test',methods=['POST'])
def test():
    if not request.json or not 'x1' in request.json:
        abort(404)
#    c=int(request.json['a'])+int(request.json['b'])
    x1=float(request.json['x1'])
    x2=float(request.json['x2'])
    x3=float(request.json['x3'])
    x4=float(request.json['x4'])
    x_predict=np.c_[x1,x2,x3,x4]
    rf=joblib.load("train_model_RF.m")
    y_predict=rf.predict(x_predict)
    result=json.dumps(list(y_predict))
    result={
            'result':result
            }
    return jsonify({'result':result}),201

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error':'Not Found'}),404


app.config['SECRET_KEY']='xxx'
if __name__ == '__main__': 
    app.run()


