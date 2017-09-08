# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:22:34 2017

@author: zy
"""
import random
import scipy.stats as sts


from flask import Flask, jsonify,render_template,url_for,redirect,session
from flask import request
from flask import json
from flask import abort
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm as Form

#form wtforms import TextField
#from flask_wtf import TextField, StringField,SubmitField,TextAreaField,IntegerField
#from wtforms.validators import Required 
#from flask_wtf import TextField,StringField,SubmitField,TextAreaField,IntegerField
from wtforms import TextField, StringField,SubmitField,TextAreaField,IntegerField,BooleanField,RadioField,SelectField
from wtforms import fields
from wtforms.validators import DataRequired
from wtforms import validators

from sklearn.externals import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
import pymysql

import random
from pyecharts import Scatter3D
from pyecharts.constants import DEFAULT_HOST
from pyecharts import Bar

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
    
class Hist_Graph(Form):
    item=SelectField("选择列表",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=0)
    submit=SubmitField("提交")    
    
    
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
         return columns_name,data_count,data

@app.route('/login', methods=['POST','GET'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username']=='admin' and request.form['password']=='123456':
            return redirect(url_for('home',username=request.form['username']))
        else:
            error = 'Invalid username/password'
    return render_template('login.html', error=error)

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
        columns_str_x_train="table1"
        GD=Get_Data()
        columns,data_count,data=GD.get_data(columns_str_x_train)
        session['columns_name1']=columns[0][0]
        session['columns_name2']=columns[1][0]
        session['columns_name3']=columns[2][0]
        session['columns_name4']=columns[3][0]
        session['columns_name5']=columns[4][0]
        session['columns_type1']=columns[0][1]
        session['columns_type2']=columns[1][1]
        session['columns_type3']=columns[2][1]
        session['columns_type4']=columns[3][1]
        session['columns_type5']=columns[4][1]
        session['data_count']=data_count[0][0]
        
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
#            print(samples)
        global item
        item=SelectField("选择列表",[DataRequired()],choices=[(0,str(samples.columns[0])),\
                                 (1,str(samples.columns[1])),(2,str(samples.columns[2])),(3,str(samples.columns[3])),\
                                                (4,str(samples.columns[4]))],default=1)
#        print(samples.columns)
#        print(item)
        
    form3=Missing_Process_Submit()
    if form3.validate_on_submit():
        form4=Hist_Graph()
        return render_template('statistics.html',form2=form4)
    return render_template('preprocess.html',form1=form1, form2=form2,form3=form3,\
                           columns_name1=session.get('columns_name1'),columns_name2=session.get('columns_name2'),\
                           columns_name3=session.get('columns_name3'),columns_name4=session.get('columns_name4'),\
                           columns_name5=session.get('columns_name5'),columns_type1=session.get('columns_type1'),\
                           columns_type2=session.get('columns_type2'),columns_type3=session.get('columns_type3'),\
                           columns_type4=session.get('columns_type4'),columns_type5=session.get('columns_type5'),\
                           data_count=session.get('data_count'))
    

@app.route('/statistics',methods=['post','get'])
   
def statistics():
    form2=Hist_Graph()
#    form2.item.choices=[(0,'a'),(1,'b'),(2,'c'),(3,'d'), (4,'e')]
    form2.item.choices=[(0,str(samples.columns[0])),(1,str(samples.columns[1])),\
                        (2,str(samples.columns[2])),(3,str(samples.columns[3])),\
                                                (4,str(samples.columns[4]))]
    if form2.validate_on_submit():
        session['select_item']=str(samples.columns[form2.item.data])
#        return render_template('statistics.html',form2=form2,select_item=session.get('select_item'))
    
    form1=Describe_Statistics_Submit()
    if form1.validate_on_submit():
        S=Statistics()
        count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness=S.descriptive_statistics(samples)
        session['columns_name1']=samples.columns[0]
        session['columns_name2']=samples.columns[1]
        session['columns_name3']=samples.columns[2]
        session['columns_name4']=samples.columns[3]
        session['columns_name5']=samples.columns[4]
        
        session['s1_1']=count[0]
        session['s1_2']=count[1]
        session['s1_3']=count[2]
        session['s1_4']=count[3]
        session['s1_5']=count[4]
        
        session['s2_1']=mean[0]
        session['s2_2']=mean[1]
        session['s2_3']=mean[2]
        session['s2_4']=mean[3]
        session['s2_5']=mean[4]
        
        session['s3_1']=std[0]
        session['s3_2']=std[1]
        session['s3_3']=std[2]
        session['s3_4']=std[3]
        session['s3_5']=std[4]
        
        session['s4_1']=mn[0]
        session['s4_2']=mn[1]
        session['s4_3']=mn[2]
        session['s4_4']=mn[3]
        session['s4_5']=mn[4]
        
        session['s5_1']=mx[0]
        session['s5_2']=mx[1]
        session['s5_3']=mx[2]
        session['s5_4']=mx[3]
        session['s5_5']=mx[4]
        
        session['s6_1']=per25[0]
        session['s6_2']=per25[1]
        session['s6_3']=per25[2]
        session['s6_4']=per25[3]
        session['s6_5']=per25[4]
        
        session['s7_1']=per50[0]
        session['s7_2']=per50[1]
        session['s7_3']=per50[2]
        session['s7_4']=per50[3]
        session['s7_5']=per50[4]
        
        session['s8_1']=per75[0]
        session['s8_2']=per75[1]
        session['s8_3']=per75[2]
        session['s8_4']=per75[3]
        session['s8_5']=per75[4]
        
        session['s9_1']=cv[0]
        session['s9_2']=cv[1]
        session['s9_3']=cv[2]
        session['s9_4']=cv[3]
        session['s9_5']=cv[4]
        
        session['s10_1']=kurtosis[0]
        session['s10_2']=kurtosis[1]
        session['s10_3']=kurtosis[2]
        session['s10_4']=kurtosis[3]
        session['s10_5']=kurtosis[4]
        
        session['s11_1']=skewness[0]
        session['s11_2']=skewness[1]
        session['s11_3']=skewness[2]
        session['s11_4']=skewness[3]
        session['s11_5']=skewness[4]
        
        
    return render_template('statistics.html',form1=form1,form2=form2,\
                           columns_name1=session.get('columns_name1'),columns_name2=session.get('columns_name2'),\
                           columns_name3=session.get('columns_name3'),columns_name4=session.get('columns_name4'),\
                           columns_name5=session.get('columns_name5'),\
                           s1_1=session.get('s1_1'),s1_2=session.get('s1_2'),s1_3=session.get('s1_3'),s1_4=session.get('s1_4'),s1_5=session.get('s1_5'),\
                           s2_1=session.get('s2_1'),s2_2=session.get('s2_2'),s2_3=session.get('s2_3'),s2_4=session.get('s2_4'),s2_5=session.get('s2_5'),\
                           s3_1=session.get('s3_1'),s3_2=session.get('s3_2'),s3_3=session.get('s3_3'),s3_4=session.get('s3_4'),s3_5=session.get('s3_5'),\
                           s4_1=session.get('s4_1'),s4_2=session.get('s4_2'),s4_3=session.get('s4_3'),s4_4=session.get('s4_4'),s4_5=session.get('s4_5'),\
                           s5_1=session.get('s5_1'),s5_2=session.get('s5_2'),s5_3=session.get('s5_3'),s5_4=session.get('s5_4'),s5_5=session.get('s5_5'),\
                           s6_1=session.get('s6_1'),s6_2=session.get('s6_2'),s6_3=session.get('s6_3'),s6_4=session.get('s6_4'),s6_5=session.get('s6_5'),\
                           s7_1=session.get('s7_1'),s7_2=session.get('s7_2'),s7_3=session.get('s7_3'),s7_4=session.get('s7_4'),s7_5=session.get('s7_5'),\
                           s8_1=session.get('s8_1'),s8_2=session.get('s8_2'),s8_3=session.get('s8_3'),s8_4=session.get('s8_4'),s8_5=session.get('s8_5'),\
                           s9_1=session.get('s9_1'),s9_2=session.get('s9_2'),s9_3=session.get('s9_3'),s9_4=session.get('s9_4'),s9_5=session.get('s9_5'),\
                           s10_1=session.get('s10_1'),s10_2=session.get('s10_2'),s10_3=session.get('s10_3'),s10_4=session.get('s10_4'),s10_5=session.get('s10_5'),\
                           s11_1=session.get('s11_1'),s11_2=session.get('s11_2'),s11_3=session.get('s11_3'),s11_4=session.get('s11_4'),s11_5=session.get('s11_5'),\
                           select_item=session.get('select_item'))
    

@app.route('/regression',methods=['post','get'])
def regression():
    return render_template('regression.html')

@app.route('/')
def index():
    return render_template('index.html')    

@app.route('/3D')
def hello():
    s3d = scatter3d()
#    return render_template('pyecharts.html',
#                           myechart=s3d.render_embed(),
#                           host='C:\\Users\\zy\\Downloads\\jupyter-echarts-master\\jupyter-echarts-master\\echarts',
#                           script_list=s3d.get_js_dependencies())
#    return render_template('pyecharts.html',
#                           myechart=s3d.render_embed(),
#                           host=DEFAULT_HOST,
#                           script_list=s3d.get_js_dependencies())

    return render_template('pyecharts.html',
                           myechart=s3d.render_embed(),
                           host='http://chfw.github.io/jupyter-echarts/echarts',
                           script_list=s3d.get_js_dependencies())

def scatter3d():
    data = [generate_3d_random_point() for _ in range(80)]
    range_color = [
        '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
        '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
    scatter3D = Scatter3D("3D scattering plot demo", width=1200, height=600)
    scatter3D.add("", data, is_visualmap=True, visual_range_color=range_color)
    return scatter3D


def generate_3d_random_point():
    return [random.randint(0, 100),
            random.randint(0, 100),
            random.randint(0, 100)]

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
    h=hist_g()
    return render_template('hist.html')
def hist_g():
    a=samples['Steam_in_Q']
    hist,bin_edges=np.histogram(a,bins='auto',density=True)
    attr=list(bin_edges[0:-1].round(1))
    v1=list(hist)
#    v1=[23,44,21,34,56,77,88,32,33,44]
    bar=Bar("柱状图数据堆叠示例")
    bar.add("商家A",attr,v1)
    bar.render('C:/Users/zy/python_practice/flask/templates/hist.html')
    print(v1)
    print(attr)



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


