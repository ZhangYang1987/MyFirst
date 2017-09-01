# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:22:34 2017

@author: zy
"""

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
from wtforms import TextField, StringField,SubmitField,TextAreaField,IntegerField
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
    submit=SubmitField('提交',render_kw={"style":"class:btn-primary"})

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
    sql_x_train=str("select %s from table1" %columns_str_x_train)
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


