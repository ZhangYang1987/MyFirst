# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:22:34 2017

@author: zy
"""

from flask import Flask, jsonify
from flask import request
from flask import json
from flask import abort

from sklearn.externals import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
import pymysql


app=Flask(__name__)

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

@app.route('/')
def index():
    return "Hello,World!"

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
    
    rf=RandomForestRegressor()
    rf.fit(x_train,y_train)
    x_predict=x_train.copy()
    y_predict=rf.predict(x_predict)
    mse=sum((y_train-y_predict)**2)
#    print(mse)
#直接**2会报错：unsupported operand type(s) for ** or pow(): 'list' and 'int'
    R=1-math.sqrt(mse/sum(np.array(y_train)**2)) 
    joblib.dump(rf, "train_model_RF.m")
    return jsonify({"R":R}),201
        
        
        

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
    rf=joblib.load("C:/Users/zy/python_practice/QT/zy20170626/zy20170626/train_model_RF.m")
    y_predict=rf.predict(x_predict)
    result=json.dumps(list(y_predict))
    result={
            'result':result
            }
    return jsonify({'result':result}),201

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error':'Not Found'}),404



if __name__ == '__main__': 
    app.run()


