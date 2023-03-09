#coding=gbk
'''
@author: Xin Huang
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import copy
from math import sqrt

import csv
import sys
import os
import pdb

#——————————————————train——————————————————
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=5800):
    
    batch_index=[]
    data_train=whole_data[train_begin:train_end]   
    result_train=result_data[train_begin:train_end]  
    
    train_x,train_y=[],[]   
    for i in range(len(data_train)-time_step):
       
       if i % batch_size==0:
           batch_index.append(i)
       x=data_train[i:i+time_step,:]
       y=result_train[i:i+time_step,:]     
       train_x.append(x.tolist())
       train_y.append(y.tolist())
       
    batch_index.append((len(data_train)-time_step))
    return batch_index,train_x,train_y


#-------------------------------------test-------------------------------------------
def get_test_data(time_step=20,test_begin=5800):
    
    data_test=whole_data[(test_begin-time_step):]   
    result_test=result_data[(test_begin-time_step):]  
    test_x,test_y=[],[]
   
    for i in range(len(data_test)-time_step+1):   
        x=data_test[i:i+time_step,:]
        y=result_test[i:i+time_step,:]
        test_x.append(x.tolist())
        test_y.append(y.tolist()) 
    
    return test_x,test_y

#——————————————————lstm——————————————————
def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  
    
    if is_training and keep_prob < 1:
        input_rnn = tf.nn.dropout(input_rnn, keep_prob)
    
    basic_cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit,forget_bias=0.0,state_is_tuple=True,activation=tf.nn.softsign)       
    
    if is_training and keep_prob < 1:
        basic_cell = tf.nn.rnn_cell.DropoutWrapper(basic_cell,output_keep_prob=keep_prob)

    cell=tf.nn.rnn_cell.MultiRNNCell([basic_cell] * layer_num,state_is_tuple=True)       

    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#——————————————————train——————————————————
def train_lstm(train_end,batch_size=80,time_step=15,train_begin=0):
    
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    
    is_training=True
    
    with tf.variable_scope("sec_lstm"):  
        pred,_=lstm(X)     
        
    tv = tf.trainable_variables() 
    regularization_cost = 0.001* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) 
    
    original_cost_function=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    
    loss = original_cost_function + regularization_cost
    
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    module_file = tf.train.latest_checkpoint(base_path)    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(100):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 5==0:
                print("保存模型：",saver.save(sess,'multi_factor_groundwater_model/groundwater.model',global_step=i))
                

#————————————————validation————————————————————
def validation(test_begin,time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    test_x,test_y=get_test_data(time_step,test_begin)    
    
    original_x=test_x
    original_y=test_y     
        
    is_training=False
    
    with tf.variable_scope("sec_lstm",reuse=True):  
         pred,_=lstm(X)
    
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        
        module_file = tf.train.latest_checkpoint(base_path)
        saver.restore(sess, module_file)
        
        test_predict=[]
        for i in range(len(original_x)):    
          prob=sess.run(pred,feed_dict={X:[original_x[i]]})            
          test_predict.append(prob[-1])
        
        temp_y=[]
        for w in range(len(original_y)):
            linshi_y=np.array(original_y[w])*std[int(input_size/jingnum)]+mean[int(input_size/jingnum)]
            temp_y.append(linshi_y.tolist())
        original_y=np.array(temp_y)
        
        test_predict=np.array(test_predict)*std[int(input_size/jingnum)]+mean[int(input_size/jingnum)]
        

#——————————————————Main——————————————————————
start=0
end=10

whole_data=[]   
data=[]              
final_factor_data=[]     
final_result_data=[]     
mean,std=[],[]  
jingnum=0     
borenum=[]   

for i in range(start, end):
    
    current_bore=i
    
    filename = 'dataset/' + str(current_bore) + '.csv'

    borenum.append(i) 
    jingnum=jingnum+1
    
    f=open(filename)
    df=pd.read_csv(f)     

    column=df.columns.tolist()
    input_size = len(column) - 3
    
    for u in range(len(df.iloc[:,2:len(column)])):
        whole_data.append(df.iloc[:,2:len(column)].values.tolist()[u])  

mean=np.mean(whole_data,axis=0)
std=np.std(whole_data,axis=0)
data=(whole_data-np.mean(whole_data,axis=0))/np.std(whole_data,axis=0)  

for t in range(len(df.iloc[:,2:len(column)])):
    
    temp_data=[]
    temp_num=[]
    temp_result=[]
    xiabiao=t
    
    while xiabiao < len(data):
        temp_data=data[xiabiao].tolist()
        temp_num.extend(temp_data[:len(temp_data)-1])
        temp_result.append(temp_data[-1])
        xiabiao=xiabiao+len(df.iloc[:,2:len(column)])

    final_factor_data.append(temp_num)
    final_result_data.append(temp_result)

whole_data=np.array(final_factor_data)
result_data=np.array(final_result_data)

time_step=8      
rnn_unit=128       
batch_size=24     
input_size=input_size*jingnum      
output_size=jingnum     
lr=0.01         
layer_num=3       

validation_rate=0.8   
train_data_num=0   
validation_data_num=0   

base_path='multi_factor_groundwater_model/'

train_data_num=int(len(df.iloc[:,2:len(column)])*validation_rate)    
validation_data_num=len(df.iloc[:,2:len(column)])-train_data_num     

X=tf.placeholder(tf.float32, [None,time_step,input_size])    
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   

keep_prob=0.75
is_training=True

initializer=tf.contrib.layers.xavier_initializer()
weights={
            'in':tf.Variable(initializer([input_size,rnn_unit])),   
            'out':tf.Variable(initializer([rnn_unit,output_size]))
        }
biases={
             'in':tf.Variable(initializer([rnn_unit,])),         
            'out':tf.Variable(initializer([output_size,]))
         }

test_x,test_y=[],[]

train_lstm(train_data_num,batch_size,time_step,0)

validation(time_step,time_step)

 
