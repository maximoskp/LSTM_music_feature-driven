#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:48:43 2018

@author: maximoskaliakatsos-papakostas

Simple Socket server

"""

import socket
import threading
import numpy as np
import tensorflow as tf

HOST = ''   # Symbolic name meaning all available interfaces
PORT = 8888 # Arbitrary non-privileged port

def makeSequencerMatrix(barLength, polyphony, intensity, pauses, notes_list):
    m = np.zeros((128,barLength))
    eimaste = 0
    for i in range(len(polyphony)):
        if polyphony[i] > 0:
            for j in range(int(polyphony[i])):
                m[ int(notes_list[eimaste]) , i] = intensity[i]
                eimaste += 1
    for i in range(barLength):
        if pauses[i] == -1:
            m[0, i] = -1
    return m
# end makeSequencerMatrix
def makeSequencerColumn(p):
    m = np.zeros((128))
    '''
    print("lowest_limit: ", lowest_limit)
    print("highest_limit: ", highest_limit)
    print("len(p): ", len(p))
    '''
    if len(p) > 0:
        tmpIDX = np.append(np.zeros(lowest_limit), p)
        tmpIDX = np.append(tmpIDX, np.zeros(128-highest_limit))
        m[ tmpIDX > 0 ] = 120
    return m
# end makeSequencerMatrix

# in order to get the client message as an array
def clientRequestToArray(r):
    print("r: ", r)
    s = str(r)
    x = s.split('_')
    if len(x) - 2 > 0:
        ra = np.zeros(len(x) - 2) # getting rid of the pre/sufix
        print('len(x): ' + str(len(x)))
        print('x: ', x)
        for i in range(len(x)-2):
            print('x[' + str(i) + ']: ' + x[i+1])
            ra[i] = float(x[i+1])
    else:
        ra = np.zeros(1)
        ra[0] = -1000.0
    return ra
def responseArrayToString(x):
    s = "*_"
    for i in range(len(x)):
        s += str(x[i])
        if i < len(x)-1:
            s += ","
    s += "_/"
    return s

# creating socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

msg = [];

s.bind((HOST, PORT))
print('Socket bind complete')

s.listen(10)
print('Socket now listening')

# retrieve saved data
d = np.load('saved_data/training_data.npz')
m = d['m']
f = d['f']
lowest_limit = d['lowest_limit']
highest_limit = d['highest_limit']
seed_init = d['seed']
# test batch generation example
max_len = 16
batch_size = 320
step = 1
input_rows = m.shape[0] + f.shape[0]
output_rows = m.shape[0]
num_units = [128, 256, 128]
learning_rate = 0.001
epochs = 5000
temperature = 0.5

def rnn(x, weight, bias, input_rows):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, input_rows])
    x = tf.split(x, max_len, 0)
    
    cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in num_units]
    stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
    outputs, states = tf.contrib.rnn.static_rnn(stacked_rnn_cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction
# end rnn

def sample(predicted_in):
    # keep only positive
    predicted = np.zeros(len(predicted_in))
    passes = np.where(predicted_in >= 0.0)[0]
    next_event = np.zeros( (len(predicted),1) )
    if len(passes) > 4:
        # get the 4 most possible events
        predicted[ passes ] = predicted_in[ passes ]
        passes = predicted.argsort()[-4:][::1]
    elif len(passes) > 0:
        passes = passes[0:np.min([4, len(passes)])]
    
    next_event[passes] = 1
    return next_event
# end sample

#Function for handling connections. This will be used to create threads
def clientthread(conn):

    # initialise befor sending connection message
    tf.reset_default_graph()

    x = tf.placeholder("float", [None, max_len, input_rows])
    y = tf.placeholder("float", [None, output_rows])
    weight = tf.Variable(tf.random_normal([num_units[-1], output_rows]))
    bias = tf.Variable(tf.random_normal([output_rows]))

    prediction = rnn(x, weight, bias, input_rows)
    dist = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(dist)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'saved_model/file.ckpt')
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)

    # GENERATE
    # generate seed
    seed = seed_init
    composition = np.array(seed[0,:,2:]).transpose()

    #Sending message to connected client
    conn.send('Welcome to the server. Type something and hit enter\n'.encode()) #send only takes string
    # dummy prediction
    predicted_output = []
    # predicted_output = np.append( [1], np.zeros(127) )
    #infinite loop so that function do not terminate and thread do not end.
    while True:
        #Receiving from client
        # reply = 'OK...' + str(data)
        data = conn.recv(1024)
        if not data: 
            break
        # divide respective features
        x_client = clientRequestToArray(data)
        # print('x_client: ' + str(x_client))
        if (x_client[0] != -1000.0):
            # update features
            # print('seed: ', seed)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            predicted_output = sample(predicted)
            remove_fist_event = seed[:,1:,:]
            new_input = np.append( x_client , predicted_output)
            seed = np.append(remove_fist_event, np.reshape(new_input, [1, 1, input_rows]), axis=1)

        # make final column
        m = makeSequencerColumn(predicted_output)
        
        y_out = responseArrayToString(m)
        # print('y_out: ' + y_out)
        reply = y_out
        conn.sendall(reply.encode())
        
    #came out of loop
    conn.close()

#now keep talking with the client
while 1:
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    threading.Thread(target=clientthread ,args=(conn,), kwargs={},).start()
    # clientthread(conn)

s.close()