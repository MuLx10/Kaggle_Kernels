'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import pandas as pd
import numpy as np
# import statistics as st
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn import preprocessing
# import seaborn as sns
import matplotlib as plt
import pylab
import re



import parse

train_set = parse.TrainingDataset('train.csv')
test_set = parse.TestDataset('test.csv')

X_index = [i for i in range(12)]
X_index = [2,4,6,7,9]
# X_index = [2,4,5,6,7,9]
Y_index = 1

train_set.read_data(X_index,Y_index)
train_set.std_data(X_index)
batch_x, batch_y = train_set.next_batch(train_set.SIZE)

train_set.reset_counter()
test_f,test_l = train_set.next_batch(100)
test_f,test_l = train_set.next_batch(100)









skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

parameters = {'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5,10,15,20], "min_samples_split": [6, 8, 10]}
rfc = RandomForestClassifier(n_estimators=150, random_state=42, 
                             n_jobs=2, oob_score=True)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
clf=gcv.fit(batch_x, batch_y)
clf=gcv.fit(batch_x, batch_y)

pred_test=clf.predict(test_f)
ac=accuracy_score(test_l,pred_test)*100
print(ac)

# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(batch_x, batch_y)
# pred_test = clf.predict(test_f)

# ac=accuracy_score(test_l,pred_test)*100
# print(ac)






X_index = [1,3,5,6,8]

test_set = parse.TestDataset('test.csv')
test_set.read_data(X_index)
test_set.std_data(X_index)
# X_,X_i = test_set.next_batch(test_set.SIZE)
# for i in range(len(X_)):
#     print(i,X_[i])
f = open('out.csv','w')
f.write("PassengerId,Survived\n")
# test_set.std_data(X_index)
for i in range(2):
    X_,X_i = test_set.next_batch(test_set.SIZE/2)
    pred_test=clf.predict(X_)
    for i in range(test_set.SIZE/2):
        # tt = pred_test[i]
        f.write(str(X_i[i])+','+str(int(pred_test[i]))+'\n')

f.close()










# train_set.read_data(X_index,Y_index)

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# import tensorflow as tf

# # Parameters
# learning_rate = 0.001
# training_epochs = 200
# batch_size = 100
# display_step = 2

# # Network Parameters
# n_hidden_1 = 512 # 1st layer number of features
# n_hidden_2 = 512 # 2nd layer number of features
# n_input = 4 # MNIST data input (img shape: 28*28)
# n_classes = 2 # MNIST total classes (0-9 digits)

# # tf Graph input
# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])

# def AccuracY(predictions, labels):
#     # print labels[15],predictions[15]
#     # for i in range(len(labels)):
#     #     print predictions[i],labels[i]
#     correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     return accuracy.eval()*100.0
# # Create model
# def multilayer_perceptron(x, weights, biases):
#     # Hidden layer with RELU activation
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     # Hidden layer with RELU activation
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.tanh(layer_2)
#     # Output layer with linear activation
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     # out_layer = tf.argmax(out_layer, 1)
#     # out_layer = tf.nn.softmax(out_layer)
#     out_layer = tf.cast(out_layer > 0, out_layer.dtype) * out_layer
#     return out_layer

# # Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }

# # Construct model
# pred = multilayer_perceptron(x, weights, biases)

# # Define loss and optimizer
# # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))+tf.reduce_mean(tf.square(pred - y))//132020232.000000000
# cost = tf.reduce_mean(tf.square(pred-y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # Initializing the variables
# init = tf.global_variables_initializer()

# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)

#     # Training cycle
#     for epoch in range(training_epochs):
#         avg_cost = 0.
#         total_batch = int(train_set.SIZE/batch_size)
#         train_set.reset_counter()
#         # Loop over all batches
#         for i in range(total_batch):
#             batch_x, batch_y = train_set.next_batch(batch_size)
#             # Run optimization op (backprop) and cost op (to get loss value)
#             _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
#                                                           y: batch_y})
#             # Compute average loss
#             avg_cost += c / total_batch
#         # Display logs per epoch step
#         if epoch % display_step == 0:
#             # print("Epoch:", '%04d' % (epoch+1), "cost=", \
#             #     "{:.9f}".format(avg_cost))
           

#             # Test model
#             correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#             # Calculate accuracy
#             accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#             print("Epoch:", '%04d' % (epoch+1),"Accuracy:", accuracy.eval({x: batch_x, y: batch_y}))
#             out = sess.run([pred],feed_dict={x:batch_x})
#             acc = AccuracY(out[0],batch_y)
#             # for i in a
#             print ("Epoch:", '%04d' % (epoch+1),"Accuracy out :",acc)
#             print ()
#             # print (out[0][12],batch_y[12])
#     print("Optimization Finished!")

#     # X_index = [i for i in range(12)]
#     X_index = [1,3,5,6,8]
#     # X_index = [2,4,5,6,7,9]
#     # Y_index = 1
#     # train_set.read_data(X_index,Y_index,True,2)
#     test_set = parse.TestDataset('test.csv')
#     test_set.read_data(X_index)
#     f = open('out.csv','w')
#     f.write("PassengerId,Survived\n")
#     # test_set.std_data(X_index)
#     for i in range(2):
#         X_epoch,X_i = test_set.next_batch(test_set.SIZE/2)
#         # print X_epoch,X_i
#         ot = sess.run([pred],feed_dict={x:X_epoch})
#         # print ot[0][10]
        
#         for i in range(test_set.SIZE/2):
#             tt = ot[0][i]
#             f.write(str(X_i[i])+','+str(int(1-tt[0]>tt[1]))+'\n')

#     f.close()