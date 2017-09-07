"""
CNN on Not MNIST
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
# On command line: python3 mnist_a2j_cnn.py
# Prerequisite: tensorflow 1.0 (see tensorflow.org)
# must run mnist_a2j_2pickle.py first (one-time) to generate the data

from __future__ import print_function
print (25)
import numpy as np
import tensorflow as tf
# import pickle
import parse
import time

from tensorflow.examples.tutorials.mnist import input_data
print("Import done")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# use of pickle to speed up loading of data
# pickle_file = open( "mnist_a2j.pickle", "rb" )
# data = pickle.load(pickle_file)
# test_labels = data["test_labels"]




X_labels = [i for i in xrange(1,785)]
train_set = parse.Training_Dataset('train.csv')
train_set.read_data(X_labels,0,True,10)


# train_labels = data["all_labels"]
# # test_dataset = data["test_dataset"]
# train_dataset = data["all_dataset"]


num_labels = 10
num_data = train_set.SIZE

image_size = 28
channel = 1
# train_dataset = train_dataset.reshape((-1,image_size, image_size, channel)).astype(np.float32)
# test_dataset = test_dataset.reshape((-1,image_size, image_size, channel)).astype(np.float32)

# print(train_dataset.shape)
# print(train_labels.shape)
# print(test_dataset.shape)
# print(test_labels.shape)

# small batch size appears to work
batch_size = 100
depth1 = 32
depth2 = 64

hidden_units1 = int((depth2*image_size/4*image_size/4))
hidden_units2 = 1024
patch_size = 3
num_steps = 50*2
dropout = 0.8
learning_rate = 0.001

graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(tf.float32,shape=(batch_size, image_size , image_size, channel))
    Y = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    weights0 = tf.Variable(
        tf.truncated_normal([patch_size, patch_size, channel, depth1], stddev=1.0))
    biases0 = tf.Variable(tf.zeros([depth1]))

    weights1 = tf.Variable(
        tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev=1.0))
    biases1 = tf.Variable(tf.zeros([depth2]))

    weights2 = tf.Variable(
        tf.truncated_normal([hidden_units1, hidden_units2], stddev=0.1))
    biases2 = tf.Variable(tf.constant(1.0, shape=[hidden_units2]))

    weights3 = tf.Variable(
        tf.truncated_normal([hidden_units2, num_labels], stddev=0.1))
    biases3 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    keep_prob = tf.placeholder(tf.float32)

    def model(data, dropout):
        conv1 = tf.nn.conv2d(data, weights0, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.bias_add(conv1, biases0)
        pool1 = tf.nn.max_pool(tf.nn.relu(relu1), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.conv2d(pool1, weights1, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.bias_add(conv2, biases1)
        pool2 = tf.nn.max_pool(tf.nn.relu(relu2), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

        logits1 = tf.nn.relu( tf.add(tf.matmul(reshape, weights2), biases2) )
        logits1 = tf.nn.dropout(logits1, dropout)
        logits2 = tf.add( tf.matmul(logits1, weights3), biases3 )

        return logits2, tf.nn.softmax(logits2)

    logits, train_pred = model(X,dropout=dropout)
    loss = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # test_logits, test_pred = model(test_dataset,1.0)

def accuracy(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.eval()*100.0

# with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as session:
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(tf.__version__)
    continue_train = True
    for step in range(num_steps):
    	if not continue_train:
    		break
        # offset = (step * batch_size)% (num_data - batch_size)
        # Generate a minibatch.
        # batch_data = train_dataset[offset:(offset + batch_size), :]
        # batch_labels = train_labels[offset:(offset + batch_size), :]
        # feed_dict = 
        train_set.reset_counter()
        # for _ in range(1):
        for oou in range(train_set.SIZE/batch_size):
            batch_data,batch_labels,cctr = train_set.next_batch(batch_size)
            batch_data = batch_data.reshape((-1,image_size, image_size, channel)).astype(np.float32)

            _, l, predictions = session.run([optimizer, loss, train_pred], feed_dict={X:batch_data, Y: batch_labels})
            if (oou % 280 == 0):
                print("Minibatch (size=%d) loss at step %d: %f" % (batch_size, step, l))
                acc = accuracy(predictions,batch_labels)
                # if(acc>=98.5):
                # 	print(continue_train)
                # 	continue_train = False
                # 	break
                print("Minibatch accuracy: %f%%" % acc)

        for oou in range(int(mnist.train.num_examples/batch_size)):
            batch_data,batch_labels = mnist.train.next_batch(batch_size)
            batch_data = batch_data.reshape((-1,image_size, image_size, channel)).astype(np.float32)

            _, l, predictions = session.run([optimizer, loss, train_pred], feed_dict={X:batch_data, Y: batch_labels})
            if (oou % 280 == 0):
                print("Minibatch (size=%d) loss at step %d: %f_mnist" % (batch_size, step, l))
                acc = accuracy(predictions,batch_labels)
                # if(acc>=98.5):
                # 	print(continue_train)
                # 	continue_train = False
                # 	break
                print("Minibatch accuracy_mnist: %f%%" % acc)

    # # Accuracy: 97.4%
    # print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(),
    #                                          test_labels))
    print("Elapsed: ", elapsed(time.time() - start_time))

    # print(np.rint(test_pred.eval()))


    x_labels = [i for i in xrange(784)]
    test_set = parse.TestDataset('test.csv')
    test_set.read_data(x_labels)
    f = open("output.csv",'w')
    f.write("ImageId,Label\n")
    counTr = 1
    print(test_set.SizE, batch_size,test_set.SizE/ batch_size)
    for _ in xrange(test_set.SizE/ batch_size):
        x_test_batch,_ = test_set.next_batch(batch_size)
        x_test_batch = x_test_batch.reshape((-1,image_size, image_size, channel)).astype(np.float32)
        y_label = session.run([train_pred],feed_dict={X:x_test_batch})

        for yi in range(len(y_label[0])):
            y_o = list(y_label[0][yi])
            # print len(y_label[0][yi]),len(y_label)
            f.write(str(counTr)+','+str(y_o.index(max(y_o)))+'\n')
            counTr += 1       
        # print(counTr) 


    # for _ in xrange(test_set.SizE% batch_size):
    # x_test_batch,_ = test_set.next_batch(test_set.SizE% batch_size)
    # x_test_batch = x_test_batch.reshape((-1,image_size, image_size, channel)).astype(np.float32)
    # y_label = session.run([train_pred],feed_dict={X:x_test_batch})

    # for yi in range(len(y_label[0])):
    #     y_o = list(y_label[0][yi])
    #     f.write(str(counTr)+','+str(y_o.index(max(y_o)))+'\n')
    #     counTr += 1
    f.close()

