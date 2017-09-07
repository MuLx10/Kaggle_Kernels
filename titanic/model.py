import parse

train_set = parse.TrainingDataset('train.csv')
test_set = parse.TestDataset('test.csv')
# X_index = [i for i in range(12)]
# X_index = [2,4,6,7,9]
# # X_index = [2,4,5,6,7,9]
# Y_index = 1

# train_set.read_data(X_index,Y_index)
# # train_set.std_data(X_index)

# p,y = train_set.next_batch(10)
# print p,y




import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_input = 4
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_nodes_hl4 = 500

n_classes = 1
batch_size = 100

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float',[None,1])

def accuracy(predictions, labels):
    # print labels[15],predictions[15]
    for i in range(len(labels)):
        print predictions[i],labels[i]
    correct_prediction = tf.equal(labels,predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.eval()*100.0


def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_input, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    #0.95639998
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    # # 0.95959997


    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    # 0.95730001
    output = tf.matmul(l4,output_layer['weights']) + output_layer['biases']

    # output = tf.nn.softmax(output)
    output = tf.nn.relu(output)

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)


    X_index = [i for i in range(12)]
    X_index = [2,4,6,7,9]
    # X_index = [2,4,5,6,7,9]
    Y_index = 1
    # train_set.read_data(X_index,Y_index,True,2)
    train_set.read_data(X_index,Y_index)
    # train_set.std_data(X_index)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.square(tf.square(prediction - y)))
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost*cost)
    # optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            train_set.reset_counter()
            epoch_loss = 0
            for _ in range(int(train_set.SIZE/batch_size)):

                epoch_x, epoch_y = train_set.next_batch(batch_size)
                # print epoch_x[0],epoch_y[0]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        
        ot = sess.run([prediction],feed_dict={x:epoch_x})       
        acc = accuracy(ot[0],epoch_y)
        print ("Accuracy :",acc)

        # # X_index = [i for i in range(12)]
        # X_index = [1,3,5,6,8]
        # # X_index = [2,4,5,6,7,9]
        # # Y_index = 1
        # # train_set.read_data(X_index,Y_index,True,2)
        # test_set.read_data(X_index)
        # # test_set.std_data(X_index)
        # X_epoch,X_i = test_set.next_batch(test_set.SIZE)
        # # print X_epoch,X_i
        # ot = sess.run([prediction],feed_dict={x:X_epoch})
        # print ot[0][10]
        # f = open('out.csv','w')
        # for i in range(test_set.SIZE):
        #     tt = ot[0][i][0]
        #     if(tt == 1.0):
        #         tt = 1
        #     else:
        #         tt = 0
        #     f.write(str(X_i[i])+','+str(tt)+'\n')

        # f.close()
        




            # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
# print(neural_network_model(tx),ty)
# 

# test_set = parse.TestDataset('test.csv')



