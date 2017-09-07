import tensorflow as tf
import parse 



n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_nodes_hl4 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    #0.95639998
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.tanh(l3)
    # # 0.95959997


    # l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    # l4 = tf.nn.relu(l4)
    # 0.95730001
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    output = tf.nn.softmax(output)

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    
    X_labels = [i for i in xrange(1,785)]
    train_set = parse.Training_Dataset('train.csv')
    train_set.read_data(X_labels,0,True,10)
    hm_epochs = 1000
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            train_set.reset_counter()
            for _ in range(int(train_set.SIZE/ batch_size)):
                epoch_x, epoch_y, ix = train_set.next_batch(batch_size)

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            outputt = sess.run([prediction],feed_dict={x:epoch_x})

            epoch_y_ = list(epoch_y[9])
            outputt_ = list(outputt[0][9])
            print(epoch_y_.index(max(epoch_y_)),outputt_.index(max(outputt_)))
            # print(len(epoch_y[0]))
        
        # print len(y_),len(epoch_y),len(epoch_y[0]),(y_[7])
        
        for it in xrange(len(epoch_y)):
            y_ = list(epoch_y[it])
            outputt_ = list(outputt[0][it])
            # print it
            print(y_.index(max(y_)),outputt_.index(max(outputt_)))



            
        x_labels = [i for i in xrange(784)]
        test_set = parse.TestDataset('test.csv')
        test_set.read_data(x_labels)
        f = open("output.csv",'w')
        f.write("ImageId,Label\n")
        counTr = 1
        print(test_set.SizE, batch_size,test_set.SizE/ batch_size)
        for _ in xrange(test_set.SizE/ batch_size):
            x_test_batch,_ = test_set.next_batch(batch_size)
            y_label = sess.run([prediction],feed_dict={x:x_test_batch})

            for yi in range(len(y_label[0])):
                y_o = list(y_label[0][yi])
                # print len(y_label[0][yi]),len(y_label)
                f.write(str(counTr)+','+str(y_o.index(max(y_o)))+'\n')
                counTr += 1       
            # print(counTr) 


        # for _ in xrange(test_set.SizE% batch_size):
        x_test_batch,_ = test_set.next_batch(test_set.SizE% batch_size)
        y_label = sess.run([prediction],feed_dict={x:x_test_batch})

        for yi in range(len(y_label[0])):
            y_o = list(y_label[0][yi])
            f.write(str(counTr)+','+str(y_o.index(max(y_o)))+'\n')
            counTr += 1
        f.close()




	        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	        # print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
# print(neural_network_model(tx),ty)