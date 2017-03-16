#!/usr/bin/env python
import numpy as np
import tensorflow as tf


with open('datalines_500.npy') as f:
    datalines = np.load(f)

with open('labels_500.npy') as f:
    labels = np.load(f)


def accuracy(predictions, labels):
    return (1.* np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/
        predictions.shape[0])


count = datalines.shape[0]
image_size = datalines.shape[1]
num_chanels = datalines.shape[3]

train_set = datalines[:int(count*.8)]
train_labels = labels[:int(count*.8)]
validation_set = datalines[int(count*.8):int(count*0.9)]
validation_labels = labels[int(count*.8):int(count*0.9)]

test_set = datalines[int(count*0.9):]
test_labels = labels[int(count*0.9):]

print "Train set:      ", train_set.shape, train_labels.shape
print "Validation set: ", validation_set.shape, validation_labels.shape
print "Test set:       ", test_set.shape, test_labels.shape



num_labels = 2
batch_size = 32
patch_size = 3
depth = 16

num_hidden = 64

graph = tf.Graph()

def weigth_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(data, weights):
    return tf.nn.conv2d(data, weights, strides=[1,1,1,1], padding='SAME')

def pool2x2(data):
    return tf.nn.max_pool(data, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with graph.as_default():
    tf_train_dataset = tf.placeholder(
        tf.float32,
        shape=(batch_size, image_size, image_size, num_chanels))
    tf_train_labels = tf.placeholder(
        tf.float32,
        shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(validation_set)
    tf_test_dataset = tf.constant(test_set)
    
    layer1_weigths = weigth_var([patch_size, patch_size, num_chanels, depth])
    layer1_biases = bias_var([depth])
    
    layer2_weigths = weigth_var([patch_size, patch_size, depth, depth])
    layer2_biases = bias_var([depth])
    
    layer3_weigths = weigth_var([patch_size, patch_size, depth, depth])
    layer3_biases = bias_var([depth])

    layer4_weigths = weigth_var([image_size//8*image_size//8*(depth), num_hidden])
    layer4_biases = bias_var([num_hidden])
   
    #layer5_weigths = weigth_var([num_hidden, num_hidden])
    #layer5_biases = bias_var([num_hidden])

    layer6_weigths = weigth_var([num_hidden, num_labels])
    layer6_biases = bias_var([num_labels])
    
    def model(data):
        conv = pool2x2(conv2d(data, layer1_weigths) + layer1_biases)
        hidden = tf.nn.relu(conv)
        
        conv = pool2x2(conv2d(hidden, layer2_weigths) + layer2_biases)
        hidden = tf.nn.relu(conv)

        conv = pool2x2(conv2d(hidden, layer3_weigths) + layer3_biases)
        hidden = tf.nn.relu(conv)

        #conv = pool2x2(conv2d(hidden, layer31_weights) + layer31_biases)
        #hidden = tf.nn.relu(conv)
        
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], -1])
        hidden = tf.nn.relu(tf.matmul(reshape, layer4_weigths) + layer4_biases)

        #hidden = tf.nn.relu(tf.matmul(hidden, layer5_weigths) + layer5_biases)
        
        return tf.matmul(hidden, layer6_weigths) + layer6_biases
    
    logits = model(tf_train_dataset)
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf_train_labels))
    
    global_step = tf.Variable(0)
    learn_rate = tf.train.exponential_decay(0.005, global_step, 100, 0.7)
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(
        loss, global_step=global_step)
    
    batch_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


# In[8]:

num_steps = 5001
epoch = -1
prev_epoch = -1
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print "Initialized"
    for step in range(num_steps):
        prev_epoch = epoch
        epoch=(step * batch_size) // (train_labels.shape[0]-batch_size)
        if (prev_epoch != epoch):
            permutation = np.random.permutation(train_labels.shape[0])
            train_labels = train_labels[permutation]
            train_set = train_set[permutation]

        offset = (step * batch_size) % (train_labels.shape[0]-batch_size)
        batch_data = train_set[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]
        
        feed_dict = {
            tf_train_dataset: batch_data,
            tf_train_labels: batch_labels,
        }
        
        _, l, predictions = session.run(
            [optimizer, loss, batch_prediction], feed_dict=feed_dict)
        
        if (prev_epoch != epoch):
            v = valid_prediction.eval()
            print "epoch: {:4}, learn_rate {:.8f} loss: {:.4f} Batch: {:.2%}, valid: {:.2%}".format(
                    epoch,
                    learn_rate.eval(),
                    l,
                    accuracy(predictions, batch_labels),
                    accuracy(v, validation_labels))
            
    #print "Test accuracy: {:.2f}%".format(accuracy(
    #    test_prediction.eval(), test_labels))
