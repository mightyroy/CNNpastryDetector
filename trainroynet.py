#import modules
import numpy as np
import tensorflow as tf
import cv2
import time
import sys
sys.path.insert(0, './dataset/')
import importBakeryDataCSV
from tensorflow.python.framework import ops

#import dataset
data = importBakeryDataCSV.BakeryDataset()
total_images = data.total_images
print "Total images in dataset: %i" % total_images

#constants
disp_console = True
alpha = 0.1
grid_size = 7
classes =  ["bananabread", "cinnamonroll", "croissant", "hotcross"]
n_classes = len(classes)

#define layers
def conv_layer(idx,inputs,filters,size,stride):
    channels = inputs.get_shape()[3]
    with tf.variable_scope("conv"):
        #weight = tf.get_variable('w',[size,size,int(channels),filters])
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1),name='w')
        biases = tf.Variable(tf.constant(0.1, shape=[filters]),name='b')

    pad_size = size//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    inputs_pad = tf.pad(inputs,pad_mat)

    conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')	
    conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')	
    if disp_console : print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels))
    #return tf.maximum(alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')
    return tf.maximum(conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

def pooling_layer(idx,inputs,size,stride):
    if disp_console : print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride)
    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

def fc_layer(idx,inputs,hiddens, flat = False,linear = False):
    input_shape = inputs.get_shape().as_list()
    if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_transposed = tf.transpose(inputs,(0,3,1,2))
        inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
    else:
        dim = input_shape[1]
        inputs_processed = inputs
    with tf.variable_scope("fc"):
        weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1),name='w')
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]),name='b')
    if disp_console : print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))	
    if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
    ip = tf.add(tf.matmul(inputs_processed,weight),biases)
    return tf.maximum(alpha*ip,ip,name=str(idx)+'_fc')

#build layers
print "Building graph..."
x = tf.placeholder(tf.float32,[None,448,448,3])
y = tf.placeholder(tf.float32, [None, n_classes])
conv_1 = conv_layer(1,x,16,3,1)
pool_2 = pooling_layer(2,conv_1,2,2)
conv_3 = conv_layer(3,pool_2,32,3,1)
pool_4 = pooling_layer(4,conv_3,2,2)
conv_5 = conv_layer(5,pool_4,64,3,1)
pool_6 = pooling_layer(6,conv_5,2,2)
conv_7 = conv_layer(7,pool_6,128,3,1)
pool_8 = pooling_layer(8,conv_7,2,2)
conv_9 = conv_layer(9,pool_8,256,3,1)
pool_10 = pooling_layer(10,conv_9,2,2)
conv_11 = conv_layer(11,pool_10,512,3,1)
pool_12 = pooling_layer(12,conv_11,2,2)
conv_13 = conv_layer(13,pool_12,1024,3,1)
conv_14 = conv_layer(14,conv_13,1024,3,1)
conv_15 = conv_layer(15,conv_14,1024,3,1)
fc_16 = fc_layer(16,conv_15,256,flat=True,linear=False)
fc_17 = fc_layer(17,fc_16,4096,flat=False,linear=False)
fc_19 = fc_layer(19,fc_17,4,flat=False,linear=False)

varlist = []
finallayer = []
print ""

for v in tf.all_variables():
    varlist.append(v)
    if v.name == 'fc_2/w:0' or v.name =='fc_2/b:0' or v.name == 'fc_1/w:0' or v.name == 'fc_1/b:0' or v.name == 'fc/w:0' or v.name == 'fc/b:0' or v.name == 'conv_15/w:0' or v.name == 'conv_15/b:0':
        finallayer.append(v)

#Start session
NUM_THREADS = 8
cfg = tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,
                   intra_op_parallelism_threads=NUM_THREADS,
                   log_device_placement=True)
sess = tf.Session(config=cfg)

#Hyperparameter placeholder
learning_rate = tf.placeholder(tf.float32, shape=[])
    
#loss and gradient descent functions
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc_19, y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,var_list=finallayer)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(fc_19,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initialize all variables
sess.run(tf.initialize_all_variables())

#restore select variables from checkpoint
saver = tf.train.Saver(varlist)
saver.restore(sess,"tmp/saved/modelRoy7.ckpt")
print '/nWeights succesfully loaded from checkpoint'

#tensorboard
tf.scalar_summary("cross_entropy", loss)
tf.scalar_summary("training_accuracy", accuracy)
tf.image_summary('layer3', sess.run(varlist[0])[:,:,:,0:3])

summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('./tmp/roynet_log', graph_def=sess.graph_def)


#set learning rate
lr = 0.000000001
training_iters = 20000
batch_size = 40
display_step = 1

try:
	while True:

		for i in range(training_iters):
		            
		    epoch = (batch_size*i)/float(total_images)
		    
		    batch_x , batch_y = data.pickSample(batch_size)
		    
		    if i % display_step == 0:
		        # Calculate batch accuracy and loss
		        summary_str, _, acc, batch_loss = sess.run([summary_op,optimizer, accuracy,loss], feed_dict={x: batch_x, y: batch_y,learning_rate: lr})
		        print "Step " + str(i) + ", Epoch "+ str(epoch)+ ", Minibatch Loss= " + str(batch_loss) + ", Training Accuracy= " + str(acc)
		        if batch_loss != batch_loss:
		            #check for loss = NaN
		            break
		    else:
		        summary_str, _ = sess.run([summary_op,optimizer], feed_dict={x: batch_x, y: batch_y,learning_rate: lr})
		    
		    #save at specific steps:
		    if i % 100 == 0:
		        saver = tf.train.Saver(varlist)
		        save_path = saver.save(sess, "./tmp/modelRoy.ckpt")
		        print("Model saved in file: %s" % save_path)
		    
		    #saving tensorboard summary
		    
		    if i%10 ==0 :
		        summary_writer.add_summary(summary_str, i)

		save_path = saver.save(sess, "./tmp/modelRoy.ckpt")
		sess.close()
		print 'done'

except KeyboardInterrupt:
	print "Stopping training... saving checkpoint first.."
	save_path = saver.save(sess, "./tmp/modelRoy.ckpt")
	sess.close()
	print 'done'


