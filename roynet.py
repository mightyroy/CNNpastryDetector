#import modules
import numpy as np
import tensorflow as tf
import cv2
import time
import sys
from tensorflow.python.framework import ops

class RoyNet():

	
	    #define layers
	def conv_layer(self,idx,inputs,filters,size,stride):
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
	    if self.disp_console : print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels))
	    #return tf.maximum(alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')
	    return tf.maximum(conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

	def pooling_layer(self,idx,inputs,size,stride):
	    if self.disp_console : print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride)
	    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

	def fc_layer(self,idx,inputs,hiddens, flat = False,linear = False):
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
	    if self.disp_console : print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))	
	    if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
	    ip = tf.add(tf.matmul(inputs_processed,weight),biases)
	    return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

	def buildLayers(self):


		#build layers
		print "Building graph..."
		x = tf.placeholder(tf.float32,[None,448,448,3])
		#y = tf.placeholder(tf.float32, [None, self.n_classes])
		conv_1 = self.conv_layer(1,x,16,3,1)
		pool_2 = self.pooling_layer(2,conv_1,2,2)
		conv_3 = self.conv_layer(3,pool_2,32,3,1)
		pool_4 = self.pooling_layer(4,conv_3,2,2)
		conv_5 = self.conv_layer(5,pool_4,64,3,1)
		pool_6 = self.pooling_layer(6,conv_5,2,2)
		conv_7 = self.conv_layer(7,pool_6,128,3,1)
		pool_8 = self.pooling_layer(8,conv_7,2,2)
		conv_9 = self.conv_layer(9,pool_8,256,3,1)
		pool_10 = self.pooling_layer(10,conv_9,2,2)
		conv_11 = self.conv_layer(11,pool_10,512,3,1)
		pool_12 = self.pooling_layer(12,conv_11,2,2)
		conv_13 = self.conv_layer(13,pool_12,1024,3,1)
		conv_14 = self.conv_layer(14,conv_13,1024,3,1)
		conv_15 = self.conv_layer(15,conv_14,1024,3,1)
		fc_16 = self.fc_layer(16,conv_15,256,flat=True,linear=False)
		fc_17 = self.fc_layer(17,fc_16,4096,flat=False,linear=False)
		fc_19 = self.fc_layer(19,fc_17,4,flat=False,linear=False)
		scores = tf.nn.softmax(fc_19)

		return scores, x

	def shutdown():
		self.sess.close()

	def __init__(self):

		#constants
		self.disp_console = False
		self.alpha = 0.1
		self.grid_size = 7
		self.classes =  ["bananabread", "cinnamonroll", "croissant", "hotcross"]
		self.n_classes = len(self.classes)

		#build layers
		self.scores , self.x= self.buildLayers()
		varlist = []
		finallayer = []
		print ""

		for v in tf.all_variables():
		    varlist.append(v)
		    if v.name == 'fc_2/w:0' or v.name =='fc_2/b:0' or v.name == 'fc_1/w:0' or v.name == 'fc_1/b:0' or v.name == 'fc/w:0' or v.name == 'fc/b:0' or v.name == 'conv_15/w:0' or v.name == 'conv_15/b:0':
		        finallayer.append(v)

		#Start session

		cfg = tf.ConfigProto(inter_op_parallelism_threads=8,
		                   intra_op_parallelism_threads=8,
		                   log_device_placement=True)
		self.sess = tf.Session(config=cfg)

		#restore select variables from checkpoint
		saver = tf.train.Saver(varlist)
		saver.restore(self.sess,"tmp/saved/modelRoy7.ckpt")
		print '/nWeights succesfully loaded from checkpoint'

	def predict(self,img):
		img = img.reshape((1,448,448,3))
		feed_dict = {self.x: img}
		y = self.sess.run(self.scores, feed_dict)
		return y




if __name__ == '__main__':
	r = RoyNet()
