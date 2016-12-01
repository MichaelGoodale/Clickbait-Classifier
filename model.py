import tensorflow as tf
import numpy as np
from tensorflow.contrib.session_bundle import exporter

##Model hyperparams
keep_probability = 0.5 #Drop out probability
batch_size = 50 
wordVecSize = 50 #The dimensionality of the word vectors
sentenceMax = 30 #Number of words to pad the sentence to
filter_amount= 100 #number of filters per window size
h = [3,4,5] #differing window sizes
epochNumber = 1

data = np.loadtxt('ProcessedData/data.txt')
labels = np.loadtxt('ProcessedData/labels.txt',dtype=np.int32)
labels = np.eye(2)[labels]

test_data = np.loadtxt('ProcessedData/testData.txt')
test_labels = np.loadtxt('ProcessedData/testLabels.txt', dtype=np.int32)
test_labels = np.eye(2)[test_labels]
test_data = test_data[:,np.newaxis,:,np.newaxis]

#Data information
sentence_size = data.shape[1]
data_size = data.shape[0]-data.shape[0]%batch_size
data=data[0:data_size,:] #This gets rid of data that can't fit in a batch.
labels = labels[0:data_size]
#Calculate number of steps necessary
steps = epochNumber*(data_size//batch_size)
print(data_size)
print(steps)
sentence_placeholder = tf.placeholder(tf.float32, shape=[None, 1, wordVecSize*sentenceMax, 1], name='Batch_Data')
labels_placeholder = tf.placeholder(tf.float32, shape=[None,2], name='Batch_Labels')

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, b):
	return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')+b)

def max_pool(x , window_size):
	  return tf.nn.max_pool(x, ksize=[1, 1, sentenceMax*wordVecSize, 1],
                        strides=[1, 1, sentenceMax*wordVecSize, 1], padding='SAME')
def get_batch(x, i):
	i=i%(data_size//batch_size)
	batch = x[i*batch_size:i*batch_size+batch_size,:]
	return batch

def shuffle_data(x, y):
	assert len(x) == len(y)
	rand_index = np.random.permutation(len(x))
	return x[rand_index], y[rand_index]
	
#Convolution
with tf.name_scope('Convolution') as scope:
	conv_filters = []
	conv_ops = []
	max_pools = []
	bias = bias_variable([1, 1, wordVecSize*sentenceMax, 1], 'Convolution_Bias')
	for i in h:
		conv_filters.append(weight_variable([1,wordVecSize*i,1,filter_amount], 'ConvFilter_'+str(i)))
		conv_ops.append(conv2d(sentence_placeholder, conv_filters[i-h[0]],bias))
		max_pools.append(max_pool(conv_ops[i-h[0]],i))	
	max_pooled = tf.squeeze(tf.concat(3,max_pools))

dropped_out = tf.nn.dropout(max_pooled,keep_probability)

#Softmax Model
with tf.name_scope('Softmax') as scope:
	softmax_bias = bias_variable([1,2], 'Softmax_Bias')
	softmax_weight = weight_variable([len(h)*filter_amount,2], 'Softmax_Weights')
	soft_model = tf.nn.softmax(tf.matmul(dropped_out,softmax_weight) + softmax_bias, name = 'Softmax_Model')

#Cost and Training
cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels_placeholder*tf.log(soft_model),reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#Evaluation Metrics
with tf.name_scope('Metrics') as scope:
	squaredErr = tf.scalar_summary('Squared Error',tf.reduce_mean(tf.square(labels_placeholder - soft_model)))
	crossEnt = tf.scalar_summary('Cross Entropy',cross_entropy)
	labels_ans = tf.argmax(labels_placeholder,1)
	model_ans = tf.argmax(soft_model,1)
	correct_answers = tf.equal(labels_ans,model_ans)
	#Precision/Recall Diagnostic Info
	true_positives = tf.reduce_sum(model_ans*labels_ans)
	false_positives = tf.reduce_sum(model_ans)-true_positives
	true_negatives = tf.reduce_sum(tf.cast(tf.equal(model_ans+labels_ans, tf.zeros_like(model_ans)),tf.int64))
	false_negatives = tf.reduce_sum(tf.cast(tf.equal(model_ans,tf.zeros_like(model_ans)),tf.int64))-true_negatives
	precision = true_positives/(true_positives+false_positives)
	recall = true_positives/(true_positives+false_negatives)
	accuracy = tf.scalar_summary('Accuracy',tf.reduce_mean(tf.cast(correct_answers,tf.float32)))
	precision_sum = tf.scalar_summary('Precision',precision)
	recall_sum = tf.scalar_summary('Recall',recall)
	f1_score = tf.scalar_summary('F1 Score', (2*precision*recall)/(precision+recall))
	summaries = tf.merge_all_summaries()
	
#Initalisation
sess=tf.Session()
sess.run(tf.initialize_all_variables())
train_writer = tf.train.SummaryWriter('Train', graph = sess.graph)
test_writer = tf.train.SummaryWriter('Test', graph = sess.graph)
#Saver
saver = tf.train.Saver(sharded=True)

#Uncomment to continue training
saver.restore(sess,'savedModels/model.ckpt')

for step in range(steps):
	if(step%(data_size)==0):#shuffle dataset for each epoch
		data,labels = shuffle_data(data,labels)
	data_batch = get_batch(data, step)
	labels_batch = get_batch(labels, step)
	data_batch = data_batch[:,np.newaxis,:,np.newaxis]
	summary, _  = sess.run([summaries,train_op], feed_dict={sentence_placeholder:data_batch, labels_placeholder:labels_batch})
	train_writer.add_summary(summary, step)
	
	if(step%500 == 0):
		summary = sess.run(summaries, feed_dict = {sentence_placeholder:test_data, labels_placeholder:test_labels})
		test_writer.add_summary(summary, step)
	#Model Saving
	#if step%1000==0:
		#save = saver.save(sess, 'savedModels/model.ckpt')
	
export_path = 'exportedModel'
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'sentences': sentence_placeholder}),
        'outputs': exporter.generic_signature({'scores': soft_model})})
model_exporter.export(export_path,tf.constant(1), sess)
