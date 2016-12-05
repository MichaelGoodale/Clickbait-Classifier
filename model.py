import tensorflow as tf
import numpy as np
from tensorflow.contrib.session_bundle import exporter

SAVE_MODEL = True
LOAD_MODEL = False
CLASS_NAMES = tf.constant(["News", "Clickbait"])

##Model hyperparams
KEEP_PROBABILITY = 0.5 #Drop out probability
BATCH_SIZE = 50
WORD_VECTOR_SIZE = 50 #The dimensionality of the word vectors
SENTENCE_MAX = 30 #Number of words to pad the sentence to
FILTER_AMOUNT= 100 #number of filters per window size
h = [3,4,5] #differing window sizes
NUMBER_EPOCHS = 2

def data_load(data_path, label_path):
    X = np.loadtxt(data_path, dtype=np.float32)
    Y = np.loadtxt(label_path, dtype=np.int32)
    Y = np.eye(2)[Y]
    X = X[:, np.newaxis, :, np.newaxis]
    return X, Y

data, labels = data_load("ProcessedData/data.txt","ProcessedData/labels.txt")
test_data, test_labels = data_load("ProcessedData/testData.txt","ProcessedData/testLabels.txt")

#Data information
sentence_size = data.shape[1]
data_size = data.shape[0]-data.shape[0]%BATCH_SIZE

#Calculate number of steps necessary
steps = NUMBER_EPOCHS*(data_size//BATCH_SIZE)
 
sentence_placeholder = tf.placeholder(tf.float32, shape=[None, 1, WORD_VECTOR_SIZE*SENTENCE_MAX, 1], name="Batch_Data")
labels_placeholder = tf.placeholder(tf.float32, shape=[None,2], name="Batch_Labels")
drop_out_prob = tf.placeholder(tf.float32,shape=())

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, filter, b):
    return tf.nn.relu(tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding= "SAME" ) + b)

def max_pool(x):
      return tf.reduce_max(x, axis=2,keep_dims=True)
	  
def get_batch(x, i):
    i = i % (data_size // BATCH_SIZE)
    if x.ndim == 2:
        return x[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE, :]
    else:
        return x[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE, :, :, :]

def shuffle_data(x, y):
    assert len(x) == len(y)
    rand_index = np.random.permutation(len(x))
    return x[rand_index], y[rand_index]

#Convolution
with tf.name_scope("Convolution") as scope:
    conv_filters = []
    conv_ops = []
    max_pools = []
    for i in h:
        conv_filters.append(weight_variable([1, WORD_VECTOR_SIZE*i, 1, FILTER_AMOUNT], "ConvFilter_"+str(i)))
        conv_ops.append(conv2d(sentence_placeholder, conv_filters[i-h[0]], bias_variable([1, 1, WORD_VECTOR_SIZE*SENTENCE_MAX, 1], "Conv_Bias")))
        max_pools.append(max_pool(conv_ops[i-h[0]]))
    max_pooled = tf.squeeze(tf.concat(3, max_pools), [1, 2])

dropped_out = tf.nn.dropout(max_pooled, drop_out_prob)

#Softmax Model
with tf.name_scope("Softmax") as scope:
    softmax_bias = bias_variable([1,2], "Softmax_Bias")
    softmax_weight = weight_variable([len(h)*FILTER_AMOUNT,2], "Softmax_Weights")
    soft_model = tf.matmul(dropped_out, softmax_weight) + softmax_bias
    soft_max = tf.nn.softmax(soft_model)

#Cost and Training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(soft_model, labels_placeholder))
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Evaluation Metrics
with tf.name_scope("Metrics") as scope:
    squaredErr = tf.scalar_summary("Squared Error",tf.reduce_mean(tf.square(labels_placeholder - soft_model)))
    crossEnt = tf.scalar_summary("Cross Entropy",cross_entropy)
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
    accuracy = tf.scalar_summary("Accuracy",tf.reduce_mean(tf.cast(correct_answers,tf.float32)))
    precision_sum = tf.scalar_summary("Precision",precision)
    recall_sum = tf.scalar_summary("Recall",recall)
    f1_score = tf.scalar_summary("F1 Score", (2*precision*recall)/(precision+recall))
    summaries = tf.merge_all_summaries()


#Initalisation
sess=tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
train_writer = tf.train.SummaryWriter("Train", graph = sess.graph)
test_writer = tf.train.SummaryWriter("Test", graph = sess.graph)

#Saver
saver = tf.train.Saver(sharded=True)

if LOAD_MODEL:
    saver.restore(sess,"savedModels/model.ckpt")

for step in range(steps):
    if step % (data_size//BATCH_SIZE) == 0 :#shuffle dataset for each epoch
        data,labels = shuffle_data(data, labels)
    data_batch = get_batch(data, step)
    labels_batch = get_batch(labels, step)
    summary, _  = sess.run([summaries, train_op], feed_dict={sentence_placeholder:data_batch, labels_placeholder:labels_batch, drop_out_prob: KEEP_PROBABILITY})
    train_writer.add_summary(summary, step)
    print("Step "+ str(step+1) + " out of "+str(steps))
    
    if step%100 == 0:
        print(sess.run(cross_entropy, feed_dict = {sentence_placeholder:data_batch, labels_placeholder:labels_batch, drop_out_prob: KEEP_PROBABILITY}))
    
    if step%500 == 0:
        summary = sess.run(summaries, feed_dict = {sentence_placeholder:test_data, labels_placeholder:test_labels, drop_out_prob: 1})
        test_writer.add_summary(summary, step)
    #Model Saving
    if (step%1000==0 and SAVE_MODEL):
        save = saver.save(sess, "savedModels/model.ckpt")

export_path = "exportedModel"
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    init_op=init_op,
    named_graph_signatures = {
        "inputs": exporter.generic_signature({"sentences": sentence_placeholder, "dropout": drop_out_prob}),
        "outputs": exporter.generic_signature({"scores": soft_max, "classes": CLASS_NAMES})})
model_exporter.export(export_path,tf.constant(1), sess)
