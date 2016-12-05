#!/usr/bin/env python
from grpc.beta import implementations
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string("server", "localhost:9000",
                           "PredictionService host:port")
tf.app.flags.DEFINE_string("sentence", "", "Sentence that is to be evaluated")
FLAGS = tf.app.flags.FLAGS
SENTENCE_MAX = 30
WORD_VECTOR_SIZE = 50

def main(_):
    host, port = FLAGS.server.split(":")
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    vectors_df = pd.read_csv("GloVeData/glove.6B.50d.txt",sep=" ",header=None,quoting=3)
    vectors_df = vectors_df.set_index([0])
    EvalSentence = FLAGS.sentence
    EvalSentence = EvalSentence.lower().strip()
    wordArray = nltk.word_tokenize(EvalSentence)
    sentence = np.zeros((SENTENCE_MAX,WORD_VECTOR_SIZE))
    
    for i, word in enumerate(wordArray):
        if word in vectors_df.index:
            sentence[i,:]=vectors_df.loc[word, :].values
        else:
            sentence[i,:]=np.ones(WORD_VECTOR_SIZE)
    
    data = sentence.flatten()
    data = data[np.newaxis,np.newaxis,:,np.newaxis]
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "clickbait"
    request.inputs["sentences"].CopyFrom(tf.contrib.util.make_tensor_proto(data, dtype = tf.float32))
    request.inputs["dropout"].CopyFrom(tf.contrib.util.make_tensor_proto(np.ones([]), dtype=tf.float32))
    result = stub.Predict(request, 15.0)
    print(result)

if __name__ == "__main__":
    tf.app.run()
