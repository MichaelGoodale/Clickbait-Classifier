#!/usr/bin/env python
from grpc.beta import implementations
import tensorflow as tf
import numpy as np
import nltk
from flask import g
import sqlite3
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

DATABASE = 'database/word-vec.db'

def grpc_get(server, in_sentence, SENTENCE_MAX, WORD_VECTOR_SIZE):
    host, port = server.split(":")
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    data = sql_lookup(in_sentence, SENTENCE_MAX, WORD_VECTOR_SIZE)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "clickbait"
    request.inputs["sentences"].CopyFrom(tf.contrib.util.make_tensor_proto(data, dtype = tf.float32))
    request.inputs["dropout"].CopyFrom(tf.contrib.util.make_tensor_proto(np.array(1), dtype=tf.float32))
    result = stub.Predict(request, 15.0)    
    scores = tf.contrib.util.make_ndarray(result.outputs["scores"])[0].tolist() 
    classes = tf.contrib.util.make_ndarray(result.outputs["classes"]).astype(str).tolist() 
    return scores, classes
    
def sql_lookup(in_sentence, SENTENCE_MAX, WORD_VECTOR_SIZE):
    cur = get_db().cursor()
    sentence = in_sentence.lower().strip()
    wordArray = nltk.word_tokenize(sentence)
    sentence = np.zeros((SENTENCE_MAX,WORD_VECTOR_SIZE))
    wordArray = wordArray[:SENTENCE_MAX]
    for i, word in enumerate(wordArray):
        word_string = (word,)
        cur.execute("SELECT * FROM glove WHERE field1=?", word_string)
        row = cur.fetchone()
        if row is None:
            sentence[i, :] = np.ones(WORD_VECTOR_SIZE)
        else:
            sentence[i] = np.array(row[1:]).astype(np.float32)
    data = sentence.flatten()
    return data[np.newaxis,np.newaxis,:,np.newaxis]
    
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
       db = g._database = sqlite3.connect(DATABASE)
    return db
