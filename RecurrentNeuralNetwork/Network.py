import tflearn
import tensorflow as tf
from Config import *

class RecurrentNetwork(object):
    def __init__(self,session,state_dim,action_dim,learning_rate):
        self.sess = session
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate

        # Network
        self.inputs, self.labels, self.out,self.loss = self.create_network()

        self.network_params = tf.trainable_variables()
        
        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.num_trainable_vars = len(
            self.network_params)
        
    def create_network(self): 
        inputs = tflearn.input_data(shape=[None, self.s_dim[0],self.s_dim[1]])
  
        net = tflearn.fully_connected(inputs, 400, activation='relu',regularizer="L2",weight_decay=L2_REG)
        if DROPOUT:
            net = tflearn.dropout(net,KEEP_PROB)
        net = tflearn.fully_connected(net, 300, activation='relu',regularizer="L2",weight_decay=L2_REG)
        if DROPOUT:
            net = tflearn.dropout(net,KEEP_PROB)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-1, maxval=1)
        out = tflearn.fully_connected(net, self.a_dim, activation='softmax', weights_init=w_init,regularizer="L2",weight_decay=L2_REG)
        labels = tflearn.input_data(shape=[None,self.a_dim])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=out))
        return inputs, labels, out,loss
    
    def configure(self):
        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
    def train(self, inputs, labels):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.labels:labels
        })
        
    def validate(self,inputs,labels):
        return self.sess.run(self.out,feed_dict={self.inputs:inputs,self.labels:labels})
        
    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })
        
    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    train_accuracy = tf.Variable(0.)
    tf.summary.scalar("Training_Accuracy", train_accuracy)
    validation_accuracy = tf.Variable(0.)
    tf.summary.scalar("Validation_Accuracy", validation_accuracy)
    test_score = tf.Variable(0.)
    tf.summary.scalar("Test_Score", test_score)

    summary_vars = [train_accuracy, validation_accuracy,test_score]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
    