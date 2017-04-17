from RecurrentNeuralNetwork import Train
import tensorflow as tf

def main(_):
    Train.main(_)

if __name__ == '__main__':
    tf.app.run()
    meta_graph_def = tf.train.export_meta_graph(filename=SUMMARY_DIR+"/rubiks_rnn.tfl")