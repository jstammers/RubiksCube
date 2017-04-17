from RecurrentNeuralNetwork.Network import *
from Config import *
from Cube import RubiksCube
import numpy as np
import os.path
from sklearn.model_selection import train_test_split
import pickle

def train(sess,state_list,move_list,rnn,cube):
     # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)



    for i in range(EPOCHS):
        train_data,valid_data,train_labels,valid_labels = train_test_split(state_list,move_list,train_size=TRAIN_FRAC)
        
        for j in range(int(len(train_labels)/MINIBATCH_SIZE)):
        #get the next state batch and predict their outcomes
            train_batch =  train_data[j:(j+1)*MINIBATCH_SIZE]
            train_move_batch = train_labels[j:(j+1)*MINIBATCH_SIZE]
          
            rnn.train(train_batch,train_move_batch)
        
            train_score = rnn.predict(train_data)
            valid_score = rnn.predict(valid_data)
            train_acc = sum(np.argmax(train_score,-1)==np.argmax(train_labels,-1))/len(train_score)
            valid_acc =  sum(np.argmax(valid_score,-1)==np.argmax(valid_labels,-1))/len(valid_score)
            cube.random_rotate(30)
            for i in range(TEST_STEPS):
                pred_move = rnn.predict(cube._cube.reshape(1,*cube._cube.shape))
                cube.rotation(np.argmax(pred_move))
                if np.mean(cube.score_similarity()) == 1:
                    break
            test_score = np.mean(cube.score_similarity())
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: train_acc,
                summary_vars[1]: valid_acc,
                summary_vars[2]: test_score
                })

            writer.add_summary(summary_str, j)
            writer.flush()
            print('| Training Acc.: %.4f' % train_acc, " | Episode", j, \
                              '| Valid Acc: %.4f' % valid_acc,"| Cube Score: %.4f" %test_score)

def main(_):
    with tf.Session() as sess:
        
        if GENERATE_DATA and not os.path.isfile(DATA_DIR+"/{0}Cube_Data.npz".format(N_CUBE)):
            state_list = []
            move_list = []
            for _ in range(10000):
                cube= RubiksCube(N_CUBE)
                for m in range(30):
                    cube.random_rotate()
                    state_list.append(cube._cube)
                    move = cube.move_list[-1]
                    #The direction of the last move must be reversed
                    rev_move = (move[0],move[1],-move[2])
                    move_list.append([rev_move == x for x in cube.rots])
            move_list = np.array(move_list)
            state_list = np.array(state_list)
            np.savez(DATA_DIR+"/{0}Cube_Data.npz".format(N_CUBE),states=state_list,moves=move_list)
        else:
            npfiles = np.load(DATA_DIR+"/{0}Cube_Data.npz".format(N_CUBE))
            move_list = npfiles["moves"]
            state_list = npfiles["states"]
                                        
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        cube = RubiksCube(N_CUBE)
        state_dim = cube._cube.shape
        action_dim = N_CUBE*N_CUBE*2

        rnn = RecurrentNetwork(sess, state_dim, action_dim,
                             LEARNING_RATE)


        train(sess, state_list,move_list, rnn,cube)

            
if __name__ == '__main__':
    tf.app.run()
    meta_graph_def = tf.train.export_meta_graph(filename=SUMMARY_DIR+"/rubiks_model.tfl")