# ===========================
#   Agent Training
# ===========================
from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import numpy as np
from Networks import *
from gym import wrappers
from Config import *
from CubeEnv import CubeEnv

def train(sess, env, actor, critic):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()
                
            # Added exploration noise
            #a = actor.predict(s) + (1. / (1. + i))
            a=actor.predict(s.reshape(1,*s.shape))
            a+=int((np.random.rand()-0.5)*actor.a_dim/(1.+i))
            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(s, np.reshape(a, (actor.a_dim,)), r,
                              terminal, s2)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(np.array(y_i), (MINIBATCH_SIZE,1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r


            if terminal:

                
                break
        summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

        writer.add_summary(summary_str, i)
        writer.flush()
        print('| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                      '| Qmax: %.4f' % (ep_ave_max_q / float(j)))


def main(_):
    with tf.Session() as sess:

        env = CubeEnv(N_CUBE)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, actor, critic)

        if GYM_MONITOR_EN:
            env.monitor.close()
if __name__ == '__main__':
    tf.app.run()
    meta_graph_def = tf.train.export_meta_graph(filename=SUMMARY_DIR+"/rubiks_model.tfl")