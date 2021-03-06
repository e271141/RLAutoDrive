import time
import numpy as np
import json
import os

import tensorflow as tf
import threading

# Prevent TensorFlow from allocating the entire GPU at the start of the program.
# Otherwise, AirSim will sometimes refuse to launch, as it will be unable to 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

# A wrapper class for the DQN model
class ACModel():
    def __init__(self, weights_path, train_conv_layers):
        self.__angle_values = [-1, -0.5, 0, 0.5, 1]

        self.__nb_actions = 5
        self.__gamma = 0.99

        #Define the model

        self.sess = session
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Build evaluation net
            self.act = action
            self.q = self.__bulid_net(scope='eval_net', trainable=True)

            # Build target net
            self.q_ = self.__bulid_net(scope='target_net', trainable=False)

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = Reward + self.gamma*self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

        self.replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def learn(self, pre_states, post_states):

        # We only have labels for the action that the agent actually took.
        # To prevent the model from training the other actions, figure out what the model currently predicts for each input.
        # Then, the gradients with respect to those outputs will always be zero.
        with self.__action_context.as_default():
            labels = self.__action_model.predict([pre_states], batch_size=32)
        
        # Find out what the target model will predict for each post-decision state.
        with self.__target_context.as_default():
            q_futures = self.__target_model.predict([post_states], batch_size=32)

        # Apply the Bellman equation
        q_futures_max = np.max(q_futures, axis=1)
        q_labels = (q_futures_max * is_not_terminal * self.__gamma) + rewards
        
        # Update the label only for the actions that were actually taken.
        for i in range(0, len(actions), 1):
            labels[i][actions[i]] = q_labels[i]


        q_value, q_next = self.sess.run([self.q_eval, self.q_next],
                    feed_dict={
                        self.eval_inputs: batches,
                        self.q_target: q_target})

        error, _ = self.sess.run([self.train_op, self.loss], 
                    feed_dict={
                        self.eval_inputs: batches,
                        self.q_target: q_target})

    def __bulid_net(self, scope, trainable):

        with tf.variable_scope(scope):
        
            self.eval_inputs = tf.placeholder(tf.uint8, shape=(59,255,3), name="eval_input")
            self.q_target = tf.placeholder(tf.uint8, name="Q_target")

            conv_eval_1 = __conv_block(self.inputs, filter=16, kernel=(3,3), train_conv_layers)
            conv_eval_2 = __conv_block(conv_eval_1, filter=32, kernel=(3,3), train_conv_layers)
            conv_eval_3 = __conv_block(conv_eval_2, filter=32, kernel=(3,3), train_conv_layers)
            conv_eval_4 = __conv_block(conv_eval_3, filter=32, kernel=(3,3), train_conv_layers)
            flatten = tf.layers.flatten(conv_eval_4)
            self.eval_net_output = tf.layers.dense(
                units=6,
                inputs=flatten, 
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name="output"
            )



    # An block of layer with:
    # convolution, batch_normalization, relu activation
    def __conv_block(self, inputs, filter, kernel, freeze):
        conv = tf.layers.Conv2D(
            inputs=self.inputs,
            filters=filter, 
            kernel_size=kernel, 
            padding="same",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0.1),
            trainable=trainning
        )
        mean_conv, var_conv = tf.nn.moments(conv, axes=[0])    # Compute mean & variance for conv1
        bn = tf.nn.batchnormalization(x=conv, mean=mean_conv, variance=var_conv) # Normalization on batch only
        conv_layer_block = tf.nn.relu(bn)

        return conv_layer_block
        
        # If we are using pretrained weights for the conv layers, load them and verify the first layer.
        if (weights_path is not None and len(weights_path) > 0):
            print('Loading weights from my_model_weights.h5...')
            print('Current working dir is {0}'.format(os.getcwd()))
            self.__action_model.load_weights(weights_path, by_name=True)
            
            print('First layer: ')
            w = np.array(self.__action_model.get_weights()[0])
            print(w)
        else:
            print('Not loading weights')

        # Set up the target model. 
        # This is a trick that will allow the model to converge more rapidly.
        self.__action_context = tf.get_default_graph()
        self.__target_model = clone_model(self.__action_model)

        self.__target_context = tf.get_default_graph()
        self.__model_lock = threading.Lock()
            
    def update_critic(self):
        with self.__target_context.as_default():
            self.__target_model.set_weights([np.array(w, copy=True) for w in self.__action_model.get_weights()])
    
            
    # Given a set of training data, trains the model and determine the gradients.
    # The agent will use this to compute the model updates to send to the trainer
    def get_gradient_update_from_batches(self, batches):
        pre_states = np.array(batches['pre_states'])
        post_states = np.array(batches['post_states'])
        rewards = np.array(batches['rewards'])
        actions = list(batches['actions'])
        is_not_terminal = np.array(batches['is_not_terminal'])
        
        # For now, our model only takes a single image in as input. 
        # Only read in the last image from each set of examples
        pre_states = pre_states[:, 3, :, :, :]
        post_states = post_states[:, 3, :, :, :]
        
        labels = self.critic.learn()

        # Perform a training iteration.
        with self.__action_context.as_default():
            self.actor.learn([pre_states], labels, epochs=1, batch_size=32, verbose=1)
        
        print('END GET GRADIENT UPDATE DEBUG')

    # Performs a state prediction given the model input
    def predict_state(self, observation):
        if (type(observation) == type([])):
            observation = np.array(observation)
        
        # Our model only predicts on a single state.
        # Take the latest image
        observation = observation[3, :, :, :]
        observation = observation.reshape(1, 59,255,3)
        with self.__action_context.as_default():
            predicted_qs = self.actor.choose_action([observation])

        # Select the action with the highest Q value
        predicted_state = np.argmax(predicted_qs)
        return (predicted_state, predicted_qs[0][predicted_state])

    # Convert the current state to control signals to drive the car.
    # As we are only predicting steering angle, we will use a simple controller to keep the car at a constant speed
    def state_to_control_signals(self, state, car_state):
        if car_state.speed > 9:
            return (self.__angle_values[state], 0, 1)
        else:
            return (self.__angle_values[state], 1, 0)

    # Gets a random state
    # Used during annealing
    def get_random_state(self):
        return np.random.randint(low=0, high=(self.__nb_actions) - 1)

# DQN
class Critic(object):
    def __init(self, weights, session, train_conv_layers):
