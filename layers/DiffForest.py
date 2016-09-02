from keras import backend as K
from keras.engine.topology import Layer
from keras import initializations
import numpy as np
import tensorflow as tf

class DiffForest(Layer):
    def __init__(self, output_classes, n_trees=5, n_depth=3,
            d_init=None, l_init=None, randomize_training=0,
            name='diff_forest', **kwargs):

        self.output_classes = output_classes
        self.n_trees = n_trees
        self.n_depth = n_depth
        self.randomize_training = randomize_training
        self.name = name

        def norm(scale):
          return lambda shape, name=None: initializations.uniform(shape, scale=scale, name=name)

        #Not clear if these are generally good initializations
        #Or if they are just good for MNIST

        if d_init is None:
          self.d_init = norm(1)
        else:
          self.d_init = initializations.get(init)

        if l_init is None:
          self.l_init = norm(2)
        else:
          self.l_init = initializations.get(init)

        super(DiffForest, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]

        #Per tree
        N_DECISION = (2 ** (self.n_depth)) - 1  # Number of decision nodes
        N_LEAF  = 2 ** (self.n_depth + 1)  # Number of leaf nodes

        if self.randomize_training:
            #Construct a mask that lets N trees get trained per minibatch
            train_mask = np.zeros(self.n_trees, dtype=np.float32)
            for i in xrange(self.randomize_training):
                train_mask[i] = 1
            self.random_mask = tf.random_shuffle(tf.constant(train_mask))

        self.w_d_ensemble = []
        self.w_l_ensemble = []
        self.trainable_weights = []
        for i in xrange(self.n_trees):
            decision_weights = self.d_init((input_dim, N_DECISION), name=self.name+"_tree"+i+"_dW")
            leaf_distributions = self.l_init((N_LEAF, self.output_classes), name=self.name+"_tree"+i+"_lW")

            self.trainable_weights.append(decision_weights)
            self.trainable_weights.append(leaf_distributions)

            if self.randomize_training:
                do_gradient = self.random_mask[i]
                no_gradient = 1 - do_gradient
                
                #This should always allow inference, but block gradient flow when do_gradient = 0 
                decision_weights = do_gradient * decision_weights + no_gradient * tf.stop_gradient(decision_weights)

                leaf_distributions = do_gradient * leaf_distributions + no_gradient * tf.stop_gradient(leaf_distributions)

            self.w_d_ensemble.append(decision_weights)
            self.w_l_ensemble.append(leaf_distributions)

    def call(self, x, mask=None):
        N_DECISION = (2 ** (self.n_depth)) - 1  # Number of decision nodes
        N_LEAF  = 2 ** (self.n_depth + 1)  # Number of leaf nodes

        flat_decision_p_e = []
        leaf_p_e = []
        for w_d, w_l in zip(self.w_d_ensemble, self.w_l_ensemble):

            decision_p = K.sigmoid((K.dot(x, w_d)))
            leaf_p = K.softmax(w_l)

            decision_p_comp = 1 - decision_p

            decision_p_pack = K.concatenate([decision_p, decision_p_comp])

            flat_decision_p_e.append(decision_p_pack)
            leaf_p_e.append(leaf_p)

        #Construct tiling pattern for decision probability matrix
        #Could be done in TF, but I think it's better statically
        tiling_pattern = np.zeros((N_LEAF, self.n_depth), dtype=np.int32)
        comp_offset = N_DECISION
        dec_idx = 0
        for n in xrange(self.n_depth):
            j = 0
            for depth_idx in xrange(2**n):
                repeat_times = 2 ** (self.n_depth - n)
                for _ in xrange(repeat_times):
                    tiling_pattern[j][n] = dec_idx 
                    j = j + 1

                for _ in xrange(repeat_times):
                    tiling_pattern[j][n] = comp_offset + dec_idx 
                    j = j + 1

                dec_idx = dec_idx + 1

        flat_pattern = tiling_pattern.flatten()

        # iterate over each tree
        tree_ret = None
        for flat_decision_p, leaf_p in zip(flat_decision_p_e, leaf_p_e):
            flat_mu = tf.transpose(tf.gather(tf.transpose(flat_decision_p), flat_pattern))
            
            batch_size = tf.shape(flat_decision_p)[0]
            shape = tf.pack([batch_size, N_LEAF, self.n_depth])

            mu = K.reshape(flat_mu, shape)
            leaf_prob = K.prod(mu, [2])
            prob_label = K.dot(leaf_prob, leaf_p)

            if tree_ret is None:
              tree_ret = prob_label
            else:
              tree_ret = tree_ret + prob_label

        return tree_ret/self.n_trees

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_classes)