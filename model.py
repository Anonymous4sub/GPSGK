
import tensorflow as tf
import numpy as np

from layers import glorot, Layer, NeuralNet, InferenceNet, GCNAggregator, RFFAggregator, Res_bolck



def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


class StructAwareGP(object):

    def __init__(self, placeholders, input_features, feature_dim, n_samples, latent_layer_units, output_dim,
                transform_feature=False, node_neighbors=None, linear_layer=False, lambda1=1.0, lambda2 = 1.0, sample_size=5,
                dropout=0., bias=False, act=tf.nn.relu, weight_decay=0.0, lr=0.001, name="sagp", n_neighbors=10, node_neighbors_neg=None, **kwargs):
        """
        feature_dim: dimension of transformed feature, only used when transform_feture is True
        """
        super(StructAwareGP, self).__init__(**kwargs)

        self.batch = placeholders['nodes']
        self.Y = placeholders['Y']
        self.label_mask = placeholders['label_mask']
        self.localSim = placeholders['localSim']
        # self.globalSim = placeholders['globalSim']
        self.batch_size = placeholders["batch_size"]

        self.input_dim = input_features.shape[1]
        self.n_classes = placeholders['Y'].get_shape().as_list()[1]

        self.input_features = input_features
        self.n_samples = n_samples
        self.latent_layer_units = latent_layer_units
        self.output_dim = output_dim

        self.transform_feature = transform_feature
        self.linear_layer = linear_layer

        self.lambda1 = lambda1
        self.lambda2 = lambda2 
        self.sample_size = sample_size
        self.dropout = dropout
        self.act = act 
        self.weight_decay = weight_decay
        self.lr = lr

        # feature learning
        with tf.variable_scope("feature_mapping"):

            if self.transform_feature:
                self.feature_dim = feature_dim
                """
                hidden_layers = NeuralNet(self.input_dim, [self.feature_dim, self.feature_dim], self.dropout, self.act)
                self.feature = hidden_layers(tf.nn.embedding_lookup(self.input_features, self.batch))
                """
                # trans_input_features = NeuralNet(self.input_dim, [256])(self.input_features)
                self.feature = GraphConvolution(self.input_features, node_neighbors, [self.feature_dim*2, self.feature_dim], [25, 20], self.batch_size,
                                                dropout=self.dropout, act=self.act)(self.batch)
            else:
                self.feature_dim = self.input_dim
                self.feature = Constant(self.input_dim)(tf.nn.embedding_lookup(self.input_features, self.batch))
        

        # random Fourier feature
        with tf.variable_scope("ImplicitKernelNet"):

            # self.implicitkernelnet = InferenceNet(1, self.feature_dim, [32], dropout=0.0, act=self.act)
            self.implicitkernelnet = InferenceNet(1, self.feature_dim, self.latent_layer_units, dropout=self.dropout, act=self.act)
            
            self.context_weight = glorot([self.n_samples, self.n_samples])
            #self.Omega_mu = glorot([1, self.feature_dim])
            #self.Omega_logstd = glorot([1, self.feature_dim])
        
        self.epsilon = np.random.normal(0.0, 1.0, [self.n_samples, 1]).astype(np.float32)
        # self.epsilon = tf.random.normal(shape=[self.n_samples, 1], dtype=tf.float32)
        self.eps = np.random.normal(0.0, 1.0, [self.sample_size, self.n_samples, self.feature_dim]).astype(np.float32)
        # self.eps = np.random.normal(0.0, 1.0, [self.n_samples, self.feature_dim]).astype(np.float32)
        self.b = np.random.uniform(0.0, 2*np.pi, [1, self.n_samples]).astype(np.float32)

        # Bayesian linear regression
        with tf.variable_scope("posterior"):
            # self.posterior = InferenceNet(self.feature_dim, self.n_samples*self.output_dim, latent_layer_units, dropout=self.dropout, act=self.act)
            # self.posterior = InferenceNet(self.feature_dim + self.n_classes, self.n_samples, latent_layer_units, dropout=self.dropout, act=self.act)

            self.W_mu = glorot([self.n_classes, self.n_samples], name='out_weights_mu')
            self.W_logstd = glorot([self.n_classes, self.n_samples], name='out_weights_logstd')

        """
        if self.linear_layer:
            # self.W = glorot([output_dim, self.n_classes])
            self.linear_W = glorot([self.output_dim, self.n_classes])
        else:
            self.linear_W = tf.eye(self.n_classes)
        """

        self._build_graph()
    

    def _build_graph(self):

        # self.feature = self.feature_layer(self.batch)

        # obtain implicit random features
        Omega_mu, Omega_logstd = self.implicitkernelnet(self.epsilon) # n_samples, feature_dim
        # print(Omega_mu)
        # print(Omega_logstd)
        Omega = Omega_mu + self.eps * tf.math.exp(Omega_logstd)  # sample_size, n_samples, feature_dim
        self.Omega = tf.reduce_mean(Omega, axis=0)
        # self.Omega = Omega

        transform = tf.matmul(self.feature, self.Omega, transpose_b=True) # N, n_samples
        transform = np.sqrt(2. / self.n_samples) * tf.math.cos(2*np.pi*transform + self.b)
        self.kernelfeatures = tf.cast(transform, tf.float32)


        # obtain parameters of the linear mapping
        """
        feature_class = self.get_feature_label(self.feature)  # n_classes, feature_dim + n_classes
        self.W_mu, self.W_logstd = self.posterior(feature_class)  # n_classes, n_samples
        """
        u = np.random.normal(0.0, 1.0, [self.sample_size, self.n_classes, self.n_samples])
        W = self.W_mu + u * tf.math.exp(self.W_logstd)
        W = tf.reduce_mean(W, axis=0)

        output = tf.matmul(self.kernelfeatures, W, transpose_b=True)

        """
        self.W_mu, self.W_logstd = self.posterior(tf.reduce_mean(self.feature, axis=0, keepdims=True))
        self.W_mu = tf.reshape(tf.squeeze(self.W_mu, [0]), [self.n_samples, self.output_dim])
        self.W_logstd = tf.reshape(tf.squeeze(self.W_logstd, [0]), [self.n_samples, self.output_dim])
        
        u = np.random.normal(0.0, 1.0, [self.n_samples, self.output_dim])
        W = self.W_mu + u * tf.math.exp(self.W_logstd)

        output = tf.matmul(self.kernelfeatures, W)
        """

        # self.logits = tf.matmul(output, self.linear_W)
        self.logits = output

        # ============================= construct loss =====================================

        self.reconstruct_loss = masked_softmax_cross_entropy(self.logits, self.Y, self.label_mask)

        scale = 1. / tf.cast(tf.reduce_sum(self.label_mask), tf.float32)
        self.kl = scale * self.obtain_prior_KL()

        """
        featureSim = tf.matmul(self.kernelfeatures, self.kernelfeatures, transpose_b=True)
        # self.sim = self.lambda1 * tf.reduce_mean(tf.multiply(featureSim, self.localSim + 0.001)) + self.lambda2 * tf.reduce_mean(tf.multiply(featureSim, self.globalSim))

        # self.sim = self.lambda1 * tf.nn.l2_loss(featureSim - self.localSim) + self.lambda2 * tf.nn.l2_loss(featureSim - self.globalSim)
        # f = lambda x: tf.reduce_sum(tf.reduce_mean(tf.square(x), axis=1))
        f = lambda x: tf.reduce_mean(tf.square(x))
        self.sim = self.lambda1 * f(featureSim - self.localSim) + self.lambda2 * f(featureSim - self.globalSim)
        """

        """
        kernelfeatures_label = tf.boolean_mask(self.kernelfeatures, self.label_mask)
        label = tf.boolean_mask(self.Y, self.label_mask)

        # sim_contex = tf.matmul(kernelfeatures_label, kernelfeatures_label, transpose_b=True) # n_label, n_label
        context = tf.matmul(kernelfeatures_label, self.context_weight)
        sim_contex = tf.matmul(kernelfeatures_label, context, transpose_b=True) # n_label, n_label
        
        mask_label = tf.cast(tf.matmul(label, label, transpose_b=True), tf.bool)
        pos = tf.boolean_mask(sim_contex, mask_label)
        neg = tf.boolean_mask(sim_contex, tf.math.logical_not(mask_label))
        # scale_neg = tf.cast(tf.shape(neg)[0], tf.float32) / tf.cast(tf.shape(pos)[0], tf.float32)
        sim_label = -1 * tf.reduce_mean(tf.math.log(1e-6 + tf.nn.sigmoid(pos))) - tf.reduce_mean(tf.math.log(1e-6 + tf.nn.sigmoid(-1 * neg)))
        """

        context = tf.matmul(self.kernelfeatures, self.context_weight)
        sim_contex = tf.matmul(self.kernelfeatures, context, transpose_b=True)
        mask_not_neighbor = tf.equal(self.localSim, 0.0)
        # pos_neighbor = tf.boolean_mask(sim_contex, tf.math.logical_not(mask_not_neighbor))
        # neg_neighbor = tf.boolean_mask(sim_contex, mask_not_neighbor)
        pos_neighbor = tf.nn.dropout(tf.boolean_mask(sim_contex, tf.math.logical_not(mask_not_neighbor)), keep_prob=0.5)
        neg_neighbor = tf.nn.dropout(tf.boolean_mask(sim_contex, mask_not_neighbor), keep_prob=0.5)
        sim_neighbor = -1 * tf.reduce_mean(tf.math.log(1e-6 + tf.nn.sigmoid(pos_neighbor))) - tf.reduce_mean(tf.math.log(1e-6 + tf.nn.sigmoid(-1 * neg_neighbor)))

        # self.sim = self.lambda1 * sim_label + self.lambda2 * sim_neighbor
        self.sim = self.lambda2 * sim_neighbor

        """
        self.sim = self.lambda1 * sim_label
        """
        """
        featureSim_label = tf.boolean_mask(tf.boolean_mask(featureSim, self.label_mask), self.label_mask, axis=1)
        Y_label =tf.boolean_mask(self.Y, self.label_mask)
        labelSim = tf.matmul(Y_label, Y_label, transpose_b=True)
        self.sim_label = 0.001 * tf.reduce_mean(tf.multiply(featureSim_label, labelSim))
        """

        # l2_loss = tf.nn.l2_loss(self.feature_layer.get_vars()[0])
        fm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="feature_mapping")
        kn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ImplicitKernelNet")
        ps_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="posterior")
        tf_vars = fm_vars + kn_vars

        l2_loss = 0
        for var in tf_vars:
            l2_loss += tf.nn.l2_loss(var)
        self.l2_loss = self.weight_decay * l2_loss

        # ====================================================================================

        # self.loss = self.reconstruct_loss + self.kl - self.sim + self.l2_loss
        self.loss = self.reconstruct_loss + self.kl + self.sim + self.l2_loss
        # self.loss = self.reconstruct_loss + self.kl + self.l2_loss
        
        # ============================ joint updating ========================================
        
        self.opt_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        #======================================================================================

        # ============================== iterative updating ==================================

        self.loss_e = self.reconstruct_loss + self.kl + self.l2_loss
        # self.loss_m = self.reconstruct_loss - self.sim + self.l2_loss
        self.loss_m = self.reconstruct_loss + self.sim + self.l2_loss
        # self.loss_m = self.reconstruct_loss + self.l2_loss


        self.optimizer_e = tf.train.AdamOptimizer(self.lr)
        grads_and_vars_e = self.optimizer_e.compute_gradients(self.loss_e, var_list= fm_vars+ ps_vars)
        clipped_grads_and_vars_e = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                                    for grad, var in grads_and_vars_e]
        self.opt_step_e = self.optimizer_e.apply_gradients(clipped_grads_and_vars_e)

        self.optimizer_m = tf.train.AdamOptimizer(self.lr)
        grads_and_vars_m = self.optimizer_m.compute_gradients(self.loss_m, var_list= fm_vars + kn_vars)
        clipped_grads_and_vars_m = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                                    for grad, var in grads_and_vars_m]
        self.opt_step_m = self.optimizer_m.apply_gradients(clipped_grads_and_vars_m)

        # self.opt_step_kernel = tf.train.AdamOptimizer(self.lr).minimize(self.sim, var_list= kn_vars)

        #=====================================================================================


        # accuracy
        self.accuracy = masked_accuracy(self.logits, self.Y, self.label_mask)



    def set_lambda(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def obtain_prior_KL(self):
        # return KL divergence of W

        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.math.square(self.W_mu) + tf.math.square(tf.math.exp(self.W_logstd)) \
                                    - 2*self.W_logstd - 1, axis=1))

        

    def get_feature_label(self, feature):

        feature_train = tf.boolean_mask(feature, self.label_mask)
        label_train = tf.boolean_mask(self.Y, self.label_mask)
        supprot_mean_feature_list = []

        for c in range(self.n_classes):

            class_mask = tf.equal(tf.argmax(label_train, axis=1), c)

            class_feature = tf.boolean_mask(feature_train, class_mask)
            class_label = tf.boolean_mask(label_train, class_mask)

            # Pool across dimensions
            feature_label = tf.concat([class_feature, class_label], axis=1)
            nu = tf.expand_dims(tf.reduce_mean(feature_label, axis=0), axis=0)
            supprot_mean_feature_list.append(nu)

        support_mean_features = tf.concat(supprot_mean_feature_list, axis=0)

        return support_mean_features


class StructAwareGP_Inductive(object):

    def __init__(self, placeholders, input_features, feature_dim, n_samples, latent_layer_units, output_dim,
                transform_feature=False, node_neighbors=None, linear_layer=False, lambda1=1.0, lambda2 = 1.0, sample_size=5,
                dropout=0., bias=False, act=tf.nn.relu, weight_decay=0.0, lr=0.001, name="sagp", 
                sigmoid_loss=False, n_neighbors=10, node_neighbors_neg=None, **kwargs):

        super(StructAwareGP_Inductive, self).__init__(**kwargs)

        self.batch = placeholders['nodes']
        self.Y = placeholders['Y']
        self.label_mask = placeholders['label_mask']
        self.localSim = placeholders['localSim']
        self.batch_size = placeholders["batch_size"]

        self.input_dim = input_features.shape[1]
        self.n_classes = placeholders['Y'].get_shape().as_list()[1]

        self.input_features = input_features
        self.n_samples = n_samples
        self.latent_layer_units = latent_layer_units
        self.output_dim = output_dim

        self.transform_feature = transform_feature
        self.linear_layer = linear_layer

        self.lambda1 = lambda1
        self.lambda2 = lambda2 
        self.sample_size = sample_size
        self.dropout = dropout
        self.act = act 
        self.weight_decay = weight_decay
        self.lr = lr
        self.sigmoid_loss = sigmoid_loss

        # feature learning
        with tf.variable_scope("feature_mapping"):

            if self.transform_feature:
                self.feature_dim = feature_dim
                self.feature = GraphConvolution(self.input_features, node_neighbors, [self.feature_dim*2, self.feature_dim], [25, 20], self.batch_size,
                                                dropout=self.dropout, act=self.act)(self.batch)
            else:
                self.feature_dim = self.input_dim
                self.feature = Constant(self.input_dim)(tf.nn.embedding_lookup(self.input_features, self.batch))
        

        # random Fourier feature
        with tf.variable_scope("ImplicitKernelNet"):

            self.implicitkernelnet = InferenceNet(1, self.feature_dim, self.latent_layer_units, dropout=self.dropout, act=self.act)
            
            self.context_weight = glorot([self.n_samples, self.n_samples])
            
        self.epsilon = np.random.normal(0.0, 1.0, [self.n_samples, 1]).astype(np.float32)
        self.eps = np.random.normal(0.0, 1.0, [self.sample_size, self.n_samples, self.feature_dim]).astype(np.float32)
        self.b = np.random.uniform(0.0, 2*np.pi, [1, self.n_samples]).astype(np.float32)

        # Bayesian linear regression
        with tf.variable_scope("posterior"):
            self.W_mu = glorot([self.output_dim, self.n_samples], name='out_weights_mu')
            self.W_logstd = glorot([self.output_dim, self.n_samples], name='out_weights_logstd')

            if self.linear_layer:
                self.linear_W = glorot([self.output_dim, self.n_classes])
            else:
                self.linear_W = tf.eye(self.n_classes)
        

        self._build_graph()
    

    def _build_graph(self):


        # obtain implicit random features
        Omega_mu, Omega_logstd = self.implicitkernelnet(self.epsilon) # n_samples, feature_dim
        Omega = Omega_mu + self.eps * tf.math.exp(Omega_logstd)  # sample_size, n_samples, feature_dim
        self.Omega = tf.reduce_mean(Omega, axis=0)

        transform = tf.matmul(self.feature, self.Omega, transpose_b=True) # N, n_samples
        transform = np.sqrt(2. / self.n_samples) * tf.math.cos(2*np.pi*transform + self.b)
        self.kernelfeatures = tf.cast(transform, tf.float32)


        # obtain parameters of the linear mapping
        u = np.random.normal(0.0, 1.0, [self.sample_size, self.n_classes, self.n_samples])
        W = self.W_mu + u * tf.math.exp(self.W_logstd)
        W = tf.reduce_mean(W, axis=0)

        output = tf.matmul(self.kernelfeatures, W, transpose_b=True)

        self.logits = tf.matmul(output, self.linear_W)
        # self.logits = output

        # ============================= construct loss =====================================
        if not self.sigmoid_loss:
            self.reconstruct_loss = masked_softmax_cross_entropy(self.logits, self.Y, self.label_mask)
        else:
            self.reconstruct_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        scale = 1. / tf.cast(tf.reduce_sum(self.label_mask), tf.float32)
        self.kl = scale * self.obtain_prior_KL()


        context = tf.matmul(self.kernelfeatures, self.context_weight)
        sim_contex = tf.matmul(self.kernelfeatures, context, transpose_b=True)
        mask_not_neighbor = tf.equal(self.localSim, 0.0)
        pos_neighbor = tf.nn.dropout(tf.boolean_mask(sim_contex, tf.math.logical_not(mask_not_neighbor)), keep_prob=0.5)
        neg_neighbor = tf.nn.dropout(tf.boolean_mask(sim_contex, mask_not_neighbor), keep_prob=0.5)
        sim_neighbor = -1 * tf.reduce_mean(tf.math.log(1e-6 + tf.nn.sigmoid(pos_neighbor))) - tf.reduce_mean(tf.math.log(1e-6 + tf.nn.sigmoid(-1 * neg_neighbor)))

        # self.sim = self.lambda1 * sim_label + self.lambda2 * sim_neighbor
        self.sim = self.lambda2 * sim_neighbor

        fm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="feature_mapping")
        kn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ImplicitKernelNet")
        ps_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="posterior")
        tf_vars = fm_vars + kn_vars

        l2_loss = 0
        for var in tf_vars:
            l2_loss += tf.nn.l2_loss(var)
        self.l2_loss = self.weight_decay * l2_loss

        # ====================================================================================

        # self.loss = self.reconstruct_loss + self.kl - self.sim + self.l2_loss
        self.loss = self.reconstruct_loss + self.kl + self.sim + self.l2_loss
        # self.loss = self.reconstruct_loss + self.kl + self.l2_loss

        # ============================== iterative updating ==================================

        self.loss_e = self.reconstruct_loss + self.kl + self.l2_loss
        self.loss_m = self.reconstruct_loss + self.sim + self.l2_loss


        self.optimizer_e = tf.train.AdamOptimizer(self.lr)
        grads_and_vars_e = self.optimizer_e.compute_gradients(self.loss_e, var_list= fm_vars+ ps_vars)
        clipped_grads_and_vars_e = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                                    for grad, var in grads_and_vars_e]
        self.opt_step_e = self.optimizer_e.apply_gradients(clipped_grads_and_vars_e)

        self.optimizer_m = tf.train.AdamOptimizer(self.lr)
        grads_and_vars_m = self.optimizer_m.compute_gradients(self.loss_m, var_list= fm_vars + kn_vars)
        clipped_grads_and_vars_m = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                                    for grad, var in grads_and_vars_m]
        self.opt_step_m = self.optimizer_m.apply_gradients(clipped_grads_and_vars_m)

        #=====================================================================================
        
        # accuracy
        if not self.sigmoid_loss:
            self.accuracy = masked_accuracy(self.logits, self.Y, self.label_mask)


    def obtain_prior_KL(self):
        # return KL divergence of W

        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.math.square(self.W_mu) + tf.math.square(tf.math.exp(self.W_logstd)) \
                                    - 2*self.W_logstd - 1, axis=1))




class GCN(object):

    def __init__(self, placeholders, dropout=0.0, act=tf.nn.relu, weight_decay=0.0):

        self.inputs = placeholders['X']
        self.label = placeholders['Y']
        self.label_mask = placeholders['label_mask']
        self.localSim = placeholders['localSim']
        self.globalSim = placeholders['globalSim']

        self.input_dim = placeholders['X'].get_shape().as_list()[1]
        self.n_classes = placeholders['Y'].get_shape().as_list()[1]


        with tf.variable_scope("gcn_layers"):
            layer1 = GraphConvolution(self.input_dim, 16, self.localSim, dropout=dropout, act=act)
            layer2 = GraphConvolution(16, self.n_classes, self.localSim, dropout=dropout, act=lambda x:x)

        hidden = layer1(self.inputs)
        logits = layer2(hidden)

        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gcn_layers")
        l2 = weight_decay * tf.nn.l2_loss(layer1.get_vars()[0])

        nllh = masked_softmax_cross_entropy(logits, self.label, self.label_mask)

        self.loss = l2 + nllh 

        self.opt_step = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        self.accuracy = masked_accuracy(logits, self.label, self.label_mask)



class GraphConvolution(object):

    def __init__(self, input_features, node_neighbors, dims, num_samples, batch_size, dropout=0.0, bias=False, act=tf.nn.relu):

        self.input_features = input_features
        self.node_neighbors = node_neighbors
        self.dims = [self.input_features.shape[1]]
        # self.dims = [self.input_features.get_shape().as_list()[1]]
        self.dims.extend(dims)
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.dropout = dropout
        self.act = act

        # sample neighborhood nodes for aggregation
        self.aggregators = []

        for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:]):
            self.aggregators.append(GCNAggregator(in_dim, out_dim, dropout=self.dropout, bias=False, act=self.act))
        """
        for in_dim, out_dim in zip(self.dims[:-2], self.dims[1:-1]):
            self.aggregators.append(GCNAggregator(in_dim, out_dim, dropout=self.dropout, bias=False, act=self.act))
        self.aggregators.append(GCNAggregator(self.dims[-2], self.dims[-1], dropout=self.dropout, bias=False, act=lambda x:x))
        """

    def __call__(self, inputs):

        samples, support_size = self.sample(inputs)

        hidden = [tf.nn.embedding_lookup(self.input_features, node_samples) for node_samples in samples]

        for layer in range(len(self.num_samples)):
            # print(hidden)
            aggregator = self.aggregators[layer]

            next_hidden = []
            for hop in range(len(self.num_samples) - layer):
                neighbor_dims = [self.batch_size * support_size[hop], self.num_samples[len(self.num_samples)-hop-1],
                                    self.dims[layer]]
                h = aggregator((hidden[hop], tf.reshape(hidden[hop+1], neighbor_dims)))
                # h = tf.layers.batch_normalization(h)
                next_hidden.append(h)
            
            hidden = next_hidden
        
        return hidden[0]


    def sample_neighbors(self, nodes, n_samples):

        adj_list = tf.nn.embedding_lookup(self.node_neighbors, nodes)
        adj_list = tf.transpose(tf.random_shuffle(tf.transpose(adj_list)))
        adj_list = tf.slice(adj_list, [0, 0], [-1, n_samples])
        
        return adj_list


    def sample(self, inputs):

        """ 
        Sample neighbors to be the supportive fields for multi-layer convolutions.
        Args:
            inputs: batch inputs
        """
        samples = [inputs]
        support_size = 1 
        support_sizes = [support_size]

        for k in range(len(self.num_samples)):
            t = len(self.num_samples) -k - 1
            support_size *= self.num_samples[t]
            node = self.sample_neighbors(samples[k], self.num_samples[t])
            samples.append(tf.reshape(node, [support_size * self.batch_size, ]))
            support_sizes.append(support_size)
        
        return samples, support_sizes



    # input_dim, output_dim, neigh_input_dim=None, dropout=0., bias=False, act=tf.nn.relu




class RFDGP(object):

    def __init__(self, input_features, node_neighbors, dims, num_samples, batch_size, 
                latent_units, n_omega, sample_size=1, dropout=0.0, bias=False, act=tf.nn.relu):

        """
        node_neighbors: [N, max_degree]
        dims: [h1_units, h2_units], number of latent units
        num_samples: [h1_neighbors, h2_neighbors], numer of neighbors to be aggregated in each layer
        latent_units: [[h1], [h2]]latent units of inference network of omega
        n_omega: number of samples of omega
        """

        self.input_features = input_features
        self.node_neighbors = node_neighbors
        # self.dims = [self.input_features.shape[1]]
        self.dims = [self.input_features.get_shape().as_list()[1]]
        self.dims.extend(dims)
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.dropout = dropout
        self.act = act

        # sample neighborhood nodes for aggregation
        self.aggregators = []

        i = 0
        for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:]):
            # self.aggregators.append(GCNAggregator(in_dim, out_dim, dropout=self.dropout, bias=False, act=self.act))
            self.aggregators.append(RFFAggregator(in_dim, out_dim, latent_units[i], n_omega, self.dropout, act=self.act, 
                                    sample_size=sample_size, res_connection=False, name="layer{}".format(i)))
            i += 1


    def __call__(self, inputs):

        samples, support_size = self.sample(inputs)

        hidden = [tf.nn.embedding_lookup(self.input_features, node_samples) for node_samples in samples]

        for layer in range(len(self.num_samples)):
            # print(hidden)
            aggregator = self.aggregators[layer]

            next_hidden = []
            for hop in range(len(self.num_samples) - layer):
                neighbor_dims = [self.batch_size * support_size[hop], self.num_samples[len(self.num_samples)-hop-1],
                                    self.dims[layer]]
                h = aggregator((hidden[hop], tf.reshape(hidden[hop+1], neighbor_dims)))
                # h = tf.layers.batch_normalization(h)
                # h = h + res_block(hidden[hop])
                next_hidden.append(h)
            
            hidden = next_hidden
        
        return hidden[0]


    def sample_neighbors(self, nodes, n_samples):

        adj_list = tf.nn.embedding_lookup(self.node_neighbors, nodes)
        adj_list = tf.transpose(tf.random_shuffle(tf.transpose(adj_list)))
        adj_list = tf.slice(adj_list, [0, 0], [-1, n_samples])
        
        return adj_list


    def sample(self, inputs):

        """ 
        Sample neighbors to be the supportive fields for multi-layer convolutions.
        Args:
            inputs: batch inputs
        """
        samples = [inputs]
        support_size = 1 
        support_sizes = [support_size]

        for k in range(len(self.num_samples)):
            t = len(self.num_samples) -k - 1
            support_size *= self.num_samples[t]
            node = self.sample_neighbors(samples[k], self.num_samples[t])
            samples.append(tf.reshape(node, [support_size * self.batch_size, ]))
            support_sizes.append(support_size)
        
        return samples, support_sizes




class SemiRFDGP(object):

    def __init__(self, placeholders, input_features, node_neighbors, dims, num_samples, latent_units, n_omega, 
                trans_feature=False, feature_dim=[512], sample_size=1, dropout=0.0, bias=False, 
                act=tf.nn.relu, weight_decay=5e-4, lamb=1.0, lr=1e-3):

        self.batch = placeholders['nodes']
        self.Y = placeholders['Y']
        self.label_mask = placeholders['label_mask']
        self.batch_size = placeholders["batch_size"]

        self.input_dim = input_features.shape[1]
        self.n_classes = placeholders['Y'].get_shape().as_list()[1]

        self.num_samples = num_samples
        self.weight_decay = weight_decay
        self.lamb = lamb
        self.lr = lr

        with tf.variable_scope("trans_feature"):    
            if trans_feature:
                feature = NeuralNet(self.input_dim, feature_dim, dropout=dropout, act=act)(input_features)
                # feature = GraphConvolution(input_features, node_neighbors, feature_dim, [25, 10], self.batch_size, dropout=dropout, act=tf.nn.relu)(self.batch)
            else:
                feature = tf.identity(input_features)
        
        with tf.variable_scope("DGPlayers"):
            self.layers = RFDGP(feature, node_neighbors, dims, num_samples, self.batch_size, latent_units,
                                n_omega, sample_size=sample_size, dropout=dropout, act=act)
        
        """
        with tf.variable_scope("softmax"):
            self.softmax_weight = glorot([dims[-1], self.n_classes])
        """

        self._build()
    
    
    def _build(self):

        outputs = self.layers(self.batch)
        # outputs = tf.matmul(outputs, self.softmax_weight)

        # ================================== build loss =====================================

        self.reconstruct_loss = masked_softmax_cross_entropy(outputs, self.Y, self.label_mask)

        scale = 1. / tf.cast(tf.reduce_sum(self.label_mask), tf.float32)
        kl = 0.0 

        for layer in self.layers.aggregators:
            kl += layer.obtain_KL_prior()
        self.kl = scale * kl * self.lamb

        # l2_loss = tf.nn.l2_loss(self.feature_layer.get_vars()[0])
        transf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="trans_feature")

        dgp_infernet_vars = []
        dgp_w_vars = []
        dgp_res_vars = []
        for i in range(len(self.num_samples)):
            dgp_infernet_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DGPlayers/layer{}_inferencenet".format(i))
            dgp_w_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DGPlayers/layer{}_W".format(i))
            dgp_res_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DGPlayers/layer{}_res_block".format(i))
        
        # softmax_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="softmax")

        # tf_vars = transf_vars + dgp_infernet_vars + softmax_vars
        tf_vars = transf_vars + dgp_infernet_vars + dgp_res_vars
        
        l2_loss = 0.0
        for var in tf_vars:
            l2_loss += tf.nn.l2_loss(var)
        self.l2_loss = self.weight_decay * l2_loss

        self.loss = self.reconstruct_loss + self.kl + self.l2_loss

        # ====================================================================================

        self.pt_optimizer = tf.train.AdamOptimizer(self.lr)
        pt_grads_and_vars = self.pt_optimizer.compute_gradients(self.loss, var_list= transf_vars + dgp_w_vars)
        self.opt_step_pt = self.pt_optimizer.apply_gradients(pt_grads_and_vars)

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                    for grad, var in grads_and_vars]
        self.opt_step = self.optimizer.apply_gradients(clipped_grads_and_vars)

        # accuracy
        self.accuracy = masked_accuracy(outputs, self.Y, self.label_mask)

       
        






        
        




from Datasets import Graph

if __name__ == "__main__":

    batch_size = 512
    g = Graph("cora", batch_size = batch_size)
    
    placeholders = {
        'nodes': tf.placeholder(dtype=tf.int32, shape=[None]),
        'Y': tf.placeholder(dtype=tf.float32, shape=[None, g.n_classes]),
        'label_mask': tf.placeholder(dtype=tf.int32, shape=[None]),
        'localSim': tf.placeholder(dtype=tf.float32, shape=[None, None]), 
        'globalSim': tf.placeholder(dtype=tf.float32, shape=[None, None]),
        "batch_size": tf.placeholder(tf.int32, name='batch_size')
    }

    model = GraphConvolution(g.feature, g.node_neighbors, [64, 32], [25, 10], placeholders["batch_size"], dropout=0.5)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):

        feed_dict = g.next_batch_feed_dict(placeholders)
        # print(feed_dict)

        out = model(placeholders["nodes"])
        out = sess.run(out, feed_dict=feed_dict)
        print(out)
        print(out.shape)
    




        

        



