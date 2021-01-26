import tensorflow as tf
import os, sys
from collections import namedtuple

src = os.path.realpath(__file__)
to_vec = lambda t: tf.reshape(t, [-1])
to_row = lambda t: tf.reshape(t, [1, -1])
to_col = lambda t: tf.reshape(t, [-1, 1])

def tf_repeat(base, repeats, axis=0):
    if tf.__version__.startswith('1.15'):
        return tf.repeat(base, repeats, axis=axis)
    assert axis == 0
    base = tf.reshape(base, [-1, 1])
    max_repeat = tf.reduce_max(repeats)
    tiled = tf.tile(base, [1, max_repeat])
    mask = tf.sequence_mask(repeats)
    return to_col(tf.boolean_mask(tiled, mask))

sparse_embedding = namedtuple(  # similar to tf.IndexedSlices.
    'sparse_embedding', ['embedding_2d', 'term_indices', 'sequence_length'],
    verbose=False)

def segment_outer_range(segment_lengths, out_idx=tf.int32):
    """Given a list A of lengths, create [i for i, x in enumerate(A) for _ in range(x)]
    For example [2, 3, 1] -> [0, 0, 1, 1, 1, 2]
    """
    max_length = tf.reduce_max(segment_lengths)
    tiled_range = tf.tile(tf.expand_dims(tf.range(tf.size(
        segment_lengths, out_type=out_idx)), 1), [1, max_length])
    return tf.boolean_mask(
        tiled_range, tf.sequence_mask(segment_lengths, max_length))

def top_k_per_row(gate_values, expert_top_k):
    # replace raw gate values by softmax of top k within each row, and 0 otherwise.
    batch_size = tf.shape(gate_values)[0]
    top_k_values, expert_indices = tf.math.top_k(
        gate_values, k=expert_top_k, sorted=False)
    top_k_values = to_vec(tf.nn.softmax(top_k_values, axis=-1))
    row_indices = to_col(tf.tile(
        tf.reshape(tf.range(batch_size), [-1, 1]), [1, expert_top_k]))
    indices = tf.concat([row_indices, to_col(expert_indices)], axis=1)
    return tf.scatter_nd(indices, top_k_values, tf.shape(gate_values))


def cross_variation(x):
    epsilon = 1e-10
    float_size = tf.to_float(tf.size(x)) + epsilon
    mean = tf.reduce_sum(x) / float_size
    variance = tf.reduce_sum(tf.squared_difference(x, mean)) / float_size
    return variance / (tf.square(mean) + epsilon)
    

def broadcast_sparse_concat(sparse_embeddings_or_tensors, axis):
    ret = []
    sequence_lengths = [x.sequence_length for x in sparse_embeddings_or_tensors
        if isinstance(x, sparse_embedding)]
    if sequence_lengths:
        sequence_length = sequence_lengths[0]
    else:
        batch_size = tf.shape(sparse_embeddings_or_tensors[0])[0]
        sequence_length = tf.ones([batch_size], dtype=tf.int64)
    segment_ids = segment_outer_range(sequence_length)
    for t in sparse_embeddings_or_tensors:
        if isinstance(t, tf.Tensor):
            ret.append(tf_repeat(t, sequence_length, axis=0))
        else:
            ret.append(t.embedding_2d)
    return tf.concat(ret, axis=axis), segment_ids, sequence_length

def normal_distribution_cdf(x, stddev):
    return 0.5 * (1.0 + tf.erf(x / (tf.math.sqrt(2.0) * stddev + 1e-20)))

class MixtureOfExperts():
    def __init__(self, num_experts, expert_top_k, *args,
        adversarial_bottom_k=0,
        hsc_lambda=1.0, adv_lambda=1.0, lb_lambda=1.0, **kwargs):
        """Build MOE networks.

        Args:
            num_experts: Total number of experts.
            expert_top_k: the number of experts assigned to each example.
            gating_layers: the layer sizes for the gating network.
            expert_layers: the layer sizes of each MLP expert network.
            adversarial_bottom_k: number of bottom experts to contrast against
                top k expert scores.
            hsc_lambda: multiplier of HSC loss before adding to total loss.
            adv_lambda: multiplier of Adv loss before adding to total loss.
            lb_lambda: multiplier for load balancing loss.
        """
        self._num_experts = num_experts
        self._expert_top_k = expert_top_k
        self._adv_bottom_k = adversarial_bottom_k
        self._hsc_lambda, self._adv_lambda, self._lb_lambda = (
            hsc_lambda, adv_lambda, lb_lambda)

    def gating_network(self, gate_input):
        with tf.variable_scope('gating_network'):
            gating_net = tf.layers.dense(gate_input, self._num_experts, activation=None)
        return gating_net

    def noise_network(self, gate_input):
        batch_size = tf.shape(gate_input)[0]
        with tf.variable_scope('noise_network'):
            random_weight = tf.random.normal([batch_size, self._num_experts])
            noise_net = tf.layers.dense(gate_input, self._num_experts, activation=None)
        return random_weight * noise_net, noise_net

    def compute_one_expert(self, expert_input):
        bn1 = tf.layers.batch_normalization(inputs=expert_input)
        dnn1 = tf.layers.dense(bn1, 128, activation='relu')
        dnn2 = tf.layers.dense(dnn1, 64, activation='relu')
        dnn3 = tf.layers.dense(dnn2, 1, activation=None)
        return dnn3

    def compute_main(self, features, gate_output, expert_top_k):
        all_input = features 
        batch_size = tf.shape(all_input)[0]
        expert_weights = top_k_per_row(gate_output, expert_top_k)
        top_k_gates = tf.split(
            expert_weights, num_or_size_splits=self._num_experts, axis=1)
        logits, indices = [], []
        for i, gate_values in enumerate(top_k_gates):
            with tf.variable_scope('expert_{}'.format(i)):
                row_indices = tf.where(tf.greater(gate_values, 0))[:, 0]
                expert_input = tf.gather(all_input, row_indices)
                logits.append(self.compute_one_expert(expert_input))
            indices.append(
                tf.stack([row_indices, i * tf.ones_like(row_indices)], axis=1))
        logits, indices = tf.concat(logits, axis=0), tf.concat(indices, axis=0)
        scattered_logits = tf.scatter_nd(indices, logits, [
            batch_size, self._num_experts, tf.shape(logits)[1]])
        mask = tf.scatter_nd(indices, tf.ones_like(indices[:, 0]),
            [batch_size, self._num_experts])
        return scattered_logits, expert_weights, mask

    def adv_loss(self, features, leaf_gate_output, top_logits):
        with tf.variable_scope('adversarial'):
            bottom_logits, _, mask = self.compute_main(
                features, -leaf_gate_output, self._adv_bottom_k)
        # compute \sum_i \sum_{j,k} |top_{ij} - bottom_{ik}|^2
        bottom_logits = tf.tile(tf.reshape(tf.boolean_mask(bottom_logits, mask),
            [-1, 1, self._adv_bottom_k]), [1, self._expert_top_k, 1])
        top_logits = tf.tile(tf.expand_dims(
            top_logits, axis=2), [1, 1, self._adv_bottom_k])
        return -self._adv_lambda * tf.reduce_sum(
            tf.square(top_logits - bottom_logits))

    def hsc_loss(self, segment_ids, all_gate_output):
        parent_idx = to_vec(tf.where(tf.equal(segment_ids[:-1], segment_ids[1:])))
        parent = tf.gather(all_gate_output, parent_idx)
        child = tf.gather(all_gate_output, parent_idx + 1)
        ret = tf.reduce_sum(tf.square(parent - child))
        return self._hsc_lambda * ret

    def exlusive_top_k_prob(self, gate_output, noise_stddev):
        top_kp1, _ = tf.math.top_k(
            gate_output, k=self._expert_top_k + 1, sorted=True)
        kth_values = to_col(top_kp1[:, self._expert_top_k - 1])
        def tile_width(t):
            return tf.tile(to_col(t), [1, tf.shape(gate_output)[1]])
        threshold = tf.where(tf.greater_equal(gate_output, kth_values),
            tile_width(top_kp1[:, self._expert_top_k]),
            tile_width(top_kp1[:, self._expert_top_k - 1]))
        return normal_distribution_cdf(gate_output - threshold, noise_stddev)

    def aux_loss(self, gate_output, expert_weights, noise_stddev):
        # importance CV and load-balance loss.
        load_weights = tf.reduce_sum(expert_weights, axis=0)
        aux_loss = self._lb_lambda * cross_variation(load_weights)
        if self._lb_lambda != 0:
            assert self._expert_top_k < self._num_experts
            load_probs = tf.reduce_sum(self.exlusive_top_k_prob(
                gate_output, noise_stddev), axis=0)
            aux_loss += self._lb_lambda * cross_variation(load_probs)
        return aux_loss

    # Hierarchical Soft Constraint (see https://arxiv.org/pdf/2007.12349.pdf)
    def hsc_adv_moe(self, features, gating_features):
        concat_gate_input, segment_ids, _ = broadcast_sparse_concat([v for k, v
            in gating_features.items()], axis=1)
        leaf_pos = to_vec(tf.where(tf.concat([
            tf.not_equal(segment_ids[:-1], segment_ids[1:]), [True]], axis=0)))
        all_gate_output = self.gating_network(concat_gate_input)
        all_noise_output, noise_stddev = self.noise_network(concat_gate_input)
        leaf_gate_output = tf.gather(all_gate_output + all_noise_output, leaf_pos)
        scattered_logits, weights, top_mask = self.compute_main(
            features, leaf_gate_output, self._expert_top_k)
        top_logits, expert_weights = [tf.reshape(tf.boolean_mask(t, top_mask),
            [-1, self._expert_top_k]) for t in [scattered_logits, weights]]
        logits = tf.reduce_sum(top_logits * expert_weights, axis=1, keepdims=True)
        aux_loss = self.hsc_loss(segment_ids, all_gate_output) + self.aux_loss(
            leaf_gate_output, weights, tf.gather(noise_stddev, leaf_pos))
        if self._adv_bottom_k != 0:
            aux_loss += self.adv_loss(features, leaf_gate_output, top_logits)
        return logits, aux_loss 

    def basic_moe(self, features, gating_features):
        """Computes output logit of vanilla MOE model.

        Args:
            features: dict of all feature names to dense tensors
                (of the same batch size).
            gating_features: dict of gating input feature names to dense tensors. 

        Returns:
            logits: expert-gate weighted sum of logits for each example in the batch.
            load_balance_loss: expert load-balance to contribute to the final loss.
        """
        gate_input = tf.concat(
            [v for k, v in gating_features.items()], axis=1)
        gate_output = self.gating_network(gate_input)
        noise_output, noise_stddev = self.noise_network(gate_input)
        scattered_logits, expert_weights, _ = self.compute_main(
            features, gate_output + noise_output, self._expert_top_k)
        # reduce_sum(batch x num_expts x out_width, axis=1) => batch x out_width.
        logits = tf.reduce_sum(scattered_logits * tf.expand_dims(
            expert_weights, axis=2), axis=1)
        return logits, self.aux_loss(
            gate_output + noise_output, expert_weights, noise_stddev) 
