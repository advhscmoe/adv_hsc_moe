import tensorflow as tf
from utils import log_min_max
import mixture_of_experts

class BaseModel(object):
    def __init__(self, n_uid, n_iid, n_cat, n_brand, EMBEDDING_DIM):
        self.n_uid = n_uid
        self.n_iid = n_iid
        self.n_cat = n_cat
        self.n_brand = n_brand
        with tf.name_scope('numeric_features'):
            self.asin_overall_cnt_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='asin_overall_cnt_ph'), 1.0, 14346.0)
            self.asin_overall_cnt_1_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='asin_overall_cnt_1_ph'), 0.0, 1659.0)
            self.asin_overall_cnt_2_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='asin_overall_cnt_2_ph'), 0.0, 1242.0)
            self.asin_overall_cnt_3_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='asin_overall_cnt_3_ph'), 0.0, 2280.0)
            self.asin_overall_cnt_4_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='asin_overall_cnt_4_ph'), 0.0, 3346.0)
            self.asin_overall_cnt_5_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='asin_overall_cnt_5_ph'), 0.0, 9558.0)
            self.reviewer_overall_cnt_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='reviewer_overall_cnt_ph'), 1.0, 1455.0)
            self.reviewer_overall_cnt_1_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='reviewer_overall_cnt_1_ph'), 0.0, 287.0)
            self.reviewer_overall_cnt_2_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='reviewer_overall_cnt_2_ph'), 0.0, 180.0)
            self.reviewer_overall_cnt_3_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='reviewer_overall_cnt_3_ph'), 0.0, 799.0)
            self.reviewer_overall_cnt_4_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='reviewer_overall_cnt_4_ph'), 0.0, 653.0)
            self.reviewer_overall_cnt_5_ph = log_min_max(
                tf.placeholder(tf.float32, [None, 1], name='reviewer_overall_cnt_5_ph'), 0.0, 750.0)

        with tf.name_scope('sparse_features'):
            # reviewerId, asin, brand, asin_hist, brand_hist, top_cat, category
            self.uid_ph = tf.placeholder(tf.int32, [None, ], name='uid_ph')
            self.iid_ph = tf.placeholder(tf.int32, [None, ], name='iid_ph')
            self.brand_ph = tf.placeholder(tf.int32, [None, ], name='brand_ph')
            self.category_ph = tf.placeholder(tf.int32, [None, ], name='category_ph') # sub category
            self.iid_hist_ph = tf.placeholder(tf.int32, [None, None], name='iid_hist_ph')
            self.brand_hist_ph = tf.placeholder(tf.int32, [None, None], name='brand_hist_ph')
            self.top_cat_ph = tf.placeholder(tf.int32, [None, None], name='top_hist_ph') # top category

        with tf.name_scope('embedding_layer'):
            self.uid_embeddings_weight = tf.get_variable('uid_embedding_weight', [n_uid, EMBEDDING_DIM])
            self.uid_embedded = tf.nn.embedding_lookup(self.uid_embeddings_weight, self.uid_ph)
            self.iid_embeddings_weight = tf.get_variable('iid_embedding_weight', [n_iid, EMBEDDING_DIM])
            self.iid_embedded = tf.nn.embedding_lookup(self.iid_embeddings_weight, self.iid_ph)
            self.brand_embeddings_weight = tf.get_variable('brand_embedding_weight', [n_brand, EMBEDDING_DIM])
            self.brand_embedded = tf.nn.embedding_lookup(self.brand_embeddings_weight, self.brand_ph)
            self.category_embeddings_weight = tf.get_variable('category_embedding_weight', [n_cat, EMBEDDING_DIM])
            self.category_embedded = tf.nn.embedding_lookup(self.category_embeddings_weight, self.category_ph)
            self.iid_hist_embeddings_weight = tf.get_variable('iid_hist_embedding_weight', [n_iid, EMBEDDING_DIM])
            self.iid_hist_embedded = tf.nn.embedding_lookup(self.iid_hist_embeddings_weight, self.iid_hist_ph)
            self.brand_hist_embeddings_weight = tf.get_variable('brand_hist_embedding_weight', [n_brand, EMBEDDING_DIM])
            self.brand_hist_embedded = tf.nn.embedding_lookup(self.brand_hist_embeddings_weight, self.brand_hist_ph)
            self.top_cat_embeddings_weight = tf.get_variable('top_cat_embedding_weight', [n_cat, EMBEDDING_DIM])
            self.top_cat_embedded = tf.nn.embedding_lookup(self.top_cat_embeddings_weight, self.top_cat_ph)

        self.numeric_fea = tf.concat([self.asin_overall_cnt_ph, self.asin_overall_cnt_1_ph, self.asin_overall_cnt_2_ph,
            self.asin_overall_cnt_3_ph, self.asin_overall_cnt_4_ph, self.asin_overall_cnt_5_ph,
            self.reviewer_overall_cnt_ph, self.reviewer_overall_cnt_1_ph, self.reviewer_overall_cnt_3_ph,
            self.reviewer_overall_cnt_4_ph, self.reviewer_overall_cnt_5_ph], 1)
        self.embedded_fea = tf.concat([self.uid_embedded, self.iid_embedded, self.brand_embedded, self.category_embedded,
            self.iid_hist_embedded, self.brand_hist_embedded, self.top_cat_embedded])
        self.label_ph = tf.placeholder(tf.float32, [None, 1], name='label_ph')
        self.lr = tf.placeholder(tf.float64, [])

    def input_to_feed_dict(self, inps):
        return {
            self.asin_overall_cnt_ph: inps[0],
            self.asin_overall_cnt_1_ph: inps[1],
            self.asin_overall_cnt_2_ph: inps[2],
            self.asin_overall_cnt_3_ph: inps[3],
            self.asin_overall_cnt_4_ph: inps[4],
            self.asin_overall_cnt_5_ph: inps[5],
            self.reviewer_overall_cnt_ph: inps[6],
            self.reviewer_overall_cnt_1_ph: inps[7],
            self.reviewer_overall_cnt_2_ph: inps[8],
            self.reviewer_overall_cnt_3_ph: inps[9],
            self.reviewer_overall_cnt_4_ph: inps[10],
            self.reviewer_overall_cnt_5_ph: inps[11],
            self.uid_ph: inps[12],
            self.iid_ph: inps[13],
            self.brand_ph: inps[14]
            self.category_ph: inps[15],
            self.iid_hist_ph: inps[16],
            self.brand_hist_ph: inps[17],
            self.top_cat_ph: inps[18],
            self.label_ph: inps[19],
            self.lr: inps[20]
        }
    def train(self, sess, inps):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], 
            feed_dict=self.input_to_feed_dict(inps))
        return loss, accuracy

    def calculate(self, sess, inps):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], 
            feed_dict=self.input_to_feed_dict(inps))
        return probs, loss, accuracy
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)
    
    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)
    
class DNN(BaseModel):
    def __init__(self, n_uid, n_iid, n_cat, EMBEDDING_DIM):
        super(DNN, self).__init__(n_uid, n_iid, n_cat, EMBEDDING_DIM)
        inp = tf.concat([self.numeric_fea, self.embedded_fea], 1)
        self.moe_model = mixture_of_experts.MixtureOfExperts(1, 1, adversarial_bottom_k=0, hsc_lambda=0.0, adv_lambda=0.0, lb_lambda=1e-1)
        self.forward(inp)

    def forward(self, inp):
        out = self.moe_model.compute_one_expert(inp)
        self.logits = out
        self.y_hat = tf.nn.sigmoid(self.logits)

        with tf.name_scope('Metrics'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits, 1.0) 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.label_ph), tf.float32))
        self.merged = tf.summary.merge_all()

class MoE(BaseModel):
    def __init__(self, n_uid, n_iid, n_cat, EMBEDDING_DIM):
        super(MoE, self).__init__(n_uid, n_iid, n_cat, EMBEDDING_DIM)
        self.gating_features = {'sc': self.category_embedded}
        self.expert_num = 4
        self.features = tf.concat([self.numeric_fea, self.embedded_fea], 1)
        self.moe_model = mixture_of_experts.MixtureOfExperts(10, 4, adversarial_bottom_k=0, hsc_lambda=0.0, adv_lambda=0.0, lb_lambda=1e-1)
        self.forward(self.features, self.gating_features)

    def forward(self, features, gating_features):
        self.logits, load_loss = self.moe_model.basic_moe(features, gating_features)
        self.y_hat = tf.nn.sigmoid(self.logits) 
        with tf.name_scope('Metrics'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits, 1.0)
            self.loss += load_loss 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.label_ph), tf.float32))
        self.merged = tf.summary.merge_all()

class MoE_HSC(BaseModel):
    def __init__(self, n_uid, n_iid, n_cat, EMBEDDING_DIM):
        super(MoE_HSC, self).__init__(n_uid, n_iid, n_cat, EMBEDDING_DIM)
        self.gating_features = {'tc': self.top_cat_embedded,'sc': self.category_embedded}
        self.expert_num = 4
        self.features = tf.concat([self.numeric_fea, self.embedded_fea], 1)
        self.moe_model = mixture_of_experts.MixtureOfExperts(10, 4,adversarial_bottom_k=0,hsc_lambda=1e-1,adv_lambda=0.0,lb_lambda=1e-1)
        self.forward(self.features, self.gating_features)

    def forward(self, features, gating_features):
        self.logits, aux_loss = self.moe_model.hsc_adv_moe(features, gating_features)
        self.y_hat = tf.nn.sigmoid(self.logits) 
        with tf.name_scope('Metrics'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits, 1.0)
            self.loss += aux_loss 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.label_ph), tf.float32))
        self.merged = tf.summary.merge_all()

class MoE_ADV(BaseModel):
    def __init__(self, n_uid, n_iid, n_cat, EMBEDDING_DIM):
        super(MoE_ADV, self).__init__(n_uid, n_iid, n_cat, EMBEDDING_DIM)
        self.gating_features = {'sc': self.category_embedded}
        self.expert_num = 4
        self.features = tf.concat([self.numeric_fea, self.embedded_fea], 1)
        self.moe_model = mixture_of_experts.MixtureOfExperts(10, 4,adversarial_bottom_k=1,hsc_lambda=0.0,adv_lambda=1e-1,lb_lambda=1e-1)
        self.forward(self.features, self.gating_features)

    def forward(self, features, gating_features):
        self.logits, aux_loss = self.moe_model.hsc_adv_moe(features, gating_features)
        self.y_hat = tf.nn.sigmoid(self.logits) 
        with tf.name_scope('Metrics'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits, 1.0)
            self.loss += aux_loss 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.label_ph), tf.float32))
        self.merged = tf.summary.merge_all()

class MoE_HSC_ADV(BaseModel):
    def __init__(self, n_uid, n_iid, n_cat, EMBEDDING_DIM):
        super(MoE_HSC_ADV, self).__init__(n_uid, n_iid, n_cat, EMBEDDING_DIM)
        self.gating_features = {'tc': self.top_cat_embedded,'sc': self.category_embedded}
        self.expert_num = 4
        self.features = tf.concat([self.numeric_fea, self.embedded_fea], 1)
        self.moe_model = mixture_of_experts.MixtureOfExperts(10, 4,adversarial_bottom_k=1,hsc_lambda=1e-1,adv_lambda=1e-1,lb_lambda=1e-1)
        self.forward(self.features, self.gating_features)

    def forward(self, features, gating_features):
        self.logits, aux_loss = self.moe_model.hsc_adv_moe(features, gating_features)
        self.y_hat = tf.nn.sigmoid(self.logits) 
        with tf.name_scope('Metrics'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits, 1.0)
            self.loss += aux_loss 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.label_ph), tf.float32))
        self.merged = tf.summary.merge_all()