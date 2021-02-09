import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import numpy as np
import pickle
import operator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_GPU = 1
num_code = 526
max_num_visit = 200


class Config(object):
    def __init__(self):
        self.vocab_dim = num_code
        self.lstm_dim = 256
        self.n_layer = 3
        self.batchsize = 100
        self.Z_DIM = 128
        self.G_DIMS = [256, 256, 256, 256, 256, 256, self.vocab_dim + 2]
        self.D_DIMS = [256, 256, 256, 256, 256, 256]
        self.max_num_visit = max_num_visit
        self.max_code_visit = 80


class PointWiseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(PointWiseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.bias = self.add_variable("bias", shape=[self.num_outputs])

    def call(self, x, y):
        return x * y + self.bias


def prob2onehot(prob):
    return tf.cast((tf.reduce_max(prob, axis=-1, keepdims=True) - prob) == 0, tf.float32)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim) for dim in config.G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5, center=False, scale=False)] + \
                                 [tf.keras.layers.BatchNormalization(epsilon=1e-5) for _ in config.G_DIMS[1:-1]]
        self.output_layer_code = tf.keras.layers.Dense(config.G_DIMS[-1], activation=tf.nn.sigmoid)
        self.output_layer_stay = tf.keras.layers.Dense(8, activation=tf.nn.softmax)
        self.output_layer_left = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        self.output_layer_right = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        self.output_layer_age = tf.keras.layers.Dense(17, activation=tf.nn.softmax)
        self.condition_layer = tf.keras.layers.Dense(config.G_DIMS[0])
        self.pointwiselayer = PointWiseLayer(config.G_DIMS[0])

    def call(self, condition):
        x = tf.random.normal(shape=[condition.shape.as_list()[0], config.Z_DIM])
        h = self.dense_layers[0](x)
        x = tf.nn.relu(
            self.pointwiselayer(self.batch_norm_layers[0](h, training=False), self.condition_layer(condition)))
        for i in range(1, len(config.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=False))
            x += h
        x = tf.concat((tf.math.round(self.output_layer_code(x)), prob2onehot(self.output_layer_stay(x)),
                       prob2onehot(self.output_layer_left(x)),
                       prob2onehot(self.output_layer_right(x)),
                       prob2onehot(self.output_layer_age(x))), axis=-1)
        return x


def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.2
    else:
        dropout_rate = 0.0
    mask = tf.nn.dropout(tf.ones([inputs.shape[0], 1, inputs.shape[2]], dtype=tf.float32), dropout_rate)
    mask = tf.tile(mask, [1, inputs.shape[1], 1])
    return inputs * mask
    # b*t*u


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        lstm = DropConnectLSTM
        self.layer = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(config.n_layer)]

    def call(self, x, is_training):
        for layer in self.layer:
            layer.set_mask(is_training)

        for i in range(config.n_layer):
            x = locked_drop(x, is_training)
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self, is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units, self.units * 4]), 0.2)
        else:
            self.mask = tf.ones([self.units, self.units * 4])

    def _process_batch(self, inputs, initial_state):
        if not self.time_major:
            inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
        input_h = initial_state[0]
        input_c = initial_state[1]
        input_h = array_ops.expand_dims(input_h, axis=0)
        input_c = array_ops.expand_dims(input_c, axis=0)

        params = recurrent_v2._canonical_to_params(  # pylint: disable=protected-access
            weights=[
                self.kernel[:, :self.units],
                self.kernel[:, self.units:self.units * 2],
                self.kernel[:, self.units * 2:self.units * 3],
                self.kernel[:, self.units * 3:],
                self.recurrent_kernel[:, :self.units] * self.mask[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2] * self.mask[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3] * self.mask[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:] * self.mask[:, self.units * 3:],
            ],
            biases=[
                self.bias[:self.units],
                self.bias[self.units:self.units * 2],
                self.bias[self.units * 2:self.units * 3],
                self.bias[self.units * 3:self.units * 4],
                self.bias[self.units * 4:self.units * 5],
                self.bias[self.units * 5:self.units * 6],
                self.bias[self.units * 6:self.units * 7],
                self.bias[self.units * 7:],
            ],
            shape=self._vector_shape)

        outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
            inputs,
            input_h=input_h,
            input_c=input_c,
            params=params,
            is_training=True)

        if self.stateful or self.return_state:
            h = h[0]
            c = c[0]
        if self.return_sequences:
            if self.time_major:
                output = outputs
            else:
                output = array_ops.transpose(outputs, perm=(1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h, c]


class Embedding(tf.keras.Model):
    def __init__(self):
        super(Embedding, self).__init__()
        self.linear_forward = tf.keras.layers.Dense(config.lstm_dim)
        self.linear_backward = tf.keras.layers.Dense(config.lstm_dim)
        self.embeddings = tf.keras.layers.Embedding(2, config.lstm_dim)

    def call(self, code, others, length):
        code = tf.reduce_sum(tf.one_hot(code, depth=num_code, dtype=tf.float32), axis=-2)
        interval_forward = others[:, :-1, 27:37]
        interval_backward = tf.concat(
            (tf.expand_dims(others[:, 0, 27:37], axis=-2),
             tf.reverse_sequence(others[:, 2:, 27:37], length - 2, seq_axis=1)), axis=-2)
        feature = tf.concat((code, others[:, :, 37:], others[:, :, :27]), axis=-1)

        feature_forward = tf.concat((feature[:, :-1], interval_forward), axis=-1)
        feature_backward = tf.concat((tf.reverse_sequence(feature[:, 1:], length - 1, seq_axis=1), interval_backward),
                                     axis=-1)

        x_forward = tf.concat((self.embeddings(tf.zeros((code.shape[0], 1))), self.linear_forward(feature_forward)),
                              axis=-2)
        x_backward = tf.concat((self.embeddings(tf.ones((code.shape[0], 1))), self.linear_backward(feature_backward)),
                               axis=-2)
        return x_forward, x_backward


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.embeddings = Embedding()
        self.lstm_forward = LSTM()
        self.lstm_backward = LSTM()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dense = tf.keras.layers.Dense(256)
        self.alpha = tf.keras.layers.Dense(256, activation=tf.nn.sigmoid)

    def call(self, code, others, length, idx, is_training=False):
        x_forward, x_backward = self.embeddings(code, others, length)
        x_forward = self.lstm_forward(x_forward, is_training)
        x_backward = tf.reverse_sequence(self.lstm_backward(x_backward, is_training), length, seq_axis=1)

        x = tf.concat((x_forward, x_backward), axis=-1)
        y = self.alpha(x)
        x = y*x_forward+(1-y)*x_backward
        x = tf.gather_nd(x, tf.concat(
            (tf.expand_dims(tf.range(code.shape.as_list()[0], dtype=tf.int32), -1), tf.expand_dims(idx, -1)),
            axis=-1))
        latent = self.layer_norm(self.dense(x))
        return latent


def gen_dataset(k):
    model = Model()
    generator = Generator()

    checkpoint_directory_model = "training_checkpoints_bilstm_single"
    checkpoint_prefix_model = os.path.join(checkpoint_directory_model, "ckpt")

    checkpoint_directory_generator = "training_checkpoints_gan_single_syn"
    checkpoint_prefix_generator = os.path.join(checkpoint_directory_generator, "ckpt")

    checkpoint_model = tf.train.Checkpoint(model=model)
    checkpoint_model.restore(checkpoint_prefix_model + '-16').expect_partial()

    checkpoint_generator = tf.train.Checkpoint(generator=generator)
    checkpoint_generator.restore(checkpoint_prefix_generator + '-30').expect_partial()

    @tf.function
    def insert_step(batch_code, batch_others, batch_length, batch_idx):
        batch_latent = model(batch_code, batch_others, batch_length, batch_idx)
        batch_latent = tf.tile(tf.squeeze(batch_latent), [10, 1])
        batch_insert = generator(batch_latent)
        return batch_insert

    def gen_batch(batch_code, batch_others, batch_length):
        max_length = np.max(batch_length)
        for _ in range(5):
            insert_sequence = []
            for _ in range(len(batch_code)):
                tmp_sequence = np.arange(max_length)
                np.random.shuffle(tmp_sequence)
                insert_sequence.append(tmp_sequence)
            insert_sequence = np.array(insert_sequence)
            for i in range(max_length):
                batch_idx = insert_sequence[:, i]
                update_idx = np.arange(len(batch_code))[batch_idx - batch_length < 0]
                batch_insert = insert_step(tf.convert_to_tensor(batch_code, dtype=tf.int32),
                                           tf.convert_to_tensor(batch_others, dtype=tf.float32),
                                           tf.convert_to_tensor(batch_length, dtype=tf.int32),
                                           tf.convert_to_tensor(batch_idx, dtype=tf.int32)).numpy()

                correct_output = batch_insert[:len(batch_code)]
                age_violate = np.zeros(len(batch_code), dtype='bool')
                age_idx = np.arange(len(batch_code))[np.logical_and(batch_idx != 0, (batch_idx - batch_length) < -1)]
                age_left = np.sum(batch_others[age_idx, batch_idx[age_idx] - 1, :17] * np.arange(17),axis=-1)
                age_right = np.sum(batch_others[age_idx, batch_idx[age_idx] + 1, :17] * np.arange(17),axis=-1)
                for attempt in range(10):
                    tmp_age = np.sum(correct_output[age_idx, 556:] * np.arange(17),axis=-1)
                    age_violate[age_idx][np.logical_and((tmp_age - age_left) != 0, (tmp_age - age_right) != 0)] = True
                    num_code_visit = np.sum(correct_output[:, :526] == 1, axis=-1)
                    left_edge = np.logical_and(correct_output[:, 536] == 1, batch_idx != 0)
                    right_edge = np.logical_and(correct_output[:, 546] == 1, (batch_idx - batch_length) != -1)
                    violate = np.logical_or(np.logical_or(np.logical_or(num_code_visit == 0,
                                                                        num_code_visit - config.max_code_visit > 0),
                                                          np.logical_or(left_edge, right_edge)),
                                            age_violate)
                    if np.sum(violate[update_idx]) == 0:
                        break
                    correct_output[violate] = batch_insert[attempt * len(batch_code):(attempt + 1) * len(batch_code)][violate]

                insert_code = [np.arange(526)[correct_output[n, :526] == 1] for n in range(len(batch_code))]
                insert_code = np.array(
                    [np.pad(w, (0, config.max_code_visit - len(w)), 'constant', constant_values=-1) for w in
                     insert_code])
                batch_code[update_idx, batch_idx[update_idx]] = insert_code[update_idx]
                batch_others[update_idx, batch_idx[update_idx], 19:37] = correct_output[update_idx, 528:546]
                batch_others[update_idx, batch_idx[update_idx], -2:] = correct_output[update_idx, 526:528]
                right_update_idx = batch_idx[update_idx] + 1
                right_update = right_update_idx < max_num_visit
                batch_others[update_idx[right_update], right_update_idx[right_update], 27:37] = correct_output[update_idx[right_update], 546:556]
                batch_others[np.arange(len(batch_code))[batch_idx == 0], 0, 27:37] = np.array([1,0,0,0,0,0,0,0,0,0],dtype='float')
                batch_others[update_idx, batch_idx[update_idx], :17] = correct_output[update_idx, 556:]

        tmp_code = [w[:v] for w, v in zip(batch_code, batch_length)]
        tmp_others = [w[:v] for w, v in zip(batch_others, batch_length)]
        return tmp_code, tmp_others
    n_batch = int(len(length_train) / config.batchsize)
    feature_tmp = []
    summary_tmp = []
    for j in range(n_batch):
        x, y = gen_batch(feature_train[j * config.batchsize:(j + 1) * config.batchsize],
                         summary_train[j * config.batchsize:(j + 1) * config.batchsize],
                         length_train[j * config.batchsize:(j + 1) * config.batchsize])
        feature_tmp.extend(x)
        summary_tmp.extend(y)
    x, y = gen_batch(feature_train[n_batch * config.batchsize:],
                     summary_train[n_batch * config.batchsize:],
                     length_train[n_batch * config.batchsize:])
    feature_tmp.extend(x)
    summary_tmp.extend(y)
    with open('single_over_test_1000/code_train_' + str(k) + '.pkl', 'wb') as f:
        pickle.dump(feature_tmp, f)
    with open('single_over_test_1000/others_train_' + str(k) + '.pkl', 'wb') as f:
        pickle.dump(summary_tmp, f)


if __name__ == '__main__':
    config = Config()
    lengths = np.load('single_over/length_train_1.npy')
    features = np.load('single_over/code_train_1.npy')
    summaries = np.load('single_over/others_train_1.npy')

    order = np.argsort(lengths)
    length_train = lengths[order]
    feature_train = features[order]
    summary_train = summaries[order]

    for a in range(10):
        t = time.time()
        gen_dataset(a)
        print(time.time() - t)
