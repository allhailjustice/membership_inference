import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_GPU = 1

checkpoint_directory = "training_checkpoints_distinguish_x1_single"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
max_num_visit = 200
num_code = 526


class Config(object):
    def __init__(self):
        self.vocab_dim = num_code
        self.lstm_dim = 256
        self.n_layer = 3


def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.2
    else:
        dropout_rate = 0.0
    mask = tf.nn.dropout(tf.ones([inputs.shape[0],1,inputs.shape[2]],dtype=tf.float32), dropout_rate)
    mask = tf.tile(mask, [1,inputs.shape[1],1])
    return inputs*mask
    # b*t*u


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        lstm = DropConnectLSTM
        self.layer_f = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.layer_b = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.dense = [tf.keras.layers.Dense(config.lstm_dim) for _ in range(config.n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(config.n_layer)]

    def call(self, x, length, is_training):
        for layer in self.layer_f:
            layer.set_mask(is_training)
        for layer in self.layer_b:
            layer.set_mask(is_training)

        for i in range(config.n_layer):
            x = locked_drop(x, is_training)
            x_b = tf.reverse_sequence(x, length, seq_axis=1)
            x = self.layer_f[i](x)
            x_b = self.layer_b[i](x_b)
            x = self.dense[i](tf.concat((x,x_b),axis=-1))
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self,is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units,self.units*4]),0.2)
        else:
            self.mask = tf.ones([self.units,self.units*4])

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
                self.recurrent_kernel[:, :self.units]*self.mask[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2]*self.mask[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3]*self.mask[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:]*self.mask[:, self.units * 3:],
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

    def call(self, code, others, length):
        code = tf.reduce_sum(tf.one_hot(code, depth=num_code, dtype=tf.float32), axis=-2)
        feature = tf.concat((code, others),axis=-1)

        x_forward = self.linear_forward(feature)
        return x_forward


class FeatureNet(tf.keras.Model):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.embeddings = Embedding()
        self.lstm = LSTM()
        self.dense = tf.keras.layers.Dense(256)
        self.embed = tf.keras.layers.Embedding(1,256)
        self.mlp0 = tf.keras.layers.Dense(256,activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(256)

    def call(self, code, others, length, is_training=False):
        length = tf.squeeze(length)

        x = self.embeddings(code, others, length)
        x = self.lstm(x, length, is_training)
        y = tf.reduce_sum(self.dense(x) * self.embed(tf.zeros((100, 1))), axis=-1)
        mask = tf.sequence_mask(length, max_num_visit)
        y = tf.nn.softmax(tf.where(mask, y, -tf.ones_like(y) * 1e9))
        feature_vec = self.mlp1(self.mlp0(tf.squeeze(tf.matmul(tf.expand_dims(y, 1), x))))
        feature_vec = feature_vec / tf.math.sqrt(tf.reduce_sum(feature_vec ** 2, axis=-1, keepdims=True))
        return feature_vec


class AdamWeightDecay(tf.keras.optimizers.Adam):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 weight_decay_rate=0.0,
                 include_in_weight_decay=None,
                 exclude_from_weight_decay=None,
                 name='AdamWeightDecay',
                 **kwargs):
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                              epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                    apply_state)
        apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
            self.weight_decay_rate, name='adam_weight_decay_rate')

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var *
                apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
                use_locking=self._use_locking)
        return tf.no_op()

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        grads, tvars = list(zip(*grads_and_vars))
        if experimental_aggregate_gradients:
            # when experimental_aggregate_gradients = False, apply_gradients() no
            # longer implicitly allreduce gradients, users manually allreduce gradient
            # and passed the allreduced grads_and_vars. For now, the
            # clip_by_global_norm will be moved to before the explicit allreduce to
            # keep the math the same as TF 1 and pre TF 2.2 implementation.
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return super(AdamWeightDecay, self).apply_gradients(
            zip(grads, tvars),
            name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients['lr_t'], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super(AdamWeightDecay, self).get_config()
        config.update({
            'weight_decay_rate': self.weight_decay_rate,
        })
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


def train():
    length_test = np.load('single_over/length_train_1.npy')
    feature_test = np.load('single_over/code_train_1.npy')
    summary_test = np.load('single_over/others_train_1.npy')

    length_val = np.load('single_over/length_train_0.npy')
    feature_val = np.load('single_over/code_train_0.npy')
    summary_val = np.load('single_over/others_train_0.npy')
    val_order = np.argsort(length_val)
    test_order = np.argsort(length_test)

    length_val = length_val[val_order]
    feature_val = feature_val[val_order]
    summary_val = summary_val[val_order]
    dataset_val = tf.data.Dataset.from_tensor_slices((feature_val.astype('int32'),summary_val.astype('float32'),length_val))

    length_test = length_test[test_order]
    feature_test = feature_test[test_order]
    summary_test = summary_test[test_order]
    dataset_test = tf.data.Dataset.from_tensor_slices((feature_test.astype('int32'), summary_test.astype('float32'), length_test))

    optimizer = AdamWeightDecay(learning_rate=5e-5)
    feature_net = FeatureNet()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=feature_net)
    print('start')

    @tf.function
    def one_step(batch, batch_syn, is_training):
        with tf.GradientTape() as tape:
            feature_vec = feature_net(*batch, is_training)
            feature_vec_syn = feature_net(*batch_syn, is_training)
            pair_wise_d = tf.matmul(feature_vec, feature_vec_syn, transpose_b=True)*10
            loss = tf.linalg.diag_part(tf.nn.log_softmax(pair_wise_d))
            loss = -tf.reduce_mean(loss * (1-tf.math.exp(loss))**2)
        if is_training:
            grads = tape.gradient(loss, feature_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, feature_net.trainable_variables))
        return loss

    @tf.function
    def output_step(batch):
        feature_vec = feature_net(*batch)
        return feature_vec

    print('training start')
    for epoch in range(400):
        with open('single_over_1000/code_train_' + str(epoch%10) + '.pkl', 'rb') as file:
            feature_syn = pickle.load(file)
        feature_syn = np.array([np.concatenate((np.array(f), -np.ones((200 - len(f), 80))), axis=0) for f in feature_syn])
        with open('single_over_1000/others_train_' + str(epoch%10) + '.pkl', 'rb') as file:
            summary_syn = pickle.load(file)
        summary_syn = np.array(
            [np.concatenate((np.array(s), -np.ones((200 - len(s), 39))), axis=0) for s in summary_syn])
        dataset_syn = tf.data.Dataset.from_tensor_slices((feature_syn.astype('int32'),
                                                          summary_syn.astype('float32'),
                                                          length_val.astype('int32')))
        parsed_dataset = tf.data.Dataset.zip((dataset_val, dataset_syn)).shuffle(4096*3).batch(100, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)

        with open('single_over_test_1000/code_train_' + str(epoch%10) + '.pkl', 'rb') as file:
            feature_syn_test = pickle.load(file)
        feature_syn_test = np.array([np.concatenate((np.array(f), -np.ones((200 - len(f), 80))), axis=0) for f in feature_syn_test])
        with open('single_over_test_1000/others_train_' + str(epoch%10) + '.pkl', 'rb') as file:
            summary_syn_test = pickle.load(file)
        summary_syn_test = np.array(
            [np.concatenate((np.array(s), -np.ones((200 - len(s), 39))), axis=0) for s in summary_syn_test])

        dataset_syn_test = tf.data.Dataset.from_tensor_slices((feature_syn_test.astype('int32'),
                                                          summary_syn_test.astype('float32'),
                                                          length_test.astype('int32')))
        parsed_dataset_test = tf.data.Dataset.zip((dataset_test, dataset_syn_test)).shuffle(4096 * 3).batch(100,
                                                                                               drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        start_time = time.time()
        loss_val = 0
        loss_test = 0
        step_val = 0
        step_test = 0
        for aug1, aug2 in parsed_dataset:
            step_loss = one_step(aug1, aug2, True).numpy()
            loss_val += step_loss
            step_val += 1
        for aug1, aug2 in parsed_dataset_test:
            step_loss = one_step(aug1, aug2, False).numpy()
            loss_test += step_loss
            step_test += 1
        duration_epoch = int(time.time() - start_time)
        format_str = 'epoch: %d, train_loss = %f, test_loss = %f (%d)'
        print(format_str % (epoch, loss_val / step_val, loss_test / step_test,
                            duration_epoch))
        if epoch % 10 == 9:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    config = Config()
    train()
