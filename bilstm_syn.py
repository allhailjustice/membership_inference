import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import re
import numpy as np
import tensorflow_addons as tfa

focal = tfa.losses.SigmoidFocalCrossEntropy()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_GPU = 1

checkpoint_directory = "training_checkpoints_bilstm_syn"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
max_num_visit = 200
num_code = 526


class Config(object):
    def __init__(self):
        self.vocab_dim = num_code
        self.lstm_dim = 256
        self.n_layer = 3
        self.batchsize = 144


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
        interval_backward = tf.concat((tf.expand_dims(others[:, 0, 27:37], axis=-2),
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
        self.dense_diagnosis = tf.keras.layers.Dense(config.vocab_dim + 2, activation=tf.nn.sigmoid)
        self.dense_stay = tf.keras.layers.Dense(8, activation=tf.nn.log_softmax)
        self.dense_left = tf.keras.layers.Dense(10, activation=tf.nn.log_softmax)
        self.dense_right = tf.keras.layers.Dense(10, activation=tf.nn.log_softmax)
        self.dense_age = tf.keras.layers.Dense(17, activation=tf.nn.log_softmax)

    def call(self, code, others, length, is_training=False):
        x_forward, x_backward = self.embeddings(code, others, length)
        x_forward = self.lstm_forward(x_forward, is_training)
        x_backward = tf.reverse_sequence(self.lstm_backward(x_backward, is_training), length, seq_axis=1)

        x = tf.concat((x_forward, x_backward), axis=-1)
        y = self.alpha(x)
        latent = y * x_forward + (1 - y) * x_backward
        latent = self.layer_norm(self.dense(tf.boolean_mask(latent, tf.sequence_mask(length, max_num_visit))))
        output_left = self.dense_left(latent)
        output_right = self.dense_right(latent)
        output_stay = self.dense_stay(latent)
        output_diagnosis = self.dense_diagnosis(latent)
        output_age = self.dense_age(latent)
        return output_diagnosis, output_stay, output_left, output_right, output_age


def cal_loss(code, others, length, output):
    output_diagnosis, output_stay, output_left, output_right, output_age = output
    mask = tf.sequence_mask(length, max_num_visit)
    left = tf.boolean_mask(others[:, :, 27:37], mask)
    stay = tf.boolean_mask(others[:, :, 19:27], mask)
    age = tf.boolean_mask(others[:, :, :17], mask)
    right = tf.boolean_mask(tf.reverse_sequence(tf.concat((tf.expand_dims(others[:, 0, 27:37], axis=-2),
                                                           tf.reverse_sequence(others[:, 1:, 27:37], length - 1,
                                                                               seq_axis=1)), axis=-2), length,
                                                seq_axis=1), mask)
    feature = tf.boolean_mask(tf.concat((tf.reduce_sum(tf.one_hot(code, num_code, dtype=tf.float32), axis=-2),
                                         others[:, :, 37:]), axis=-1), mask)

    left_loss = tf.reduce_sum(left * output_left,axis=-1)
    right_loss = tf.reduce_sum(right * output_right,axis=-1)
    stay_loss = tf.reduce_sum(stay * output_stay,axis=-1)
    age_loss = tf.reduce_sum(age * output_age,axis=-1)

    left_loss = tf.reduce_sum(left_loss * (1 - tf.math.exp(left_loss)) ** 2)
    right_loss = tf.reduce_sum(right_loss * (1 - tf.math.exp(right_loss)) ** 2)
    stay_loss = tf.reduce_sum(stay_loss * (1 - tf.math.exp(stay_loss)) ** 2)
    age_loss = tf.reduce_sum(age_loss * (1 - tf.math.exp(age_loss)) ** 2)
    feature_loss = tf.reduce_sum(focal(y_true=feature, y_pred=output_diagnosis))
    loss = (feature_loss - right_loss - left_loss - stay_loss - age_loss) / tf.cast(tf.reduce_sum(length), tf.float32)
    return loss


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
        adam_config = super(AdamWeightDecay, self).get_config()
        adam_config.update({
            'weight_decay_rate': self.weight_decay_rate,
        })
        return adam_config

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
    length_test = np.load('single/length_train_1.npy')
    feature_test = np.load('single/code_train_1.npy')
    summary_test = np.load('single/others_train_1.npy')

    length_train = np.load('single/length_train_0.npy')
    feature_train = np.load('single/code_train_0.npy')
    summary_train = np.load('single/others_train_0.npy')

    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])

    with strategy.scope():
        dataset_train = tf.data.Dataset.from_tensor_slices((feature_train.astype('int32'),
                                                            summary_train.astype('float32'),
                                                            length_train)).shuffle(4096, reshuffle_each_iteration=True)
        parsed_dataset_train = dataset_train.batch(config.batchsize * NUM_GPU, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)
        dist_dataset_train = strategy.experimental_distribute_dataset(parsed_dataset_train)

        dataset_val = tf.data.Dataset.from_tensor_slices((feature_test.astype('int32'),
                                                          summary_test.astype('float32'),
                                                          length_test))
        parsed_dataset_val = dataset_val.batch(config.batchsize * NUM_GPU, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)
        dist_dataset_val = strategy.experimental_distribute_dataset(parsed_dataset_val)

        del feature_train, feature_test, summary_train, summary_test

        optimizer = AdamWeightDecay(learning_rate=1e-4)
        model = Model()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        def one_step(batch, is_training):
            code, others, length = batch
            length = tf.squeeze(length)
            with tf.GradientTape() as tape:
                output = model(code, others, length, is_training)
                loss = cal_loss(code, others, length, output)
            if is_training:
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        @tf.function
        def function_wrap(batch, is_training):
            print('new_graph')
            return strategy.experimental_run_v2(one_step, (batch, is_training))

        def distributed_train_epoch(ds, is_training=True):
            total_loss = 0.0
            step = 0.0
            for batch in ds:
                per_replica_loss = function_wrap(batch, is_training)
                total_loss += strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
                step += 1

            return total_loss / step

    print('training start')
    for epoch in range(300):
        start_time = time.time()
        with strategy.scope():
            loss_train = distributed_train_epoch(dist_dataset_train)
            loss_test = distributed_train_epoch(dist_dataset_val, False)

        duration_epoch = int((time.time() - start_time) / 60)
        format_str = 'epoch: %d, train_loss_d = %f, test_loss_d = %f (%d)'
        print(format_str % (epoch, loss_train.numpy(),
                            loss_test.numpy(),
                            duration_epoch))
        if epoch % 10 == 9:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    config = Config()
    train()
