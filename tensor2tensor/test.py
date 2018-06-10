# -*- coding: utf-8 -*-
import numpy as np


def test_get_weigts():
  """
    layers/modalities.py: get_weigts()
  """
  vocab_size = 8236
  num_shards = 16
  shard_size = []
  for i in range(16):
    shard_size.append((vocab_size // num_shards) +
                      (1 if i < vocab_size % num_shards else 0))


def test_tf_reduce():
  """

  modalities.py classlabel top():
  x = body_output
  x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
  """
  raw = np.load("./test_sg/raw.npy")
  reduced = np.load("./test_sg/redu.npy")

  i = 10
  j = 1
  k = 120
  entry0 = raw[i, :, 0, 0]
  entry0 = np.mean(entry0)
  print(np.allclose(entry0, reduced[i, 0, 0, 0]))


def transformer_encoder_hparams():
  hparams = {
      'batch_size':
          4096,
      'use_fixed_batch_size':
          False,
      'num_hidden_layers':
          2,
      'kernel_height':
          3,
      'kernel_width':
          1,
      'hidden_size':
          128,
      'compress_steps':
          0,
      'dropout':
          0.2,
      'clip_grad_norm':
          0.0,
      'grad_noise_scale':
          0.0,
      'summarize_grads':
          False,
      'summarize_vars':
          False,
      'initializer':
          'uniform_unit_scaling',
      'initializer_gain':
          1.0,
      'label_smoothing':
          0.1,
      'optimizer':
          'Adam',
      'optimizer_adam_epsilon':
          1e-09,
      'optimizer_adam_beta1':
          0.9,
      'optimizer_adam_beta2':
          0.997,
      'optimizer_momentum_momentum':
          0.9,
      'optimizer_momentum_nesterov':
          False,
      'optimizer_adafactor_beta1':
          0.0,
      'optimizer_adafactor_beta2':
          0.999,
      'optimizer_adafactor_factored':
          True,
      'optimizer_adafactor_decay_type':
          'pow',
      'optimizer_adafactor_memory_exponent':
          0.8,
      'optimizer_adafactor_clipping_threshold':
          1.0,
      'optimizer_adafactor_multiply_by_parameter_scale':
          True,
      'weight_decay':
          0.0,
      'weight_noise':
          0.0,
      'learning_rate_schedule':
          'constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size',
      'learning_rate_constant':
          2.0,
      'learning_rate_decay_scheme':
          'noam',
      'learning_rate_decay_steps':
          5000,
      'learning_rate_decay_staircase':
          False,
      'learning_rate_minimum':
          None,
      'learning_rate_decay_rate':
          1.0,
      'learning_rate_warmup_steps':
          8000,
      'learning_rate_cosine_cycle_steps':
          250000,
      'learning_rate':
          0.2,
      'sampling_method':
          'argmax',
      'sampling_temp':
          1.0,
      'factored_logits':
          False,
      'multiply_embedding_mode':
          'sqrt_depth',
      'moe_hidden_sizes':
          '2048',
      'moe_num_experts':
          16,
      'moe_k':
          2,
      'moe_loss_coef':
          0.001,
      'layer_preprocess_sequence':
          'n',
      'layer_postprocess_sequence':
          'da',
      'layer_prepostprocess_dropout':
          0.1,
      'layer_prepostprocess_dropout_broadcast_dims':
          '',
      'symbol_dropout':
          0.0,
      'norm_type':
          'layer',
      'norm_epsilon':
          1e-06,
      'symbol_modality_num_shards':
          16,
      'min_length':
          0,
      'max_length':
          256,
      'min_length_bucket':
          8,
      'length_bucket_step':
          1.1,
      'eval_drop_long_sequences':
          False,
      'eval_run_autoregressive':
          False,
      'shared_embedding_and_softmax_weights':
          0,
      'symbol_modality_skip_top':
          False,
      'input_modalities':
          'default',
      'target_modality':
          'default',
      'max_input_seq_length':
          0,
      'max_target_seq_length':
          0,
      'split_to_length':
          0,
      'prepend_mode':
          'none',
      'scheduled_sampling_prob':
          0.0,
      'scheduled_sampling_warmup_steps':
          50000,
      'scheduled_sampling_gold_mixin_prob':
          0.5,
      'daisy_chain_variables':
          True,
      'force_full_predict':
          False,
      'no_data_parallelism':
          False,
      'activation_dtype':
          'float32',
      'weight_dtype':
          'float32',
      'filter_size':
          512,
      'num_encoder_layers':
          0,
      'num_decoder_layers':
          0,
      'num_heads':
          4,
      'attention_key_channels':
          0,
      'attention_value_channels':
          0,
      'ffn_layer':
          'dense_relu_dense',
      'parameter_attention_key_channels':
          0,
      'parameter_attention_value_channels':
          0,
      'attention_dropout':
          0.1,
      'attention_dropout_broadcast_dims':
          '',
      'relu_dropout':
          0.1,
      'relu_dropout_broadcast_dims':
          '',
      'pos':
          'timing',
      'nbr_decoder_problems':
          1,
      'proximity_bias':
          False,
      'use_pad_remover':
          True,
      'self_attention_type':
          'dot_product',
      'max_relative_position':
          0,
      'conv_first_kernel':
          3,
      'moe_overhead_train':
          1.0,
      'moe_overhead_eval':
          2.0,
      'model_dir':
          '/project/chli/capstrans_encoder/t2t_train/sentiment_imdb/capstrans_encoder-transformer_tiny',
      'data_dir':
          '/project/chli/capstrans_encoder/t2t_data',
      'train_steps':
          2000,
      'eval_steps':
          100,
      'mode':
          'train'
  }
