# MaskGAN

This project was forked from https://github.com/tensorflow/models/tree/master/research/maskgan
For hacking.


Unfortunately there is an incompatibility in the libraries used in this project.  The model saves checkpoints that cannot then be restored.  This issue is noted at the bottom of the project's README.  It prevented us from moving forward with this project.

Below is what I did... before quitting due to the bad checkpoints.



### Environment 

Note:
- tensorflow 2.0 does not work here, so we use v1.5
- But tensorflow 1.5 has specifications: python[version='2.7.\*|3.5.\*|3.6.\*']
- Moreover, this code was written for python 2.  Yes it worked to modify code for python 3.6.* but to stay safe, this environment uses python 2.7.



(Do these steps once)

Add channel for conda to find old version tensorflow 1.5:
```
$ conda config --add channels conda-forge
```
Make the environment
```
$ conda create --name maskGAN python=2.7 tensorflow=1.5 scipy
```


(Do these steps for each hack)
```
$ conda activate maskGAN
```
.... later ...
```
$ conda deactivate
```


### PTB Data

To test this out, I used the example PTB dataset.

I download this dataset from here:  https://github.com/wojzaremba/lstm/tree/master/data

And then put it in the ./data directory



### Running

Set environment variables to make this clearer... could later modify code instead.

```
export MASKGAN_DATA_DIR='./data'; export MASKGAN_BASE_DIR='./tmp'
```

1. Pretraining:

```

python train_mask_gan.py \
 --data_dir=$MASKGAN_DATA_DIR \
 --batch_size=15 \
 --sequence_length=20 \
 --base_directory=$MASKGAN_BASE_DIR \
 --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,dis_num_layers=2,gen_learning_rate=0.00074876,dis_learning_rate=5e-4,baseline_decay=0.99,dis_train_iterations=1,gen_learning_rate_decay=0.95" \
 --mode='TRAIN' \
 --max_steps=100000 \
 --generator_model='seq2seq_vd' \
 --discriminator_model='rnn_zaremba' \
 --is_present_rate=0.5 \
 --summaries_every=10 \
 --print_every=50 \
 --max_num_to_print=3 \
 --gen_training_strategy=cross_entropy \
 --seq2seq_share_embedding



```

Run MaskGAN in GAN mode

```
python train_mask_gan.py \
 --data_dir=$MASKGAN_DATA_DIR \
 --batch_size=20 \
 --sequence_length=20 \
 --base_directory=$MASKGAN_BASE_DIR \
 --mask_strategy=contiguous \
 --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,dis_num_layers=2,gen_learning_rate=0.000038877,gen_learning_rate_decay=1.0,gen_full_learning_rate_steps=2000,gen_vd_keep_prob=0.33971,rl_discount_rate=0.89072,dis_learning_rate=5e-4,baseline_decay=0.99,dis_train_iterations=2,dis_pretrain_learning_rate=0.005,critic_learning_rate=5.1761e-7,dis_vd_keep_prob=0.71940" \
 --mode='TRAIN' \
 --max_steps=1000 \
 --generator_model='seq2seq_vd' \
 --discriminator_model='seq2seq_vd' \
 --is_present_rate=0.5 \
 --summaries_every=250 \
 --print_every=250 \
 --max_num_to_print=3 \
 --gen_training_strategy='reinforce' \
 --seq2seq_share_embedding=true \
 --baseline_method=critic \
 --attention_option=luong

 not using for now:

 --maskgan_ckpt='/tmp' \

```

Generate samples:
```
python generate_samples.py \
 --data_dir=$MASKGAN_DATA_DIR \
 --data_set=ptb \
 --batch_size=256 \
 --sequence_length=20 \
 --base_directory ./tmp/ptbsample/ \
 --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,gen_vd_keep_prob=0.33971" \
 --generator_model=seq2seq_vd \
 --discriminator_model=seq2seq_vd \
 --is_present_rate=0.0 \
 --maskgan_ckpt=/tmp/maskGAN \
 --seq2seq_share_embedding=True \
 --dis_share_embedding=True \
 --attention_option=luong \
 --mask_strategy=contiguous \
 --baseline_method=critic \
 --number_epochs=4
```


### What I'm doing...


for pretrain, it takes. a while and there's a lot of. data.
I just want to see w hat happens with the PTB data before using mobility data.

So I cut the PTB data into some subsets.

Before doing this, wasn't letting pretraining finish
Then when doing step 2, there were issues with the checkpoints not found.  to deal with issues:
- maybe letting model finish on data subset will work?
- or run as if there is no pretraining stem and try setting `maskgan_ckpt` to empty




converting checkpoints

python checkpoint_convert.py ./tmp/train/model.ckpt /tmp/my_converted_checkpoint/model.ckpt

doesn't work

this is a deal breaker.




# MaskGAN: Better Text Generation via Filling in the ______

Code for [*MaskGAN: Better Text Generation via Filling in the
______*](https://arxiv.org/abs/1801.07736) published at ICLR 2018.

## Requirements

*   TensorFlow >= v1.5

## Instructions

Warning: The open-source version of this code is still in the process of being
tested. Pretraining may not work correctly.

For training on PTB:

1. Follow instructions here ([Tensorflow RNN Language Model Tutorial](https://www.tensorflow.org/tutorials/sequences/recurrent)) to train a language model on PTB dataset.
Copy PTB data downloaded from the above tensorflow RNN tutorial to folder "/tmp/ptb". It should contain following three files: ptb.train.txt, ptb.test.txt, ptb.valid.txt
Make folder /tmp/pretrain-lm and copy checkpoints from above Tensorflow RNN tutorial under this folder.


2. Run MaskGAN in MLE pretraining mode. If step 1 was not run*, set
`language_model_ckpt_dir` to empty.

```bash
python train_mask_gan.py \
 --data_dir='/tmp/ptb' \
 --batch_size=20 \
 --sequence_length=20 \
 --base_directory='/tmp/maskGAN' \
 --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,dis_num_layers=2,gen_learning_rate=0.00074876,dis_learning_rate=5e-4,baseline_decay=0.99,dis_train_iterations=1,gen_learning_rate_decay=0.95" \
 --mode='TRAIN' \
 --max_steps=100000 \
 --language_model_ckpt_dir=/tmp/pretrain-lm/ \
 --generator_model='seq2seq_vd' \
 --discriminator_model='rnn_zaremba' \
 --is_present_rate=0.5 \
 --summaries_every=10 \
 --print_every=250 \
 --max_num_to_print=3 \
 --gen_training_strategy=cross_entropy \
 --seq2seq_share_embedding
```

3. Run MaskGAN in GAN mode. If step 2 was not run, set `maskgan_ckpt` to empty.
```bash
python train_mask_gan.py \
 --data_dir='/tmp/ptb' \
 --batch_size=128 \
 --sequence_length=20 \
 --base_directory='/tmp/maskGAN' \
 --mask_strategy=contiguous \
 --maskgan_ckpt='/tmp/maskGAN' \
 --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,dis_num_layers=2,gen_learning_rate=0.000038877,gen_learning_rate_decay=1.0,gen_full_learning_rate_steps=2000000,gen_vd_keep_prob=0.33971,rl_discount_rate=0.89072,dis_learning_rate=5e-4,baseline_decay=0.99,dis_train_iterations=2,dis_pretrain_learning_rate=0.005,critic_learning_rate=5.1761e-7,dis_vd_keep_prob=0.71940" \
 --mode='TRAIN' \
 --max_steps=100000 \
 --generator_model='seq2seq_vd' \
 --discriminator_model='seq2seq_vd' \
 --is_present_rate=0.5 \
 --summaries_every=250 \
 --print_every=250 \
 --max_num_to_print=3 \
 --gen_training_strategy='reinforce' \
 --seq2seq_share_embedding=true \
 --baseline_method=critic \
 --attention_option=luong
```

4. Generate samples:
```bash
python generate_samples.py \
 --data_dir /tmp/ptb/ \
 --data_set=ptb \
 --batch_size=256 \
 --sequence_length=20 \
 --base_directory /tmp/imdbsample/ \
 --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,gen_vd_keep_prob=0.33971" \
 --generator_model=seq2seq_vd \
 --discriminator_model=seq2seq_vd \
 --is_present_rate=0.0 \
 --maskgan_ckpt=/tmp/maskGAN \
 --seq2seq_share_embedding=True \
 --dis_share_embedding=True \
 --attention_option=luong \
 --mask_strategy=contiguous \
 --baseline_method=critic \
 --number_epochs=4
```


*  While trying to run Step 2, the following error appears:
   NotFoundError (see above for traceback): Restoring from checkpoint failed. This is most likely due to a Variable name or other graph    key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original    error:

   Key critic/rnn/biases not found in checkpoint
   [[node save/RestoreV2 (defined at train_mask_gan.py:431) ]]

   This is an issue with seq2seq model because it uses the attention mechanism.
   The issue arises if you saved the model with an earlier version (seq2seq is old) and restore with a recent one (saver.restore got updated).
   The naming convention for LSTM parameters changed, e.g. cell_0/basic_lstm_cell/weights became cell_0/basic_lstm_cell/kernel.
   Which is why you cannot restore them if you try to restore old checkpoints with recent TF.
   The below script will help rename the variables and everything will work as expected.
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/tools/checkpoint_convert.py

## Contact for Issues

*   Liam Fedus, @liamb315 <liam.fedus@gmail.com>
*   Andrew M. Dai, @a-dai <adai@google.com>
