MY_README -- what i did



### Environment 


tensorflow 2.0 does not work here, so we use v1.5
But tensorflow 1.5 has specifications:
Specifications:

  - tensorflow=1.5 -> python[version='2.7.*|3.5.*|3.6.*']
  - tensorflow=1.5 -> python[version='>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0']

Moreover, this code was written for python 2.7

two options, can modify code for python 3.6.* OR can use environment with python2.7


Add channel for tensorflow 1.5:

$ conda config --add channels conda-forge
$ conda create --name maskGAN python=3.6 tensorflow=1.5 scipy


$ conda create --name maskGAN python=2.7 tensorflow=1.5 scipy

$ conda activate maskGAN



.... later ...

$ conda deactivate


Other notes:
-  this was made for python 2.something (see dict.iteritems())
- maybe need to use python 2 instead?
- for now. making. change: replated. iteritems() with. items()


### PTB Data

To test this out, I used the example PTB dataset.

I download this dataset from here:  https://github.com/wojzaremba/lstm/tree/master/data

And then put it in the ./data directory



### Running

```

python train_mask_gan.py \
 --data_dir='./data' \
 --batch_size=15 \
 --sequence_length=20 \
 --base_directory='./tmp' \
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

not using...

 --language_model_ckpt_dir=/tmp/pretrain-lm/ \

```

