# seqseq-autoencoder
This is a simple seqseq-autoencoder example of tensorflow-0.9

tensorflow中的机器翻译示例代码在我看来并不是一个很好的seq2seq例子，它用的一些包装好的函数并没有简化事情，反倒让自己对初学者来说很难理解。于是我写了一个用tied-seq2seq(编码和解码用同一个神经网络)做短句子自编码器的例子，尽量简单，尽量注释。

The nueral tranlation example in trensorflow can hardly be called a good example of seqence-to-sequence model. The functions it uses make tensorflow beginers like me puzzled. So I write a tie-seq2seq (which means the encoder and decoder use the same rnn network) short text auto encoder example, with simple structure and detailed comments. 


##1.Data
There is a chinese address dataset in data/address.txt, you can play with it, or you can use any whitespace-splited data with each token a single word and each line a single sequence.


##2.How to run
python train.py --data-path data/address.txt --model-path train
for more options: python train.py -h














