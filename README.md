# Machine Learning Links And Lessons Learned
A summary of the resources I have come across and the lessons I have learned while studying machine learning. I've been Inspired to organize my findings by Adit Deshpande's work found on his repo [here](https://github.com/adeshpande3).

* [General](#general)
* [Hyperparameters](#hyperparameters)
* [Deep Learning](#deep-learning)
* [CNNs](#cnns)
* [RNNs](#rnnss)
* [Learning Resources](#learning-resources)
* [Datasets](*datasets)
* [Research Papers](#research-papers)
* [Interesting Links](#interesting-links)
* [Other](#other)

## General

* Adversarial training: training your network on adversarial manipulations of your training set to make them more resillient to adversarial attacks

## Hyperparameters

* Need to update

## Deep Learning

* Need to update

## CNNs

* Need to update

## RNNs

* Need to update

## Learning Resources
* [tensorflow-seq2seq-tutorials](https://github.com/ematvey/tensorflow-seq2seq-tutorials)
* [Sequence to Sequence Deep Learning (Quoc Le, Google](https://www.youtube.com/watch?v=G5RY_SUJih4)
* [Improved techniques for trainings GANs](https://arxiv.org/pdf/1606.03498.pdf)
* [Google python style guide](https://google.github.io/styleguide/pyguide.html)
* [Tflearn examples](https://github.com/tflearn/tflearn/tree/master/examples#tflearn-examples)
* [Open source facial recognition with deep learning](https://cmusatyalab.github.io/openface/)

### Generative Adversarial Networks
* [GANs for beginners](https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners)
* [2016 NIPS workshop on adversarial training](https://www.youtube.com/watch?v=X1mUN6dD8uE)
* [How to train a GAN (GANhacks)](https://github.com/soumith/ganhacks)
* [Selecting batch size vs number of epochs](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)
* [GAN stability](http://www.araya.org/archives/1183)
* [MNIST GAN with Keras](https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)
* [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
* [Another DCGAN](https://github.com/yihui-he/GAN-MNIST)
* [DiscoGAN](https://github.com/carpedm20/DiscoGAN-pytorch)
* [Beta1 hyperparameter values](https://arxiv.org/pdf/1511.06434.pdf)
* [WGANs](https://paper.dropbox.com/doc/Wasserstein-GAN-GvU0p2V9ThzdwY3BbhoP7)
* [OpenAI blog on GANs](https://blog.openai.com/generative-models/)
* [Adam Geitgey's blog on GANs](https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7)



### Hyperparameters
* [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) by Yoshua Bengio
* [Deep Learning Book - chapter 11.4: Selecting Hyperparameters](http://www.deeplearningbook.org/contents/guidelines.html) by Ian Goodfellow, Yoshua Bengio, Aaron Courville
* [Neural Networks and Deep Learning Book - Chapter 3: How to choose a neural network's hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters) by Michael Nielsen
* [Efficient Backprop (pdf)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Yann LeCun
* [Kernel size for CNNs](https://www.quora.com/How-can-I-decide-the-kernel-size-output-maps-and-layers-of-CNN)

### Word2Vec and Word Embeddings
* [Word2Vec Overview](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
* [Tensorflow Word2Vec Tutorial](https://www.tensorflow.org/tutorials/word2vec)

## Datasets
* [Collection of 80+ Datasets](https://docs.google.com/spreadsheets/d/1AQvZ7-Kg0lSZtG1wlgbIsrm90HaTZrJGQMz-uKRRlFw/edit#gid=0)
* [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
	* Performed linear regression to predict housing prices
* [CIFAR-10 and CIFAR-100 Datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
	* CIFAR-10: 60000 32x32 colour images in 10 classes (6000 images per class)
	* CIFAR-100: CIFAR-10 dataset with 100 classes and 600 images per class. 
* [Microsoft Coco](http://cocodataset.org/#home)
	* Object detection, segmentation and captioning dataset with ~330K images (>200K labeled), 1.5M object instances, 80 object categories, 91 stuff categories and 5 captions per image.
* [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/)
	* A popular and well understood dataset of handwritten digits used as a benchmark to test new algorithms and approaches. 
	* 28x28 images with 60,000 training and 10,000 test examples
	* Used to train my first neural network and to play around with autoencoders and denoisers
* [text8 Dataset](http://mattmahoney.net/dc/textdata.html)
	* Wikipedia article dataset 
* [Cornell Movie Dialog Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
	* Contains over 200,000 conversational exchanges from ~600 movies
	* [Stanford Chatbot Exercise](https://github.com/chiphuyen/tf-stanford-tutorials/tree/master/assignments/chatbot)
* [French-English Translation Corpus](http://www.statmt.org/wmt10/training-giga-fren.tar)
* [Celebrity Faces Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): Dataset of over 200,000 annotated celebrity faces
* Screenshots of NES games from [The Video Game Museum Website](http://www.vgmuseum.com/nes.htm). Can be extracted using [wget](https://www.gnu.org/software/wget/)
* [Caltech Birds Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html): Dataset with ~12,000 images of birds
* [European Paliament Proceedings](http://www.statmt.org/europarl/): Text data translated into 21 languages from 1996 to 2011


## Research Papers

* [Generating image captions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)
* [Deep Visual-Semantic Alignments for Generating Image Descriptions](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf): Andrej Karpathy & Li Fei-Fei
* [Generating pictures from a description](https://arxiv.org/pdf/1506.03500.pdf)
* [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)
* [Practical Black-Box Attacks Against Machine Learning](https://arxiv.org/abs/1602.02697)
* [Defensive Distillation](https://arxiv.org/abs/1511.04508)
* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
* [Adversarial Examples in the Physical World](https://arxiv.org/abs/1607.02533)

## Interesting Links

* [Google Word2Vec](https://code.google.com/archive/p/word2vec/)
* [Image-to-Image Demo](https://affinelayer.com/pixsrv/) by Christopher Hesse
* [CycleGAN](https://github.com/junyanz/CycleGAN): examples of GANs applied to transfer image styles, change primary objects etc. 
* [List of Popular GANs](https://github.com/wiseodd/generative-models)
* [A neural parametric singing synthesizer](http://www.creativeai.net/posts/W2C3baXvf2yJSLbY6/a-neural-parametric-singing-synthesizer)
* [FaceApp](http://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/)
* [Python-based facial recognition library](https://github.com/ageitgey/face_recognition#face-recognition)
* [Kaggle competititon for adversarial training](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack)


## Other

* Need to update
