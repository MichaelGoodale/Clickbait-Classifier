# Clickbait Classifier

This is a convolutional neural network trained to distinguish sober headlines from more clickbaity ones. It's an implementation of Yoon Kim's 2014 paper, [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882). It is trained using a dataset pulled from reddit's news subreddits, several RSS feeds from Reuters, BBC, and CBC, while the clickbait comes from Buzzfeed and Viralnova. The word vectors used are pretrained, namely from [GloVe](http://nlp.stanford.edu/projects/glove/). It is implementated using Tensorflow, with a web app made in Flask which communicates with a Tensorflow Serving server. 

The model gets 95.95% accuracy and a F1 score of 0.9705 after 5 epochs. 

There is a public demo of the classifier [here](http://clickbait-classify.michaelgoodale.com/)
##Examples
###The New York Times
![Front page of the New York Times](https://cloud.githubusercontent.com/assets/1775699/21063799/86d88e00-be25-11e6-85fb-53ca4bcf848b.png)
###Buzzfeed
![Front page of BuzzFeed](https://cloud.githubusercontent.com/assets/1775699/21063800/86e7d432-be25-11e6-9e42-5f8969b0938d.png)
