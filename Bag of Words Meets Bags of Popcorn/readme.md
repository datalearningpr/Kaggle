# Bag of Words Meets Bags of Popcorn
##### Classification
This one can be a good game for learning **NLP**.

The following basic skills shall be picked up after playing this game:
* **Bag of words**, count the word frequency of each given text
* **tf-idf**, tf is the frequency of word in the document, idf is the log(number of document / (1 + document with the word)), tf-idf = tf * idf, which has the meaning as: if a word appears a lot in the document but it is a rare word of all documents, it means the word is very important.
* **word2vec**, Google version of doing word embedding. In old times, NLP is using one hot coding to mark each word, since there will be huge amount of words, the dimension is too big, and each word is separated from each other. word2vec transfer the large dimension to way lower dimension and it number of each dimension has actual meanings, it will place the similar words together.
* **DeepNLP**, train the word embedding with the data we have, and then we can use CNN or RNN to further power up the network, RNN is too slow, not tested in this game, CNN is tested.
* bags of words, tf-idf is non deep learning skills, for small datasets it is sufficient enough to give good results. with DeepNLP, the word embedding with CNN gives the best result, shows the power of deep learning.
###### Hope this script will be helpful
