# SentimentAnalysis-imdb
Sentiment Analysis of the imdb data 

* Use Recurrent Neural Networks, and in particular LSTMs, to perform sentiment analysis in Keras.
* Keras has a built-in IMDb movie reviews dataset that we can use.
* Simplification made in this code is to limit the task to binary classification. The dataset actually includes a more fine-grained review rating that is indicated in each review's filename (which is of the form `<[id]_[rating].txt>` where `[id]` is a unique identifier and `[rating]` is on a scale of 1-10; note that neutral reviews > 4 or < 7 have been excluded).

Accuracy is **87.48%**
