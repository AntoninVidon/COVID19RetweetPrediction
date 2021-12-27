# COVID19 Retweet Prediction

## TeamWeaver

![jupyter notebook x python](https://img.shields.io/badge/jupyter%20notebook-python-orange)![DeepLearning x DNN](https://img.shields.io/badge/DeepLearning-DNN-blue)![ML x Regressions](https://img.shields.io/badge/MachineLearning-Regressions-ff69b4)

We worked relentlessly as a united team to accurately predict the number of retweets a tweet can get. We started by extracting relevant features and then applying Machine Learning and Deep Learning concepts we learned through out the course.

## Path to original data

We expect you to put the original data in the `Data` folder of our directory.

## Folder Hierarchy explained

### Preprocessing

In the **preprocessing** folder, you will find `General_preprocessing.ipynb`. Execute all cells in order to get the preprocessed data. This data is used by basic **regression models**, **Simon** and **textual clustering**.

### Visualisation

In the folder **Visualisation** you will find `Data_visualisation.ipynb`. You can execute it to visualise the correlation matrix and other miscellaneous plots. 

### Text clustering

In the **text clustering** folder you will find `Kmeans_Text.ipynb`  and `Regression_using_clusters.ipynb`. You will need to execute `Kmeans_Text.ipynb` first to compute the clusters that will be saved in the folder `Data`. When the script terminates, run `Regression_using_clusters.ipynb`. It will predict retweets on each cluster.

### Basic Regression

In the **Basic Regression** folder, you will find `Basic_Regression_Models.ipynb`. Execute all cells in order to train and evaluate all our basic regression models.

### Simon neural network model

In the folder named **Simon-NN**, you will find `Simon-NN.ipynb`. By executing this notebook, you will train your own version of **Simon**, our fist neural network model.

### DNN_BERT neural network model

In the folder named **DNN_BERT**, you will find `BERT_Tokenizer for Tweets.ipynb`  and `DNN_BERT.ipynb`. You will need to execute `BERT_Tokenizer for Tweets.ipynb` first to compute the tokens that will be saved in the folder `Tensors`. When the script terminates, run `DNN_BERT.ipynb`. By executing this notebook, you will train your own version of **DNN_BERT**, our second neural network model.