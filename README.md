# Vietnamese Sentiment Analysis on VLSP 2016 Dataset - A Comparative Study of Modern Deep Learning Approaches in NLP
This project is a comprehensive exploration of various deep learning models for sentiment analysis on the Vietnamese language, utilizing the benchmark VLSP 2016 (Task SA) dataset. The primary goal is to classify user comments into three categories: positive, negative, and neutral.
The study implements and compares traditional RNN/CNN architectures using pre-trained Word2Vec embeddings against a modern Transformer-based model, PhoBERT. The results demonstrate the significant performance gains achieved through transfer learning with large, pre-trained language models.

## The main objectives of this project are:
Build and Train: To implement and train several deep learning models for Vietnamese sentiment analysis, including:
1. Convolutional Neural Network (CNN)
2. Long Short-Term Memory (LSTM) & Bidirectional LSTM
3. A hybrid CNN-LSTM architecture
4. A Transformer-based model (PhoBERT)

Evaluate Performance: To systematically evaluate each model's performance on the test set using standard metrics like Accuracy, Precision, Recall, and F1-Score.

Compare and Contrast: To analyze the results, highlighting the strengths and weaknesses of each approach and concluding with the most effective model for this specific task.
Dataset

## Methodology
The project follows a standard NLP pipeline for each model architecture.

**Preprocessing**

The raw text data undergoes the following preprocessing steps:
- Removal of all numerical digits.
- Vietnamese word tokenization and conversion to lowercase using the pyvi library.
- Label encoding:
+ One-hot encoding for CNN/LSTM models (e.g., -1 -> [1,0,0]).
+ Label encoding for the PhoBERT model (e.g., -1, 0, 1 -> 0, 1, 2).

**Word Embeddings** 

Two types of embeddings are used to convert text into numerical representations:
- Word2Vec: A pre-trained Vietnamese Word2Vec model (vi-model-CBOW.bin) with 400 dimensions is used to initialize the embedding layer for the CNN and LSTM models. The embedding layer is set to be trainable (trainable=True) to allow for fine-tuning during training.
- PhoBERT Embeddings: The vinai/phobert-base model inherently creates contextualized embeddings for each token, which are fine-tuned during the training process for the sentiment analysis task.

**Results**
The performance of each model was evaluated on the 1,050 samples in the test set. The table below summarizes the key results (using Macro Average for Precision, Recall, and F1-Score).

| Model | Accuracy | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| CNN (Word2Vec) | 62.00% | 0.6545 | 0.6200 | 0.5998 |
| LSTM (Word2Vec) | 64.29% | 0.6436 | 0.6429 | 0.6432 |
| CNN+LSTM (Word2Vec) | 61.52% | 0.6128 | 0.6152 | 0.6115 |
| **PhoBERT** | **73.14%** | **0.7329** | **0.7314**| **0.7314**|

> The PhoBERT model significantly outperformed all other models, achieving the highest accuracy of 73.14%. This highlights the effectiveness of transfer learning and the superior ability of Transformer architectures to understand deep contextual and semantic nuances in the Vietnamese language.

## Prerequisites
- Python 3.8+
- Frameworks: TensorFlow, Keras, PyTorch (via Transformers), Hugging Face Transformers, Scikit-learn, Gensim, pyvi, underthesea, Pandas, NumPy, Matplotlib, Seaborn

## Acknowledgments
- This project was completed as part of the "Modern Approaches in Natural Language Processing" course at Ho Chi Minh City University of Technology (HCMUT).
