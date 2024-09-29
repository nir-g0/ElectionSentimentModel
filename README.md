# Election Sentiment Analysis Project

This project focuses on analyzing sentiments around election-related tweets and threads to determine if they are positive, negative, or neutral towards specific candidates. Our goal is to leverage machine learning models to extract insights and trends from social media discussions about the election.

## Table of Contents
- [Overview](#overview)
- [Project Versions](#project-versions)
  - [Version 1: Naive Bayes Classifier](#version-1-naive-bayes-classifier)
  - [Version 2: Convolutional Neural Network (CNN)](#version-2-convolutional-neural-network-cnn)
  - [Version 3: Transitioning to BERT](#version-3-transitioning-to-bert)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

In this project, I aim to classify the sentiment of election-related tweets. The sentiment can be classified into three categories: positive, neutral, and negative. The current implementation explores different machine learning models as I iterate on improving the overall accuracy of the sentiment classification.

## Project Versions

### Version 1: Naive Bayes Classifier
The initial version of this project utilized a **Naive Bayes Classifier**, a simple but effective probabilistic model for text classification. Despite its simplicity, this model performed well on small datasets but struggled with more complex relationships within the text, especially for more nuanced sentiments.

Key features:
- Simplicity and quick setup.
- Moderate accuracy for basic sentiment classification.
- Limitations in understanding context and more subtle tones.

### Version 2: Convolutional Neural Network (CNN)
In version 2, I moved to a **Convolutional Neural Network (CNN)**. This deep learning model provided better feature extraction, especially for sequences like text. The CNN was able to capture more complex relationships between words, improving accuracy on our larger dataset of election tweets.

Key features:
- Increased accuracy compared to Naive Bayes.
- Better at capturing word sequences and patterns.
- Requires more computational resources.
- Not ideal for capturing long-range dependencies in sentences.

### Version 3: Transitioning to BERT (In Progress)
Currently, I am transitioning to using **BERT (Bidirectional Encoder Representations from Transformers)**. BERT is designed to pre-train deep bidirectional representations, making it highly effective at capturing context and meaning in text. This model is expected to significantly improve the accuracy of sentiment classification by understanding context at a much deeper level.

Key features:
- Excellent at capturing context in sentences.
- Handles long-range dependencies better than CNNs.
- Significantly higher computational requirements.
- Ongoing work on fine-tuning and optimization.

### UI Addition (Post-model creation)
Once the ML model is refined enough to handle real data with a good accuracy, I will work on developing a modern web UI with the use of react.js. The UI will help visualize data as well as make searching sentiments a more interactive experience.


## Data

The dataset used consists of election-related tweets and threads. Currently, the dataset is a mix of labeled sentiments: positive, neutral, and negative. The data was sourced from public social media APIs and is continuously updated.

## Preprocessing

To prepare the data for model training, I perform:
- Text tokenization.
- Stopword removal.
- Lemmatization.
- Removing special characters and URLs.
- Balancing the dataset across sentiment categories.

I are currently improving preprocessing techniques to enhance the model's performance and are considering advanced methods like word embeddings and data augmentation.

## Results

| Model       | Accuracy (on training/test data)|
|-------------|----------                       |
| Naive Bayes | 73% (Poor on real data)         |
| CNN         | 94% (Poor on real data)         | 
| BERT        | In progress                     |

More detailed performance metrics will be included as I finalize the BERT model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
