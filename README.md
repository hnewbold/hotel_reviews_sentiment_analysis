# Hotel Review Sentiment Analysis with Large Language Models (LLM): Leveraging BERT in TensorFlow

**Author**: Husani Newbold

**Date**: 2024-08-06

## Table of Contents
1. [Introduction & Project Description](#introduction--project-description)
2. [Dataset](#dataset)
3. [Model Structure](#model-structure)
4. [Training the Model](#training-the-model)
5. [Results](#results)
6. [Improvements and Recommendations](#improvements-and-recommendations)
7. [Contributors](#contributors)

## Introduction & Project Description
This project analyzes the sentiment of hotel reviews using advanced Natural Language Processing (NLP) techniques. The dataset, sourced from Kaggle, contains hotel reviews from TripAdvisor.

The reviews are cleaned using the Natural Language Toolkit (nltk) to remove noise and standardize the text. This includes removing HTML tags, URLs, special characters, and stopwords, as well as lemmatizing the text for consistency.

A pre-trained BERT model, accessed through the Hugging Face Transformers library, is fine-tuned using the cleaned hotel review data. This involves adjusting the model's parameters and training it further on the specific dataset to better capture the nuances of hotel review sentiments.

The result is a fine-tuned BERT model implemented in TensorFlow that can accurately predict the sentiment (positive, neutral, or negative) of hotel reviews. The final model achieves an accuracy of nearly 80%, demonstrating its effectiveness in classifying the sentiment of hotel reviews.

## Dataset
The dataset for this project was sourced from Kaggle and consists of approximately 20,000 hotel reviews from TripAdvisor, with ratings ranging from 1 to 5. To ensure balanced representation across the three sentiment groups (positive, neutral, and negative), the dataset was further filtered down to 6,000 records. This balanced dataset includes an equal number of reviews for each sentiment group, providing a robust foundation for training and evaluating the sentiment analysis model.

<img src="Rating Distributions.png" alt="ANN" width="500" height="300">

## Model Structure
The following code defines the training arguments for fine-tuning the BERT model using the Hugging Face Transformers library. These arguments configure various aspects of the training process to ensure optimal model performance and efficient training:

```python
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch', 
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,  
)

```

## Training the model
The following code defines the Trainer for fine-tuning the BERT model, incorporating early stopping to prevent overfitting

```python
# Define the Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  
)
```

### Training Metrics
Key metrics from the model training process are summarized below:

```
TrainOutput(global_step=1200, training_loss=0.4239166291554769, metrics={'train_runtime': 1896.2322, 'train_samples_per_second': 12.657, 'train_steps_per_second': 0.791, 'total_flos': 5051868335308800.0, 'train_loss': 0.4239166291554769, 'epoch': 4.0})
```

## Results
### Classification Report
The model performs particularly well in predicting positive and negative sentiments, with precision scores of 0.84 and 0.88, respectively. The F1-scores for positive and negative sentiments are also high, at 0.83 and 0.79, indicating a good balance between precision and recall. The neutral sentiment, while slightly lower in precision (0.63), still achieves a recall of 0.76 and an F1-score of 0.69.

```
              
                 precision    recall  f1-score   support

    Negative       0.88      0.72      0.79       391
     Neutral       0.63      0.76      0.69       401
    Positive       0.84      0.82      0.83       408

    accuracy                           0.77      1200
   macro avg       0.79      0.77      0.77      1200
weighted avg       0.78      0.77      0.77      1200
```

### Model Evaluation

Key metrics from the model evaluation process are summarized below:

```
{'eval_loss': 0.5606216192245483,
 'eval_runtime': 36.661,
 'eval_samples_per_second': 32.732,
 'eval_steps_per_second': 2.046,
 'epoch': 3.0}
```
## Improvements and Recommendations

To further enhance the model's performance, consider the following:

- **Increase Training Data**: Incorporate more labeled data to improve the model's ability to generalize.
- **Advanced Preprocessing**: Experiment with more advanced text preprocessing techniques, such as handling negations or using more sophisticated lemmatization.
- **Hyperparameter Tuning**: Perform a more extensive hyperparameter search to optimize the model's settings.
- **Model Ensemble**: Combine predictions from multiple models to potentially improve accuracy.
- **Domain Adaptation**: Fine-tune the model on a domain-specific dataset if available, to better capture the nuances of hotel reviews.


## Contributors
Husani Newbold (Author)



