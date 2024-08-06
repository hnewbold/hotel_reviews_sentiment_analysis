# Hotel Review Sentiment Analysis with Large Language Models (LLM): Leveraging BERT in TensorFlow

**Author**: Husani Newbold

**Date**: 2024-08-06

## Table of Contents
1. [Introduction & Project Description](#introduction--project-description)
2. [Model Structure](#model-structure)
3. [Training the Model](#training-the-model)
4. [Results](#results)
5. [Contributors](#contributors)

## Introduction & Project Description
This project involves analyzing the sentiment of hotel reviews using advanced natural language processing techniques. We leverage the BERT model, a large language model, through the Hugging Face Transformers library, and implement the analysis in TensorFlow. 

## Model Structure
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
```
TrainOutput(global_step=900, training_loss=0.48618685139550105, metrics={'train_runtime': 1485.9333, 'train_samples_per_second': 16.151, 'train_steps_per_second': 1.009, 'total_flos': 3788901251481600.0, 'train_loss': 0.48618685139550105, 'epoch': 3.0})
```

## Results
### Classification Report
```
              
                recision    recall  f1-score   support

    Negative       0.81      0.79      0.80       419
     Neutral       0.64      0.57      0.61       382
    Positive       0.77      0.87      0.82       399

    accuracy                           0.75      1200
   macro avg       0.74      0.75      0.74      1200
weighted avg       0.74      0.75      0.74      1200
```

### Model Evaluation
```
{'eval_loss': 0.5606216192245483,
 'eval_runtime': 36.661,
 'eval_samples_per_second': 32.732,
 'eval_steps_per_second': 2.046,
 'epoch': 3.0}
```

## Contributors
Husani Newbold (Author)



