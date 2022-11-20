import torch
import string
import pandas as pd

# Currently set up for CPU
PATH = "C:\\Users\\PC\\code\\githubProjects\\sentimentAnalysisTwitter\\"

def findWhiteSpace(text):
  spaces_array = []
  for c in range(0, len(text)):
    if text[c] == ' ':
      spaces_array.append(c)
  return spaces_array

def findWhiteSpace(text):
  spaces_array = []
  for c in range(0, len(text)):
    if text[c] == ' ':
      spaces_array.append(c)
  return spaces_array

def cleanData(tweet):
  # remove @ and replace with 'person'
  if "@" in tweet:
    start = tweet.index("@")
    spaces = findWhiteSpace(tweet)
    last = len(tweet)
    for i in spaces:
      if i > start:
        last = i
        break
    person = tweet[start:last]
    betterTweet = tweet.replace(person, "person")

  # remove the change the link to 'link'
  elif "http" in tweet:
    start = tweet.index("http")
    spaces = findWhiteSpace(tweet)
    last = len(tweet)
    for i2 in spaces:
      if i2 > start:
        last = i2
        break
    link = tweet[start:last]
    betterTweet = tweet.replace(link, "link")
  else:
    betterTweet = tweet

  if "@" in betterTweet or "https" in betterTweet:
    betterTweet = cleanData(betterTweet)

  return betterTweet

# clean up data
ids =[]
sentiments=[]
lines =[]

with open(r'Training.txt',encoding='utf8') as f:
  for line in f:
    result = line.split("\t")
    if(len(result)==3):
      ids.append(result[0])
      if result[1] == "+":
        sentiments.append(1)
      else:
        sentiments.append(0)
      lines.append(cleanData(result[2]))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

trainData = pd.DataFrame(list(zip(lines[1:50000], sentiments[1:50000])), columns=['text', 'labels'])
trainData.to_csv(PATH + 'trainingData2.csv', index=False)

testData = pd.DataFrame(list(zip(lines[100000:500000], sentiments[100000:500000])), columns=['text', 'labels'])
testData.to_csv(PATH + 'testingData2.csv', index=False)

from datasets import load_dataset
trainDataset = load_dataset("csv", data_files=PATH + "trainingData2.csv")
testDataset = load_dataset("csv", data_files=PATH + "testingData2.csv")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets_train = trainDataset.map(tokenize_function, batched=True)
tokenized_datasets_test = testDataset.map(tokenize_function, batched=True)

print(tokenized_datasets_train['train'])
print(tokenized_datasets_test['train'][1])

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

import numpy as np
from datasets import load_metric

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

from huggingface_hub import notebook_login
notebook_login()

# Define a new Trainer with all the objects we constructed so far
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='C:\\Users\\PC\\code\\githubProjects\\sentimentAnalysisTwitter\\testing\\',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
#    push_to_hub=True,
#    hub_token=TOKEN
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_train['train'],
    eval_dataset=tokenized_datasets_test['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print('TRAINING COMPLETE')
trainer.evaluate()
print('EVAL COMPLETE')