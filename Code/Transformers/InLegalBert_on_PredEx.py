import os
import random
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
# from google.colab import drive
import textwrap
import tqdm as tqdm
import progressbar
import keras
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import json


# In[2]:


train_set = pd.read_csv('train.csv')

test_set=pd.read_csv("test.csv")



# In[3]:


len(train_set),len(test_set)
# len(validation_set)


# In[4]:


print(train_set.head())



print(test_set.head())



# In[10]:

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_CLASSES = {
    'mbert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'InLegalBERT': (BertForSequenceClassification, AutoTokenizer, AutoConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)}

model_type = 'InLegalBERT' ###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
model_name = 'law-ai/InLegalBERT' 


# In[11]:


def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks


# In[12]:


def grouped_input_ids(all_toks):
  splitted_toks = []
  l=0
  r=510
  while(l<len(all_toks)):
    splitted_toks.append(all_toks[l:min(r,len(all_toks))])
    l+=410
    r+=410

  CLS = tokenizer.cls_token
  SEP = tokenizer.sep_token
  e_sents = []
  for l_t in splitted_toks:
    l_t = [CLS] + l_t + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(l_t)
    e_sents.append(encoded_sent)

  e_sents = pad_sequences(e_sents, maxlen=512, value=0, dtype="long", padding="post")
  att_masks = att_masking(e_sents)
  return e_sents, att_masks


# In[13]:


def generate_np_files_for_training(dataf, tokenizer):
  all_input_ids, all_att_masks, all_labels = [], [], []
  for i in progressbar.progressbar(range(len(dataf['Input']))):
    text = dataf['Input'].iloc[i]
    toks = tokenizer.tokenize(text)
    if(len(toks) > 10000):
      toks = toks[len(toks)-10000:]

    splitted_input_ids, splitted_att_masks = grouped_input_ids(toks)
    doc_label = dataf['Label'].iloc[i]
    for i in range(len(splitted_input_ids)):
      all_input_ids.append(splitted_input_ids[i])
      all_att_masks.append(splitted_att_masks[i])
      all_labels.append(doc_label)

  return all_input_ids, all_att_masks, all_labels


# In[14]:


# !pip install --upgrade pytest


# In[15]:


# !pip install transformers==4.29.0


# In[16]:


from transformers import *
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased', do_lower_case=True)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
print("--------Tokenizer done-----------")


train_input_ids, train_att_masks, train_labels = generate_np_files_for_training(train_set, tokenizer)


# In[16]:


def input_id_maker(dataf, tokenizer):
  input_ids = []
  lengths = []

  for i in progressbar.progressbar(range(len(dataf['Input']))):
    sen = dataf['Input'].iloc[i]
    sen = tokenizer.tokenize(sen)#, add_prefix_space=True)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    if(len(sen) > 510):
      sen = sen[len(sen)-510:]

    sen = [CLS] + sen + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=512, value=0, dtype="long", truncating="post", padding="post")
  return input_ids, lengths


# In[17]:


# validation_input_ids, validation_lengths = input_id_maker(validation_set, tokenizer)


# In[18]:


# (validation_input_ids[0])


# In[19]:


# token_len = []
# for i in progressbar.progressbar(range(len(train_set['Input']))):
#     text = train_set['Input'].iloc[i]
# #    print(train_set.iloc[i], text)
#     token_len.append(len(tokenizer.tokenize(text)))
#    break


# In[20]:


# (token_len)


# In[21]:


# sum_ = sum(token_len)
# print(sum_)

# length = len(token_len)
# print(length)

# average = sum_/length

# print(average)


# In[22]:


# validation_attention_masks = att_masking(validation_input_ids)
# validation_labels = validation_set['label'].to_numpy().astype('int')


# In[23]:


train_inputs = train_input_ids
# validation_inputs = validation_input_ids
train_masks = train_att_masks
# validation_masks = validation_attention_masks

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
# validation_inputs = torch.tensor(validation_inputs)
# validation_labels = torch.tensor(validation_labels)
# validation_masks = torch.tensor(validation_masks)


# In[24]:


batch_size = 16
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
# validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
# validation_sampler = RandomSampler(validation_data)
# validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)


# In[25]:



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = XLNetForSequenceClassification.from_pretrained("xlnet-large-cased", num_labels=2)
model = BertForSequenceClassification.from_pretrained("law-ai/InLegalBERT", num_labels=2)

model.to(device)


# In[26]:


lr = 2e-6
max_grad_norm = 1.0
epochs = 5
num_total_steps = len(train_dataloader)*epochs
num_warmup_steps = 1000
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 21


np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[27]:


loss_values = []
train_loss_values = []
train_accuracy = []
val_loss_values = []
val_accuracy = []

# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
#             print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))
            print('  Batch {:>5,}  of  {:>5,}. : loss: {:} '.format(step, len(train_dataloader), total_loss/step))


        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    print("")
    print("Running Validation...")

    t0 = time.time()

#     model.eval()

#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0

#     for batch in validation_dataloader:
#         batch = tuple(t.to(device) for t in batch)
#         b_input_ids, b_input_mask, b_labels = batch
        
#         with torch.no_grad():        
#           outputs = model(b_input_ids, attention_mask=b_input_mask)
    
#         logits = outputs[0]

#         logits = logits.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
        
#         tmp_eval_accuracy = flat_accuracy(logits, label_ids)
#         eval_accuracy += tmp_eval_accuracy

#         nb_eval_steps += 1

#     # Report the final accuracy for this validation run.
#     print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

print("")
print("Training complete!")


# In[28]:


##TESTING


# In[29]:


out_path = "saved_models"
output_dir = out_path + '/InLegalBert_15k/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))

# In[30]:



# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# model = BertForSequenceClassification.from_pretrained(output_dir, output_hidden_states=True)

# tokenizer = tokenizer_class.from_pretrained(output_dir)
# model.to(device)


# In[31]:


# for i in range(len(test_set["Label"])):
#     if test_set["Label"].iloc[i] == "Rejected":
#         test_set["Label"].iloc[i] = 0
#     else:
#         test_set["Label"].iloc[i] = 1


# In[32]:


# test_set


# In[33]:


labels = test_set.Label.to_numpy().astype(int)

input_ids, input_lengths = input_id_maker(test_set, tokenizer)
attention_masks = att_masking(input_ids)

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

# Set the size.  
batch_size = 16

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


# In[34]:


import os
import random
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
# from google.colab import drive
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import textwrap
import progressbar
import keras
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random


# In[35]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[36]:


# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []


# Predict 
for (step, batch) in enumerate(prediction_dataloader):
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  

  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')


# In[37]:


# predictions = np.concatenate(predictions, axis=0)
# true_labels = np.concatenate(true_labels, axis=0)
# pred_flat = np.argmax(predictions, axis=1).flatten()
# labels_flat = true_labels.flatten()

# # flat_accuracy(predictions,true_labels)


# In[38]:


def metrics_calculator(preds, test_labels):
    cm = confusion_matrix(test_labels, preds)
    TP = []
    FP = []
    FN = []
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[i][j]

        FN.append(summ)
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[j][i]

        FP.append(summ)
    for i in range(0,2):
        TP.append(cm[i][i])
    precision = []
    recall = []
    for i in range(0,2):
        precision.append(TP[i]/(TP[i] + FP[i]))
        recall.append(TP[i]/(TP[i] + FN[i]))

    macro_precision = sum(precision)/2
    macro_recall = sum(recall)/2
    micro_precision = sum(TP)/(sum(TP) + sum(FP))
    micro_recall = sum(TP)/(sum(TP) + sum(FN))
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision + micro_recall)
    macro_f1 = (2*macro_precision*macro_recall)/(macro_precision + macro_recall)
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

# macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(pred_flat, labels_flat)
# print(macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)


# In[39]:


# # Set the batch size.  
# batch_size = 16

# # Create the DataLoader.
# prediction_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
# prediction_sampler = SequentialSampler(validation_data)
# prediction_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)


# In[40]:


# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []


# Predict 
for (step, batch) in enumerate(prediction_dataloader):
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  

  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')


# In[41]:



predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
pred_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_labels.flatten()

flat_accuracy(predictions,true_labels)


# In[42]:


macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(pred_flat, labels_flat)
print(macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)


# In[43]:


macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(pred_flat, labels_flat)
print("macro_precision", "\t", "macro_recall", "\t\t", "macro_f1", "\t\t", "accuracy")
print(macro_precision, "\t", macro_recall, "\t", macro_f1, "\t", flat_accuracy(predictions,true_labels))





