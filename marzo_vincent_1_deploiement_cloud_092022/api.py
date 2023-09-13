# API depoyed on Azure

import os
from tempfile import TemporaryFile

import torch
from torch import nn
from transformers import DistilBertTokenizer
from transformers import DistilBertModel

from fastapi import FastAPI

from google.cloud import storage
from google.oauth2 import service_account




device = 'cpu'
class_names = ['negative', 'positive']
model_name = 'distilbert-base-uncased'
max_nb_tokens = 230



# get model tokenizer
print()
tokenizer = DistilBertTokenizer.from_pretrained(model_name)


# create model class
class Classifier(nn.Module):

    def __init__(self, nb_classes, max_nb_tokens):
        super(Classifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.flatten = nn.Flatten()
        fc_input_size = self.distilbert.config.hidden_size * max_nb_tokens
        self.fully_connected = nn.Linear(fc_input_size, nb_classes)
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.distilbert(input_ids=input_ids, 
                                    attention_mask=attention_mask
                                    ).last_hidden_state
        bert_output_flattened = self.flatten(bert_output)
        classification_layer = self.fully_connected(bert_output_flattened)

        return classification_layer

nb_classes = len(class_names)
model = Classifier(nb_classes=nb_classes, max_nb_tokens=max_nb_tokens)
print()


#get model from Cloud Storage
print('Accessing model in cloud...')
bucket_name = 'bucket_distilbert_1'
model_bucket = 'distilbert-base-uncased_weights_3.pt'
json_key_path = 'cle_des_champs.json'

credentials = service_account.Credentials.from_service_account_file(json_key_path)
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model_bucket)

print('Loading model...')
with TemporaryFile() as temp_file:
    blob.download_to_file(temp_file)
    temp_file.seek(0)
    model.load_state_dict(torch.load(temp_file, map_location=torch.device(device)))


# api interface
prediction_api = FastAPI()

@prediction_api.get('/')
def predict_sentiment(tweet: str):

    encoded_tweet = tokenizer.encode_plus(tweet, 
                                        max_length=max_nb_tokens, 
                                        add_special_tokens=True, 
                                        padding='max_length', 
                                        return_attention_mask=True, 
                                        return_token_type_ids=False, 
                                        return_tensors='pt'
                                        )

    input_ids = encoded_tweet['input_ids'].to(device)
    attention_mask = encoded_tweet['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, pred = torch.max(output, dim=1)
    proba = nn.functional.softmax(output, dim=1)

    sentiment = class_names[pred]
    proba_sentiment = round(float(proba[0][pred]), 3)

    res = {
        'tweet': tweet, 
        'sentiment': sentiment, 
        'probability': proba_sentiment, 
        }
    return res
