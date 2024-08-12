# Name-Entity-Recognition
Extracted named entities from the textual data, the surveillance research team is interested in leveraging NER model for extracting subjects and objects from the provided custom dataset. 

## 1. Dataset
The data is built from the MSCOCO2017 dataset, which initially is the image dataset with image captions. However, for the purpose of this project, only captions containing ‘a person’ in the sentence were extracted in order to have a dataset with normal, human activities in the form of the textual data. As a result, there’s a .json file, which is has [IDs, Number, Description, Keywords]. Keywords are the named entities that we want to extract from the provided data:
 

## 2. Models
The team is interested in leveraging LLMs or Transformers, such as BERT or Phi-2, for this task. The initial plan is to create a baseline model so that it only extracts NOUNS from the list. For this it could be Phi-2 (a smaller LLM). Later on I would like to have another models based on BERT or ROBERTa and other related models. The goal is to compare two different models in terms of time efficiency and performance. There should be a training and validation part with fine-tuning. Of course, it’s expected to include evaluation metrics such as accuracy, precision, recall and F1. Further, if possible, the research team wants to explore other, personalized evaluation metrics such as the percentage of covered area. 

For example, given a PERSON entity in the given sentence:
‘A man rides a skateboard’, the expected extractions are: man [3-5) and skateboard [15-24). So, if the model would extract skateboard only, then the covered percentage of correctly extracted words from the sentence is 10/13 (circa 77%).

Data annotation with the BIO2/ BIO style (B-PER, B-LOC, O, etc.) with Phi-2 and BERT models.

## Result:
1.	‘Simple’ model trained on Phi-2
2.	Simple, but bigger model trained on BERT
3.	‘Simple 2.0’ model trained on annotated data on Phi-2
4.	Bigger model trained on annotated data on BERT

## Time duration: 
3 weeks

## dataset: 
https://drive.google.com/drive/u/2/folders/1TpZsaMF6zPJ8rdjCj-Iq_PyE2ThhXoVV 

