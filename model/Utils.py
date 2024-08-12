import pandas as pd

import nltk
nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

def Data_Labelling(processed_df):
  # Define mapping dictionary
  label_map = {'Noun': 1, 'O': 0}

  # Map values in 'Labels' column
  processed_df['Class'] = processed_df['Class'].map(label_map)
  processed_df['Predicted_Class'] = processed_df['Predicted_Class'].map(label_map)
  
  return processed_df

def classifier(token,keywords):
  if token in keywords:
    class_name = "Noun"
  else:
    class_name = "O"
  return class_name

def Data_Mapping(df,var1,var2,var3):
  for index, row in df.iloc[:3,:].iterrows():
        description = row[var1]
        keywords = row[var2]
        #print(keywords,'keywords')
        

        bert_keywords = row[var3]
        #print(bert_keywords,'bert_keywords')
        
        tokens = word_tokenize(description)
        classes,bert_classes = [],[]
        for token in tokens:
          allkeywords = keywords.replace("[","").replace("]","").replace("'","").split(",")
          allkeywords = [i.strip() for i in allkeywords]
          classes.append(classifier(token,allkeywords))
          bert_classes.append(classifier(token,bert_keywords))
  # print(classes)

  output_data = []

  classified_tokens = list(zip(tokens, classes,bert_classes))
  #classified_tokens
  for token, classes,bert_classes in classified_tokens:
      output_data.append({'ID': row['ID'], 'Token': token, 'Class': classes,'Predicted_Class':bert_classes})
  processed_df = pd.DataFrame(output_data)
  return processed_df

# Function to perform IOB tagging
def iob_tag(tokens):
    tagged = []
    for i, (token, tag) in enumerate(tokens):
        if i == 0 and tag != 'O':
            tagged.append("B-" + tag)
        elif tokens[i-1][1] == tag:
            tagged.append("I-" + tag)
        else:
            tagged.append("B-" + tag)
    return tagged

def DataAnnotate(sentence):
  # Tokenize the sentence
  tokens = word_tokenize(sentence)

  # Perform part-of-speech tagging
  tagged_tokens = pos_tag(tokens)

  # IOB tagging
  iob_tagged_tokens = iob_tag(tagged_tokens)

  return iob_tagged_tokens

def classifier(token,keywords):
  if token in keywords:
    class_name = "Noun"
  else:
    class_name = "O"
  return class_name

def IOB_classifier(BIO_tags):
  if 'NN' in BIO_tags:
    class_name = "Noun"
  else:
    class_name = "O"
  return class_name

def Data_Mapping_IOB(df,var1,var2,var3):
  for index, row in df.iterrows():
        description = row[var1]
        keywords = row[var2]
        #print(keywords,'keywords')


        bert_keywords = row[var3]
        #print(bert_keywords,'bert_keywords')

        tokens = word_tokenize(description)
        classes,bert_classes = [],[]
        for t,token in enumerate(tokens):
          allkeywords = keywords.replace("[","").replace("]","").replace("'","").split(",")
          allkeywords = [i.strip() for i in allkeywords]
          classes.append(classifier(token,allkeywords))
          bert_classes.append(IOB_classifier(bert_keywords[t]))
  # print(classes)

  output_data = []

  classified_tokens = list(zip(tokens, classes,bert_classes))
  #classified_tokens
  for token, classes,bert_classes in classified_tokens:
      output_data.append({'ID': row['ID'], 'Token': token, 'Class': classes,'Predicted_Class':bert_classes})
  processed_df = pd.DataFrame(output_data)
  return processed_df

def compute_performance_metrics(metrics_df,res_df,model_name):
  # Compute accuracy,precision, recall, and F1 score
  accuracy = accuracy_score(res_df['Class'], res_df['Predicted_Class'])
  precision = precision_score(res_df['Class'], res_df['Predicted_Class'])
  recall = recall_score(res_df['Class'], res_df['Predicted_Class'])
  f1 = f1_score(res_df['Class'], res_df['Predicted_Class'])

  # Print the results
  print("Accuracy:", accuracy*100)
  print("Precision:", precision*100)
  print("Recall:", recall*100)
  print("F1 Score:", f1*100)

  pred_noun_count = res_df[res_df['Predicted_Class']==1].shape[0]
  tokens_noun_count = res_df[res_df['Class']==1].shape[0]
  covered_area = pred_noun_count/tokens_noun_count
  print("the percentage of covered area:", covered_area)
  from collections import defaultdict
  results = defaultdict(list)
  results['Model'].append(model_name)
  results['Accuracy'].append(round(accuracy*100,2))
  results['Precision'].append(round(precision*100,2))
  results['Recall'].append(round(recall*100,2))
  results['F1 Score'].append(round(f1*100,2))
  results['Covered Area'].append(round(covered_area*100,2))

  metrics_df = metrics_df.append(results, ignore_index=True)

  return metrics_df