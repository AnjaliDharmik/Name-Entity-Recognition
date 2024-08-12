import pandas as pd

import nltk
nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

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
