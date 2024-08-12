import pandas as pd

import nltk
nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Function to classify tokens
def classify_tokens(description, keywords):
    tokens = word_tokenize(description)
    classes = []
    for token in tokens:
      allkeywords = keywords.replace("[","").replace("]","").replace("'","").split(",")
      allkeywords = [i.strip() for i in allkeywords]
      if token in allkeywords:
        classes.append("Noun")
      else:
        classes.append("O")
    return list(zip(tokens, classes))

def Data_Mapping(df,var1,var2):
  output_data = []
  # Iterate through the dataset
  for index, row in df.iterrows():
      description = row[var1]
      keywords = row[var2]
      #try:
      classified_tokens = classify_tokens(description, keywords)
      for token, classification in classified_tokens:
          output_data.append({'ID': row['ID'], 'Token': token, 'Class': classification})
      # except:
      #   pass
  # Create output DataFrame
  output_df = pd.DataFrame(output_data)
  return output_df

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

# Function to classify tokens
def classify_tokens_iob(description, bio_tag,keywords):
    keywords = keywords.replace("[","").replace("]","").replace("'","").split(",")
    description_tokens = description.split(" ")
    classes = []
    for index,tag in enumerate(bio_tag):
        if 'NN' in tag:
            classes.append('NOUN')
        else:
            classes.append("O")
    return list(zip(description_tokens, classes))

def Data_Mapping_iob(df,var1,var2,var3):
  output_data = []
  # Iterate through the dataset
  for index, row in df.iterrows():
      description = row[var1]
      bio_tag = row[var2]
      keywords = row[var3]
      # try:
      classified_tokens = classify_tokens_iob(description,bio_tag, keywords)
      for token, classification in classified_tokens:
          output_data.append({'ID': row['ID'], 'Token': token, 'Class': classification})
      # except:
      #   pass
  # Create output DataFrame
  output_df = pd.DataFrame(output_data)
  return output_df