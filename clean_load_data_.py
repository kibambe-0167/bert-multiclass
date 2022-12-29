def clean_():
    '''use to load and clean data for models.'''
    return "clean function"


# from cleantext import clean # helps to remove imoji in text
import pandas as pd
import re

# 
def clean( text ):
  '''clean tweet texts and remove links, usernamas'''
  text = text.lower()
  text = ' '.join( text.split() )
  text = ' '.join( [ re.sub("^@\w+", " ", t) for t in text.split(' ') ] ) # remove usernames
  # text = ' '.join( [ re.sub("^@\w+", " ", t) for t in text.split(' ') ] ) # remove hashtags
  text = ' '.join( [ re.sub("^http\w+", " ", t) for t in text.split(' ') ] ) # remove links
  print(text)
  text = re.sub("[^a-z0-9]", " ", text) # remove imoji.
  # text = clean(text, no_emoji=True)
  return ' '.join( text.split() )

# make classes
def make_label( class_ ):
  ''' 
  neu   - 0
  pos   - 1
  neg   - 2
  vpos  - 3
  vneg  - 4
  '''
  class_ = class_.lower()
  if class_ == 'vneg': return 4
  elif class_ == 'neu': return 0
  elif class_ == 'neg': return 2
  elif class_ == 'vpos': return 3
  elif class_ == 'pos': return 1


def load_data():
    data = pd.read_csv("./traindata1.1.csv",engine="python", encoding='utf-8')
    data.drop(axis=1, inplace=True, columns=['UserID','Date/Time'] )
    data.drop_duplicates(inplace=True)
    return data