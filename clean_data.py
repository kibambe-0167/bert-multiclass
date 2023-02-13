import pandas as pd
import re


def targetNum(x ):
    x = str(x).lower()
    if x == 'neg': return -1
    elif x == 'pos': return 1
    elif x == 'neu' : return 0
    
def clean_( text):
    '''to string. to lower case. remove newline. remove #tags. remove usernames, remove links, remove special chars '''
    text = str(text)
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = ' '.join(text.split())
    text = ' '.join( [ re.sub(r"^#\S+", " ", t) for t in text.split() ] )
    text = ' '.join( [ re.sub(r"^@\S+", " ", t) for t in text.split() ] )
    text = ' '.join( [ re.sub(r"^http\S+", " ", t) for t in text.split() ] )
    text = ' '.join( [ re.sub(r"[^\w\s]", " ", t) for t in text.split() ] )
    return ' '.join( text.split())


path = "./data/traindata1.1.csv"
data = pd.read_csv(path, encoding='cp437',sep=';' )
df = pd.DataFrame()
df[['shona','target']] = data[['SN(Original Shona Tweet)', 'finalLabel3Classes']]
df['target'] = df['target'].apply( targetNum )
df['shona'] = df['shona'].apply( clean_ )

df.to_csv("./clean_data/shona.csv", index=False, encoding='utf-8' )