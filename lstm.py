from keras import Sequential
from keras.layers import Embedding, Bidirectional, Dense, Activation, LSTM

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd


_NUM_WORDS = 80000
_INPUT_LEN = 500
_BATCH_SIZE = 256
_EPOCH = 4

data = pd.DataFrame("./clean_data/shona.csv" )

X_train, X_test, y_train, y_test = train_test_split(
  data['shona'],
  data['target'],
  test_size=.25,
  stratify=data['target'],
  random_state=43
)

X_train, X_val, y_train, y_val = train_test_split(
  X_train,
  y_train,
  test_size=.25,
  stratify=y_train,
  random_state=43
  )

# view data.
print(X_train.head(5))

# reset index of data and drop them.
X_train.reset_index(drop=True,inplace=True)
X_val.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
y_val.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)


# create tokenizer object and set number of words for tokenizer.
# fit train text on class object.
text_tok_obj = Tokenizer(num_words=_NUM_WORDS)
text_tok_obj.fit_on_texts(X_train['shona'].astype(str) )

# tokenize the text .
X_train_tok = pad_sequences( X_train, maxlen=_INPUT_LEN, padding='post' )
X_val_tok = pad_sequences( X_val, maxlen=_INPUT_LEN, padding='post' )
X_test_tok = pad_sequences( X_test, maxlen=_INPUT_LEN, padding='post' )



# make model and add layers to it.
model = Sequential()
model.add(( Embedding(_NUM_WORDS, 128, input_length=_INPUT_LEN)))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, name='FC1'))
model.add(Activation('relu'))
model.add(Dense(1, name='out_layer'))
model.add(Activation('sigmoid'))

# set loss, optimizer and metric to use.
model.compile( loss='crossentropy', optimizer='adam',metrics=['accuracy'] )

model.fit( 
          X_train_tok,
          y_train,
          batch_size=_BATCH_SIZE,
          epochs=_EPOCH,
          validation_data=(X_val_tok, y_val)
          )