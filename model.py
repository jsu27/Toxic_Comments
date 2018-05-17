import numpy as np
import pandas as pd
from nltk import tokenize
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
np.random.seed(7)

max_features = 20000
maxlen = 100

#load data
trainx = pd.read_csv("data/train.csv")
trainx = trainx.sample(frac=0.1)
train, test = train_test_split(trainx, test_size=0.3)
train = train.sample(frac=1)
COMMENTS = ["I hate you!", "I like math.", "Tricky sentences are really funny.", "Why are you like this? I only wanted to be friends. That doesn't mean you can do this to me. You're a piece of trash! I hope you die!", "I really want to go to sleep."]


#clean data
list_sentences_train = train["comment_text"].fillna("").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("").values
y_test = test[list_classes].values


#tokenize + word vector
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
tokenized_comments = tokenizer.texts_to_sequences(COMMENTS)

X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
comments = sequence.pad_sequences(tokenized_comments, maxlen=maxlen)

#create model
def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

#train model
model = get_model()
batch_size = 32
epochs = 1
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {}%".format(round(scores[1]*100, 2)))

prediction = model.predict(comments)
for i in range(len(COMMENTS)):
    print(COMMENTS[i])
    print(prediction[i])
    for j in range(6):
        if prediction[i][j] >= 0.5:
            print(list_classes[j])
