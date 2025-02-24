import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ste = PorterStemmer()

# Function for text cleaning
def text_transform(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  li = [i for i in text if i.isalnum()]

  li = [i for i in li if i not in stopwords.words('english') and i not in string.punctuation]

  li = [ste.stem(i) for i in li]

  return " ".join(li)

st.markdown("""
    <style>
    .main-title {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        background-color: #517F5A;
        color: white;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: black;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Detect Your Spam SMS/Email</div>', unsafe_allow_html=True)

st.write("")
st.write("")

txt = st.text_area("", placeholder="📩 Put your text here...")

if txt:
  if st.button('Predict'):
    # preprocessing
    processed_sms = text_transform(txt)

    # vectorization
    transformed_sms = tfidf.transform([processed_sms])

    # prediction
    prediction = model.predict(transformed_sms)[0]

    # display
    st.markdown('## SPAM' if prediction == 1 else '## NOT SPAM')

elif st.button('Predict'):
  st.markdown('### Put your text ⤴')
