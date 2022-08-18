import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import scipy
import torch
import matplotlib.pyplot as plt
#import tk
import torch


@st.cache(allow_output_mutation=True)
def get_model():
    roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    return tokenizer, model

tokenizer, model = get_model()


user_input = st.text_area('Enter Tweet to Analyze')
button = st.button("Analyze")

d = {

    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
labels=['Negative','Neutral','Positive']

def get_output(a):
    #m = softmax(a) #
    for i in range(len(a)): #a
        l = labels[i]
        s = scores1[i]
        st.write(l,'sentiment probability: ',s)

def plot_analysis(b):
    #verticle plot
    #plt.barh(labels, b, width=0.2, color=['skyblue'])
    #plt.xlabel('Tweet Sentiment')
    #plt.ylabel("Probability")
    #plt.title('Tweet sentiment Analysis')

    #horizontal plot
    plt.barh(labels, b, 0.3, color=['skyblue'])  # "cyan",skyblue 'purple', 'tomato'])
    plt.xlabel("Probability")
    plt.ylabel('Tweet Sentiment')
    plt.title('Tweet sentiment Analysis')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()




if user_input and button:
    tweet= preprocess(user_input)
    test_sample = tokenizer(tweet, padding=True, truncation=True, max_length=512, return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ", output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
    scores1 = output[0][0].detach().numpy()
    #u=get_output(scores1)
    scores1 = softmax(scores1)
    w=get_output(scores1)  ##
    st.write("Prediction: ",  d[y_pred[0]])
    x1=[scores1[0],scores1[1],scores1[2]]
    x2=plot_analysis(x1)

    #build a function for barplot
