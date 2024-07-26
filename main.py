import streamlit as st
import pickle
import re
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from mlxtend.classifier import StackingCVClassifier 
# from sklearn.multiclass import OneVsRestClassifier
# from mlxtend.classifier import StackingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer


model = pickle.load(open('SGD_SVC_LR_DT_RF.pkl','rb'))
tfidf = pickle.load(open('tfidf.pickle','rb'))

def clean_text(text):
    text = re.sub(r'\:(.*?)\:','',text)
    text = str(text).lower()    
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)

    text = text.split()
    text = ' '.join(text)
    return text

labels = ['Accounting and Finance',
 'Antitrust',
 'Banking',
 'Broker Dealer',
 'Commodities Trading',
 'Compliance Management',
 'Consumer protection',
 'Contract Provisions',
 'Corporate Communications',
 'Corporate Governance',
 'Definitions',
 'Delivery',
 'Examinations',
 'Exemptions',
 'Fees and Charges',
 'Financial Accounting',
 'Financial Crime',
 'Forms',
 'Fraud',
 'IT Risk',
 'Information Filing',
 'Insurance',
 'Legal',
 'Legal Proceedings',
 'Licensing',
 'Licensure and certification',
 'Liquidity Risk',
 'Listing',
 'Market Abuse',
 'Market Risk',
 'Monetary and Economic Policy',
 'Money Services',
 'Money-Laundering and Terrorist Financing',
 'Natural Disasters',
 'Payments and Settlements',
 'Powers and Duties',
 'Quotation',
 'Records Maintenance',
 'Regulatory Actions',
 'Regulatory Reporting',
 'Required Disclosures',
 'Research',
 'Risk Management',
 'Securities Clearing',
 'Securities Issuing',
 'Securities Management',
 'Securities Sales',
 'Securities Settlement',
 'Trade Pricing',
 'Trade Settlement']



st.title('Multi-Class Document Classification')
st.write('This is a multi-class document classification web app to predict the categories of a given text.')

text = st.text_area('Enter the document text')

# Predict
if st.button('Predict'):
    text = clean_text(text)
    text = tfidf.transform([text])
    pred = model.predict(text)
    predicted_labels = []
    for i in range(len(pred[0])):
        if pred[0][i] == 1:
            predicted_labels.append(labels[i])

    if len(predicted_labels) == 0:
        predicted_labels = ['Label Not Present']

    st.write('Model Prediction: ', ", ".join(predicted_labels))
    

    
