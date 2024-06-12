import streamlit as st
import pandas as pd
import plotly.express as px
import re
import string
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from streamlit_option_menu import option_menu

# Function to clean text
def clean_text(text):
    return re.sub('[^a-zA-Z]', ' ', text).lower()

# Function to lemmatize text
def lemmatize_text(token_list):
    lemmatizer = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Dataset.csv")
    return data

def load_datas():
    data = pd.read_csv("tripadvisor_hotel_reviews.csv")
    return data

def loadsmot():
    data = pd.read_csv("rating_distribution_after_smote.csv")
    return data

# Preprocess data
def hitung_label(data):
    jumlah_label_1 = sum(data['label'] == 1)
    jumlah_label_0 = sum(data['label'] == 0)
    
    print(f"Jumlah label 1: {jumlah_label_1}")
    print(f"Jumlah label 0: {jumlah_label_0}")

# Tokenize text
def tokenize_text(text):
    return text.split()

# Count punctuation
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100

def home():
    st.header("Welcome to Sentiment Analysis of Hotel Reviews!")

    image1 = Image.open("logo.png")
    st.image(image1, use_column_width=True)

    st.markdown("""
        <div style="background-color:#f9f9f9;padding:10px;border-radius:10px;text-align:justify;">
        Hotel reviews help us understand customer sentiment towards services and experiences. Positive reviews can be used for marketing strategies, while negative reviews help improve weaknesses. Sentiment analysis supports the evaluation of service strategies and facility improvements.
        </div>
    """, unsafe_allow_html=True)

# Main function to run the app
def EDA():
    st.title("Hotel Review Analysis")
    st.subheader("Dataset")

    st.markdown("""
        <div style="background-color:#f9f9f9;padding:10px;border-radius:10px;text-align:justify;">
        The dataset used in this project is "Hotel Reviews Sentiment" obtained from Kaggle. This dataset contains 20,492 hotel reviews from the TripAdvisor website. Each review includes the review text and a hotel rating on a scale of 1 to 5.        </div>
        """, unsafe_allow_html=True)

    # Load data
    datas = load_datas()
    
    df = load_data()

    # Show data
    if st.checkbox("Show Data"):
        st.write(datas.head())

    # Membuat dropdown selectbox
    selected_option = option_menu(
        menu_title="Visualisasi Data",  # required
        options=["Distribution of Ratings", "Distribution of Review Length", "Word Cloud", "Distribution of Ratings after SMOTE"],  # required
        icons=["bar-chart", "align-left", "cloud", "line-chart"],  # Icons for each option
        menu_icon="cast",  # Main menu icon
        default_index=0,  # Default selected option
    )

    if selected_option == "Distribution of Ratings":
        st.subheader("Distribution of Ratings")
        # Membuat bar chart dengan matplotlib
        fig, ax = plt.subplots()
        # Mengubah warna bar
        ax.bar(df['rating'].value_counts().index, df['rating'].value_counts().values, color='#C0DFD4')
        st.pyplot(fig)
    
    if selected_option == "Distribution of Review Length":
        st.subheader("Distribution of Review Length")
        fig = go.Figure(data=[go.Histogram(x=df['review_len'], nbinsx=50, marker_color='#C0DFD4')])
        fig.update_layout(title_text='Distribution of Review Length', xaxis_title='Review Length', yaxis_title='Frequency')
        st.plotly_chart(fig)

    elif selected_option == "Word Cloud":
        filtered_positive = " ".join(df[df['label'] == 1]['lemmatized_review'])
        filtered_negative = " ".join(df[df['label'] == 0]['lemmatized_review'])

        # Show word cloud for positive reviews
        st.subheader("Positive Reviews Word Cloud")
        wordcloud = WordCloud(max_font_size=160, margin=0, background_color="white", colormap="Greens").generate(filtered_positive)
        plt.figure(figsize=[10, 10])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.title("Positive Reviews Word Cloud")
        st.pyplot(plt)

        # Show word cloud for negative reviews
        st.subheader("Negative Reviews Word Cloud")
        wordcloud = WordCloud(max_font_size=160, margin=0, background_color="white", colormap="Reds").generate(filtered_negative)
        plt.figure(figsize=[10, 10])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.title("Negative Reviews Word Cloud")
        st.pyplot(plt)

    elif selected_option == "Distribution of Ratings after SMOTE":
        smote = loadsmot()
        # Show distribution of ratings
        st.subheader("Distribution of Ratings after SMOTE")
        # Membuat bar chart dengan matplotlib
        fig, ax = plt.subplots()
        # Mengubah warna bar
        ax.bar(smote['label'].value_counts().index, smote['label'].value_counts().values, color='#C0DFD4')
        st.pyplot(fig)

# Second Page for Sentiment Analysis
def sentiment_analysis():
    st.title("Sentiment Analysis Hotel")
    st.write("Please enter a hotel review to predict the sentiment.")

    st.subheader("Predict Sentiment")

    # Load the pre-trained model and TfidfVectorizer
    with open('modelxgb.sav', 'rb') as f:
        modelxgb = pickle.load(f)

    with open('modelet.sav', 'rb') as f:
        modelet = pickle.load(f)

    with open('modellr.sav', 'rb') as f:
        modellr = pickle.load(f)

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    with open('PCA.pkl', 'rb') as f:
        pca = pickle.load(f)

    comment = st.text_input('Enter review text:', value="")
    
    def preprocess_text(text):
        if isinstance(text, str):
            processed_text = clean_text(text)
            review_len = len(text) - text.count(" ")
            punct = count_punct(text)
            tokenized_text = tokenize_text(processed_text)
            lemmatized_review = lemmatize_text(tokenized_text)
            return lemmatized_review, review_len, punct
        return "", 0, 0

    def predict_sentiment(input_text):
        lemmatized_review, review_len, punct = preprocess_text(input_text)
        X_comment = tfidf_vectorizer.transform([lemmatized_review])
        X_comment = pca.transform(X_comment)
        X_numerical = pd.DataFrame({'review_len': [review_len], 'punct': [punct]})
        X_vect = pd.concat([X_numerical, pd.DataFrame(X_comment)], axis=1)
        X_vect.columns = X_vect.columns.astype(str)

        # Perform predictions using all models
        prediction_xgb = modelxgb.predict(X_vect)
        prediction_et = modelet.predict(X_vect)
        prediction_lr = modellr.predict(X_vect)

        # Return predictions of all models
        return prediction_xgb, prediction_et, prediction_lr

    opsi = ["XGBoost", "Logistic Regression"]
    opsi_selection = st.selectbox('pilih opsi', opsi)

    if st.button("Predict", key="predict_button"):
        if comment:
            prediction_xgb, prediction_et, prediction_lr = predict_sentiment(comment)

            st.markdown("""
            <style>
            .secondary {
                background-color: #f0f0f0;
                color: #333;
                padding: 5px 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            if opsi_selection == "XGBoost":
                sentiment_xgb = 'Positive 😁😁😁😁' if prediction_xgb == 1 else 'Negative 😒😒😒😒'
                st.markdown(f'<div class="secondary">Model XGBoost prediction: {sentiment_xgb}</div>', unsafe_allow_html=True)

            # elif opsi_selection == "Extra Trees":
            #     sentiment_et = 'Positive 😁😁😁😁' if prediction_et == 1 else 'Negative 😒😒😒😒'
            #     st.markdown(f'<div class="secondary">Model Extra Trees prediction: {sentiment_et}</div>', unsafe_allow_html=True)

            elif opsi_selection == "Logistic Regression":
                sentiment_lr = 'Positive 😁😁😁😁' if prediction_lr == 1 else 'Negative 😒😒😒😒'
                st.markdown(f'<div class="secondary">Model Logistic Regression prediction: {sentiment_lr}</div>', unsafe_allow_html=True)

# Main Application
with st.sidebar:
    selected = option_menu("Menu", ["Home", "Data Exploration", "Prediction", "About"], 
        icons=['house', 'bar-chart', 'search', 'info'], menu_icon="cast", default_index=0)

if selected == "Home":
    home()
elif selected == "Data Exploration":
    EDA()
elif selected == "Prediction":
    sentiment_analysis()
elif selected == "About":
    st.title("About")
    st.write("This is a sentiment analysis project for hotel reviews.")

    #st.subheader("TripAdvisor")
    imageta = Image.open('TripAdvisor_Logo.png')
    st.image(imageta, use_column_width=True)
    st.markdown("""
        <div style="background-color:#f9f9f9;padding:10px;border-radius:10px;text-align:justify;">
        TripAdvisor is an online platform that provides reviews and recommendations on travel destinations, hotels, restaurants, and attractions from millions of users around the world. Through TripAdvisor, travelers can compare prices, book trips, and read experiences and ratings from other travelers to make more informed decisions.
        </div>
    """, unsafe_allow_html=True)