import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set page config
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        height: 150px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .spam {
        background-color: #ff4b4b;
        color: white;
    }
    .ham {
        background-color: #00cc00;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    dataset_path = os.path.join('sms+spam+collection', 'SMSSpamCollection')
    df = pd.read_csv(dataset_path, sep='\t', names=['label', 'text'])
    return df

@st.cache_resource
def train_model(df):
    # Prepare data
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_num'], test_size=0.2, random_state=42
    )
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Get evaluation metrics
    y_pred = model.predict(X_test_vec)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, vectorizer, conf_matrix, report, X_test, y_test

def main():
    st.title("📱 SMS Spam Classifier")
    st.write("A machine learning application to detect spam messages using Naive Bayes classification.")
    
    # Load and train
    with st.spinner("Loading dataset and training model..."):
        df = load_data()
        model, vectorizer, conf_matrix, report, X_test, y_test = train_model(df)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Spam Detection", "Model Performance", "Dataset Info"])
    
    # Tab 1: Spam Detection
    with tab1:
        st.header("Message Classification")
        message = st.text_area("Enter your message:", placeholder="Type or paste your message here...")
        
        if st.button("Classify Message"):
            if not message.strip():
                st.warning("Please enter a message.")
            else:
                # Vectorize and predict
                X_input = vectorizer.transform([message])
                prediction = model.predict(X_input)[0]
                probability = model.predict_proba(X_input)[0]
                
                # Show prediction
                if prediction == 1:
                    st.error(f"🚫 This message is likely SPAM (Confidence: {probability[1]:.2%})")
                    if probability[1] > 0.90:
                        st.warning("⚠️ High confidence spam detection! Be very cautious!")
                else:
                    st.success(f"✅ This message is likely HAM (not spam) (Confidence: {probability[0]:.2%})")
                
                # Show feature importance
                st.subheader("Message Analysis")
                words = vectorizer.get_feature_names_out()
                word_importance = pd.DataFrame({
                    'word': words,
                    'importance': X_input.toarray()[0]
                })
                word_importance = word_importance[word_importance['importance'] > 0].sort_values('importance', ascending=False)
                
                if not word_importance.empty:
                    st.write("Key words found in your message:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(word_importance.head(10)['word'], word_importance.head(10)['importance'])
                    plt.xticks(rotation=45)
                    plt.title('Top contributing words to classification')
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance Metrics")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{report['accuracy']:.2%}")
        col2.metric("Spam Precision", f"{report['1']['precision']:.2%}")
        col3.metric("Spam Recall", f"{report['1']['recall']:.2%}")
        col4.metric("Spam F1-Score", f"{report['1']['f1-score']:.2%}")
        
        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf_matrix, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Ham', 'Spam'])
        ax.set_yticklabels(['Ham', 'Spam'])
        plt.colorbar(im)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
        # Show some example predictions
        st.subheader("Example Predictions")
        sample_size = min(5, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        sample_texts = X_test.iloc[sample_indices]
        sample_predictions = model.predict(vectorizer.transform(sample_texts))
        sample_probas = model.predict_proba(vectorizer.transform(sample_texts))
        
        for i, (text, pred, proba) in enumerate(zip(sample_texts, sample_predictions, sample_probas)):
            with st.expander(f"Example {i+1}"):
                st.write("Message:", text)
                st.write("Prediction:", "SPAM" if pred == 1 else "HAM")
                st.write(f"Confidence: {max(proba):.2%}")
    
    # Tab 3: Dataset Info
    with tab3:
        st.header("Dataset Information")
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        total_messages = len(df)
        spam_messages = len(df[df['label'] == 'spam'])
        ham_messages = len(df[df['label'] == 'ham'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", total_messages)
        col2.metric("Spam Messages", spam_messages)
        col3.metric("Ham Messages", ham_messages)
        
        # Plot class distribution
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie([ham_messages, spam_messages], labels=['Ham', 'Spam'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title('Distribution of Ham vs Spam Messages')
        st.pyplot(fig)
        
        # Show sample messages
        st.subheader("Sample Messages")
        st.dataframe(df.sample(5))

if __name__ == "__main__":
    main() 