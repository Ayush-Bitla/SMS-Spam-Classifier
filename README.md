# SMS Spam Classifier

A machine learning application that uses Naive Bayes classification to detect spam messages in SMS texts. Built with Streamlit, this interactive web application provides a user-friendly interface for spam detection and model performance analysis.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Ayush-Bitla/SMS-Spam-Classifier)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Features

- 🎯 Real-time spam detection for SMS messages
- 📊 Interactive model performance metrics
- 📈 Visualization of classification results
- 🔍 Analysis of key words contributing to classification
- 📱 User-friendly web interface

## Technical Details

### Model Architecture
- **Algorithm**: Naive Bayes (MultinomialNB)
  - A probabilistic classifier based on Bayes' theorem
  - Makes the "naive" assumption of feature independence
  - Particularly effective for text classification tasks
- **Feature Extraction**: TF-IDF Vectorization
- **Training Data**: SMS Spam Collection Dataset
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

### Tech Stack
- Python 3.12
- Streamlit (Web Interface)
- scikit-learn (Machine Learning)
- pandas (Data Processing)
- matplotlib (Visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ayush-Bitla/SMS-Spam-Classifier.git
cd SMS-Spam-Classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
The model files are not included in the repository due to size constraints. When you run the application for the first time, it will automatically:
- Load the SMS Spam Collection dataset
- Preprocess the text data
- Train the Naive Bayes classifier
- Generate the necessary model files

## Usage

1. Start the Streamlit app:
```bash
streamlit run script/streamlit_app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

3. Use the application:
   - Enter a message in the text area
   - Click "Classify Message" to get the prediction
   - View model performance metrics and visualizations

## Project Structure

```
SMS-Spam-Classifier/
├── script/
│   └── streamlit_app.py      # Main application code
├── sms+spam+collection/
│   └── SMSSpamCollection     # Dataset
├── requirements.txt          # Project dependencies
├── README.md                # Project documentation
└── LICENSE                  # MIT License
```

## Dataset

The project uses the SMS Spam Collection Dataset, which contains:
- 5,574 SMS messages
- Labeled as 'ham' (legitimate) or 'spam'
- Used for training and evaluating the classifier

## Model Files

The following model files are generated when you run the application:
- `vectorizer.pkl`: TF-IDF vectorizer for text preprocessing
- `model.pkl`: Trained Naive Bayes classifier

These files are automatically saved in the `script/` directory and are gitignored to keep the repository size manageable.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- scikit-learn documentation
- Streamlit documentation

## Contact

Ayush Bitla - [@Ayush-Bitla](https://github.com/Ayush-Bitla) 