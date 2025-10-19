# 📺 YouTube Sentiment Analysis

## 📘 Overview
**YouTube Sentiment Analysis** is an end-to-end Machine Learning project that analyzes sentiments (Positive, Neutral, Negative) of YouTube comments using Natural Language Processing (NLP).  
The model is served via a **Flask API**, and a **Chrome Extension** built with HTML, CSS, and JavaScript interacts with this API to display real-time sentiment analysis directly on YouTube videos.  
The project is fully **containerized using Docker** and deployed on **AWS (EC2, S3, ECR, IAM)** for scalability and automation and CI/CD Deployment using **GitHub Actions**.

---

## 🧱 Project Structure
```
YouTube_Sentiment_Analysis/
│
├── .dvc/                      # DVC tracking and pipeline metadata
├── .github/                   # GitHub workflows (CI/CD)
├── data/                      # Raw and processed datasets
├── flask_app/                 # Flask application for API serving
├── models/                    # Serialized models and vectorizers
├── notebooks/                 # Jupyter notebooks for EDA and experiments
├── src/                       # Source scripts for data prep, training, evaluation
├── visuals/                   # Generated graphs and visualizations
├── yt-chrome-plugin/          # Chrome extension (HTML, CSS, JS)
│
├── .env                       # Environment variables (keys, configs)
├── .gitignore                 # Git ignore file
├── Dockerfile                 # Docker build configuration
├── dvc.yaml                   # DVC pipeline definition
├── dvc.lock                   # DVC lock file for reproducibility
├── params.yaml                # Model and preprocessing parameters
├── requirements.txt           # Project dependencies
├── setup.py                   # Package setup
├── README.md                  # Project documentation
│
├── lgbm_model.pkl             # Trained LightGBM sentiment model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer for text transformation
├── experiment_info.json       # Experiment tracking logs
├── errors.log                 # Error logs
└── visuals/                   # Visualization plots (sentiment graphs)
```

---

## ⚙️ Features
- **Sentiment Prediction (Positive / Neutral / Negative)** on YouTube comments  
- **Flask-based REST API** for model inference  
- **Interactive Chrome Extension** built using HTML, CSS, JS  
- **Real-time graphs and visualizations** (comment sentiment distributions)  
- **Pipeline management with DVC** for data and model versioning  
- **Containerized Deployment with Docker**  
- **AWS Integration:**  
  - **S3** for data storage  
  - **ECR** for Docker image repository  
  - **EC2** for hosting Flask API  
  - **IAM** for access and security management
- **CI/CD Pipeline:**
  - GitHub Actions  

---

## 🧠 Machine Learning Workflow
1. **Data Collection:** Fetch comments using YouTube Data API.  
2. **Text Preprocessing:** Tokenization, stopword removal, lemmatization.  
3. **Feature Extraction:** TF-IDF vectorization.  
4. **Model Training:** LightGBM classifier for sentiment prediction.  
5. **Evaluation:** Accuracy, precision, recall, F1-score.  
6. **Versioning:** Data and models tracked using DVC.  
7. **Deployment:** Flask app containerized and deployed on AWS EC2.
8. **CI/CD** CI/CD automation using GitHub Actions   
9. **Integration:** Chrome extension communicates with Flask API endpoint.

---

## 🚀 Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/Sumit-Prasad01/Real-Time-YouTube-Sentiment-Analysis.git
cd YouTube-Sentiment-Analysis
```

### Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Environment Variables
Create a `.env` file with your configurations (API keys, AWS creds, etc.).

### Run Flask API
```bash
python flask_app/app.py
```

### Run with Docker
```bash
docker build -t youtube-sentiment-analysis .
docker run -p 5000:5000 youtube-sentiment-analysis
```

Access API at: [http://localhost:5000/predict](http://localhost:5000/predict)

---

## 🧩 Chrome Extension Setup
1. Open **Chrome → Extensions → Manage Extensions**  
2. Enable **Developer Mode**  
3. Click **Load unpacked** and select the `yt-chrome-plugin` folder  
4. Open any YouTube video — the extension analyzes and displays sentiment results with graphs.

---

## 📊 Visualization & Monitoring
- **Real-time charts** show positive, neutral, and negative sentiment proportions.  
- **Visuals** stored under `/visuals` for offline inspection.  
- DVC and logs track model performance over time.

---

## ☁️ AWS Deployment
- **Dockerized Flask API** pushed to **ECR**  
- **EC2 Instance** pulls image and runs container  
- **S3** used for dataset and model artifacts  
- **IAM Roles & Policies** ensure secure resource access

---

## 🧰 Tech Stack
| Category | Tools / Frameworks |
|-----------|--------------------|
| Language | Python, JavaScript |
| ML | LightGBM, scikit-learn, TF-IDF |
| API | Flask |
| Frontend | HTML, CSS, JavaScript |
| Versioning | Git, DVC |
| Deployment | Docker, AWS EC2/ECR/S3/IAM |
| Visualization | Matplotlib, Plotly |

---

## 📈 Future Improvements
- Integrate **MLflow** for experiment tracking  
- Enhance Chrome Extension UI with **React.js**  
- Support **multilingual sentiment detection**  
---


