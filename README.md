Pancreatic Cancer Predictor — Ensemble ML Web App
This project is a Flask-based web application that predicts whether a patient is at high or low risk of pancreatic cancer based on urine and plasma biomarkers, 
using an ensemble of three powerful machine learning models: Random Forest, XGBoost, and CatBoost.


🚀 Live Demo
link -> https://pancreatic-cancer-predictor.onrender.com/

🔍 How It Works
Input: Patient details including biomarkers like LYVE1, REG1B, TFF1, CA19-9, age, sex, and creatinine

Models:

    RandomForestClassifier

    XGBClassifier (XGBoost v2.x)

    CatBoostClassifier

    
**Voting Ensemble**: Combines all three with soft voting (averages probabilities)
Output: Real-time prediction (High Risk / Low Risk) and model accuracy on test data


📁 Project Structure

    app.py                # Flask backend
    rf.pkl                # Trained Random Forest model
    cat.pkl               # Trained CatBoost model
    xgb_model.json        # Trained XGBoost model (native format
    ssv_ds.csv            # Preprocessed biomarker dataset
    requirements.txt      # Python dependencies
    templates/
    └── index.html        # Frontend HTML form
    static/
    └── css.css, js.js    # Styling and interactivity
    

📦 Tech Stack

    Python 3.9+

    Flask (Web Framework)

    scikit-learn (ML API + Ensemble)

    XGBoost (Model v2.x using native JSON save)

    CatBoost

    Render (Hosting + deployment)

    HTML/CSS/JS (Frontend)




👨‍⚕️ Inspiration

Built as part of a senior project to explore early detection tools for pancreatic cancer using non-invasive biomarkers. 
Based on research from:https://github.com/adithyanraj03/Pancreatic_Cancer_Detection_Model
