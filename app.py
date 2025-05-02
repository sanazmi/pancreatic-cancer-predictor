
# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# app = Flask(__name__)

# # Load the trained ensemble model
# model = joblib.load('ensemble_model.pkl')

# # Load and prepare the dataset for accuracy evaluation
# df = pd.read_csv('ssv_ds.csv')
# df['diagnosis'] = df['diagnosis'] == 3
# df['sex'] = df['sex'].map({'M': 1, 'F': 0})
# df = df[['creatinine', 'plasma_CA19_9', 'age', 'sex', 'LYVE1', 'REG1B', 'TFF1', 'diagnosis']]
# X = df.drop('diagnosis', axis=1)
# y = df['diagnosis']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.form

#         features = [
#             float(data['creatinine']),
#             float(data['plasma_CA19_9']),
#             float(data['age']),
#             float(data['sex']),
#             float(data['LYVE1']),
#             float(data['REG1B']),
#             float(data['TFF1'])
#         ]

#         prediction = model.predict([features])[0]

#         if prediction == 1:
#             result = "High Risk and Tumor!"
#             result_class = "danger"
#         else:
#             result = "Low Risk and Healthy"
#             result_class = "safe"

#         # Dynamically compute ensemble model accuracy
#         accuracy_val = accuracy_score(y_test, model.predict(X_test))
#         accuracy_text = f"{accuracy_val * 100:.2f}%"

#         return render_template('index.html', result=result, result_class=result_class, accuracy=accuracy_text)

#     except Exception as e:
#         return render_template('index.html', result=f"Error: {str(e)}", result_class="danger", accuracy=None)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

app = Flask(__name__)

# Load individual models
model_rf = joblib.load('rf.pkl')
model_cat = joblib.load('cat.pkl')
model_xgb = XGBClassifier()
model_xgb.load_model('xgb_model.json')  # Load in native format

# Rebuild ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', model_rf),
        ('xgb', model_xgb),
        ('cat', model_cat)
    ],
    voting='soft'
)
# Fit ensemble with dummy data to enable predict_proba
# (Render requires `.fit()` before `predict()` even if models are already trained)
df = pd.read_csv('ssv_ds.csv')
df['diagnosis'] = df['diagnosis'] == 3
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df = df[['creatinine', 'plasma_CA19_9', 'age', 'sex', 'LYVE1', 'REG1B', 'TFF1', 'diagnosis']]
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

ensemble_model.fit(X_train, y_train)  # Refit to sync model internals (needed for predict_proba)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        features = [
            float(data['creatinine']),
            float(data['plasma_CA19_9']),
            float(data['age']),
            float(data['sex']),
            float(data['LYVE1']),
            float(data['REG1B']),
            float(data['TFF1'])
        ]

        prediction = ensemble_model.predict([features])[0]

        if prediction == 1:
            result = "High Risk and Tumor!"
            result_class = "danger"
        else:
            result = "Low Risk and Healthy"
            result_class = "safe"

        # Dynamically compute ensemble model accuracy
        accuracy_val = accuracy_score(y_test, ensemble_model.predict(X_test))
        accuracy_text = f"{accuracy_val * 100:.2f}%"

        return render_template('index.html', result=result, result_class=result_class, accuracy=accuracy_text)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}", result_class="danger", accuracy=None)

if __name__ == '__main__':
    app.run(debug=True)
