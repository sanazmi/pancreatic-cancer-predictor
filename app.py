# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('cancer_model_again.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # Get data from form
# #         data = request.form

# #         # Extract values in model input order
# #         features = [
# #             float(data['creatinine']),
# #             float(data['plasma_CA19_9']),
# #             float(data['age']),
# #             float(data['sex']),
# #             float(data['LYVE1']),
# #             float(data['REG1B']),
# #             float(data['TFF1'])
# #         ]

# #         # Predict using model
# #         prediction = model.predict([features])[0]

# #         result = "High Risk and Tumor!" if prediction == 1 else "Low Risk and Healthy"
# #         return render_template('index.html', result=result)

# #     except Exception as e:
# #         return render_template('index.html', result=f"Error: {str(e)}")



# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         data = request.form

# #         features = [
# #             float(data['creatinine']),
# #             float(data['plasma_CA19_9']),
# #             float(data['age']),
# #             float(data['sex']),
# #             float(data['LYVE1']),
# #             float(data['REG1B']),
# #             float(data['TFF1'])
# #         ]

# #         prediction = model.predict([features])[0]

# #         if prediction == 1:
# #             result = "High Risk and Tumor!"
# #             result_class = "danger"
# #         else:
# #             result = "Low Risk and Healthy"
# #             result_class = "safe"

# #         return render_template('index.html', result=result, result_class=result_class)

# #     except Exception as e:
# #         return render_template('index.html', result=f"Error: {str(e)}", result_class="danger")


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

#         # Add model accuracy (update this value to match your real test accuracy)
#         model_accuracy = 0.915  # 91.5% for example
#         accuracy_text = f"Model Accuracy: {model_accuracy * 100:.2f}%"

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

app = Flask(__name__)

# Load the trained ensemble model
model = joblib.load('cancer_model_again.pkl')

# Load and prepare the dataset for accuracy evaluation
df = pd.read_csv('ssv_ds.csv')
df['diagnosis'] = df['diagnosis'] == 3
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df = df[['creatinine', 'plasma_CA19_9', 'age', 'sex', 'LYVE1', 'REG1B', 'TFF1', 'diagnosis']]
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

        prediction = model.predict([features])[0]

        if prediction == 1:
            result = "High Risk and Tumor!"
            result_class = "danger"
        else:
            result = "Low Risk and Healthy"
            result_class = "safe"

        # Dynamically compute ensemble model accuracy
        accuracy_val = accuracy_score(y_test, model.predict(X_test))
        accuracy_text = f"{accuracy_val * 100:.2f}%"

        return render_template('index.html', result=result, result_class=result_class, accuracy=accuracy_text)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}", result_class="danger", accuracy=None)

if __name__ == '__main__':
    app.run(debug=True)
