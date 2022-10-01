from flask import Flask,render_template,request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def main():
   if request.method == 'POST':
      classifier = joblib.load('classifier.pkl')
      age = request.form.get("Age")
      Pregnancies = request.form.get("Pregnancies")
      Glucose = request.form.get("Glucose")
      BloodPressure = request.form.get("BloodPressure")
      Insulin = request.form.get("Insulin")
      BMI = request.form.get("BMI")
      DiabetesPedigreeFunction = request.form.get("DiabetesPedigreeFunction")
      SkinThickness = request.form.get("SkinThickness")

      X = pd.DataFrame([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,age]])
      scalar = StandardScaler()
      input_data = X
      scalar.fit(input_data)
      standardeised_data = scalar.transform(input_data)

      predcition = classifier.predict(input_data)
      if predcition==1:
         predcition = 'The Patient is Diabetic'
      else:
         predcition = 'The Patient is Non-Diabetic'


   else:
      predcition = ""
   return render_template('index.html',output = predcition)

if __name__ == '__main__':
   app.run(debug=True)
