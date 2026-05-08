# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load(r"C:\Users\Jagatjyoti\Jagat_Code\08-05-2026\StudentMarkPred\student_mark_predictor_model.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
   
    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features)
   
    #validate input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
       

    output = model.predict([features_value])[0][0].round(2)

    # input and predicted value store in df then save in csv file
    df= pd.concat([df,pd.DataFrame({'Study Hours':input_features,'Predicted Output':[output]})],ignore_index=True)
    print(df)  
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html', prediction_text='You will get [{}%] marks, when you do study [{}] hours per day '.format(output, int(features_value[0])))


if __name__ == "__main__":
    app.run(host='127.0.0.1')

#if __name__ == "__main__":
#    app.run(host='0.0.0.0', port=8080)
















# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer

# df = pd.read_csv(r'C:\Users\Jagatjyoti\Jagat_Code\07-05-2026\StudentMarkPred\student_info.csv')

# df.info()
# # Handle missing values using SimpleImputer



# plt.scatter(x_test,y_test)
# plt.plot(x_train,lr.predict(x_train),color='red')



# import joblib
# joblib.dump(lr, 'student_mark_predictor.pkl')
# # Load the model

# loaded_model = joblib.load('student_mark_predictor.pkl')
# # Example usage
# new_data = np.array([[5, 1, 0, 1, 0, 1, 0, 1, 0, 1]])  # Replace with actual feature values
# predicted_mark = loaded_model.predict(new_data)
# print(f'Predicted Mark: {predicted_mark[0]}')   

