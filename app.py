from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from ml_complete_proj.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            data = [float(x) for x in request.form.values()]

            new_data = np.array(data).reshape(1,14)
            print(new_data.shape)
            obj = PredictionPipeline()
            predict = obj.predict(new_data)

            # return render_template('results.html', prediction = str(predict))
            return render_template('results.html',prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')
    


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)