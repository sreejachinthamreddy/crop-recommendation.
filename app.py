import numpy as np
from flask import *
import pickle


app = Flask(__name__,template_folder='templates')
model = pickle.load(open('traintest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('one.html')
@app.route('/croprecommend',methods=['GET','POST'])
def recommend():
    return render_template('two.html')
@app.route('/output',methods=['GET','POST'])
def output():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model.predict(final_features) 
    return render_template('result.html', prediction = pred)

if __name__ == "__main__":
    app.run(debug=True)
