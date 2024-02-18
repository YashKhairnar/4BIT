import base64
import io
import pickle
import os
import cv2
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.backend_bases import FigureCanvasBase
import numpy as np
import urllib.request
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_file

matplotlib.use('agg')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
#with open('plot.pickle', 'rb') as f:
    #plot = pickle.load(f)

@app.route('/', methods=['GET'])
def base():
    return render_template('base.html')

@app.route('/predictiveanalysis')
def predictiveanalysis():
    return render_template('predictive analysis.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/upload', methods=['GET', 'POST'])
def image():
    dec = {0:'no_cancer', 1:'cancer'}
    imagefile=request.files['myfile']
    image_path="newImages"+imagefile.filename
    imagefile.save(image_path)
    img=cv2.imread(image_path,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    #imagefile.save(img1)
    prediction=model.predict(img1)
    print(prediction)
    plt.title(dec[prediction[0]])
    plt.imshow(img, cmap='gray')
    output1 = io.BytesIO()
    plt.savefig(output1, format='png')
    output1.seek(0)
    plot_data1 = base64.b64encode(output1.getvalue()).decode('utf-8')
    
    
    with open('plot.pickle', 'rb') as f:
        plot_bytes = f.read()
    fig2 = pickle.loads(plot_bytes)
    plt.switch_backend('Agg')
    output2 = io.BytesIO()
    fig2.savefig(output2, format='png')
    output2.seek(0)
    plot_data2 = base64.b64encode(output2.getvalue()).decode('utf-8')
    
    
    return render_template('result.html', plot_data1=plot_data1, plot_data2=plot_data2)


    


if __name__ == '__main__':
    app.run(debug=True)
