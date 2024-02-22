import pickle
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.backend_bases import FigureCanvasBase
from flask import Flask, jsonify, render_template, request, send_file

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import json
import random

matplotlib.use('agg')

app = Flask(__name__)

plasma_list = ['asparagine',
                'benzoic acid',
                'tryptophan',
                'uric acid',
                '5-hydroxynorvaline NIST',
                'alpha-ketoglutarate',
                'citrulline',
                'glutamine',
                'hypoxanthine',
                'malic acid',
                'methionine sulfoxide',
                'nornicotine',
                'octadecanol',
                '3-phosphoglycerate',
                '5-methoxytryptamine',
                'adenosine-5-monophosphate',
                'aspartic acid',
                'lactic acid',
                'maltose',
                'maltotriose',
                'N-methylalanine',
                'phenol',
                'phosphoethanolamine',
                'pyrophosphate',
                'pyruvic acid',
                'taurine']
serum_list = ['cholesterol',
            'lactic acid',
            'N-methylalanine',
            'phenylalanine',
            'aspartic acid',
            'deoxypentitol',
            'glutamic acid',
            'malic acid',
            'phenol',
            'taurine']

#model = pickle.load(open("model.pkl", "rb"))
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
    #dec = {0:'no_cancer', 1:'cancer'}
    #imagefile=request.files['myfile']
    #image_path="newImages"+imagefile.filename
    #imagefile.save(image_path)
    #img=cv2.imread(image_path,0)
    #img1 = cv2.resize(img, (200,200))
    #img1 = img1.reshape(1,-1)/255
    #imagefile.save(img1)
    #prediction=model.predict(img1)
    #print(prediction)
    #plt.title(dec[prediction[0]])
    #plt.imshow(img, cmap='gray')
    #output1 = io.BytesIO()
    #plt.savefig(output1, format='png')
    #output1.seek(0)
    #plot_data1 = base64.b64encode(output1.getvalue()).decode('utf-8')
    
    
    #with open('plot.pickle', 'rb') as f:
    #    plot_bytes = f.read()
    #fig2 = pickle.loads(plot_bytes)
    #plt.switch_backend('Agg')
    #output2 = io.BytesIO()
    #fig2.savefig(output2, format='png')
    #output2.seek(0)
    #plot_data2 = base64.b64encode(output2.getvalue()).decode('utf-8')

    # -----

    model = load_model('D:\GitHub\\4BIT\CTScan\Lung_Model.h5')
    print(model.summary())
    imagefile = request.files['myfile']
    image_path = "static\\uploads\\" + imagefile.filename
    imagefile.save(image_path)
    image = cv2.imread(image_path)
    numpydata = np.asarray(image)
    print('numpydata : ' , numpydata.shape)
    resized_image = cv2.resize(image , (224 , 224))
    numpydata = np.asarray(resized_image.shape)
    print('numpydata after resizing : ' , numpydata)
    input_image = np.expand_dims(resized_image, axis =0)
    numpydata = np.asarray(input_image.shape)
    print('numpydata after input image : ' , numpydata)
    prediction = model.predict(input_image)
    pred = np.argmax(prediction, axis=1) #pick class with highest  probability
    print("pred : " , pred)
    result = ""
    if(pred[0]==0) : 
        result = "Malignant"
    elif(pred[0]==1) : 
        result = "Normal"
    elif(pred[0]==2) : 
        result = "Benign"

    print('Result : ' , result)
    print("prediction : " , prediction)

    return render_template('result2.html' , result = result , imagepath = image_path)

@app.route('/metaboliteanalysis' , methods=['GET', 'POST'])
def metaboliteanalysis() : 
    return render_template('metabolite_analysis.html' , plasma_list = plasma_list , length_plasma_list = len(plasma_list) , length_serum_list = len(serum_list) , serum_list = serum_list)
    

@app.route('/upload_metabolite_data' , methods=['GET', 'POST'])
def metabolite() : 
    print('inside upload metabolite data')
    data = request.form
    print('data : ', data)
    keys = data.keys()
    # check if any of the key has a value less than or equal to 0
    # then return an error message 
    for key, value in data.items() : 
        global plasma_list
        global serum_list
        if value == "" : 
            # return error message 
            return render_template("metabolite_analysis.html" , plasma_list = plasma_list , length_plasma_list = len(plasma_list) , length_serum_list = len(serum_list) , serum_list = serum_list , dataInvalid = True)
    print()
    print()
    print('keys : ' , keys)

    print('iteratinf over dict : ')
    plasma_list = []
    serum_list = []
    flag = 0
    for key, value in data.items() : 
        if "plasma_" in key : 
            plasma_list.append(value)
        elif "serum_" in key : 
            serum_list.append(value)
    

    
    print()
    print()
    plasma_list = [item if item != '' else '0' for item in plasma_list]
    serum_list = [item if item != '' else '0' for item in serum_list]
    print('plasma_list : ' , plasma_list)
    print('serum_list : ' , serum_list)
    
    
    plasma_list = np.asarray(plasma_list , dtype=float)
    serum_list = np.asarray(serum_list , dtype = float)

    # Cancer detection from plasma
    plasma_ridge_model = pickle.load(open("D:\GitHub\\4BIT\metabolites\models\plasma_ridge_model.pkl", "rb"))
    print('model : ' , plasma_ridge_model)
    plasma_list = plasma_list.reshape(1,-1)
    plasma_pred = plasma_ridge_model.predict(plasma_list)
    print()
    print()
    print('plasma_pred : ' , plasma_pred)\
    
    # Cancer detection from Serum
    serum__xgb_classifier = pickle.load(open("D:\GitHub\\4BIT\metabolites\models\serum_xgb_classifier.pkl", "rb"))
    print('xgb model : ' , serum__xgb_classifier)
    serum_list = serum_list.reshape(1,-1)
    serum_pred = serum__xgb_classifier.predict(serum_list)
    print()
    print()
    print('serum pred : ',  serum_pred)

    cancer_exists = "No"
    if(serum_pred[0] or plasma_pred[0] == 1) : 
        # lung cancer exists
        cancer_exists = "Yes"

    return render_template('metabolite_results.html' , cancer_exists = cancer_exists)


@app.route('/fill_sample_data' , methods = ['GET' , 'POST'])
def fillSample() : 
    print('inside fill sample')

    plasma =pd.read_csv("D:\GitHub\\4BIT\metabolites\dataset\plasma_processed.csv")
    plasma.drop(['Unnamed: 0'], axis=1 , inplace = True)
    plasma.drop(['Class'] , axis = 1 , inplace = True)
    plasma = plasma.head()

    randomNumber = random.randint(0,4)
    plasma_sample_data = []
    for i in plasma_list : 
        plasma_sample_data.append(plasma.loc[randomNumber,i])
    
    print('plasma_sample_data : ' , plasma_sample_data)
    print(len(plasma_sample_data))

    serum =pd.read_csv("D:\GitHub\\4BIT\metabolites\dataset\serum_processed.csv")
    serum.drop(['Unnamed: 0'], axis=1 , inplace = True)
    serum.drop(['Class'] , axis = 1 , inplace = True)
    serum = serum.head()

    serum_sample_data = []
    for i in serum_list : 
        serum_sample_data.append(serum.loc[randomNumber,i])
    
    print('serum_sample_data : ' , serum_sample_data)
    print(len(serum_sample_data))

    
    return render_template("metabolite_analysis.html" , plasma_list = plasma_list , length_plasma_list = len(plasma_list) , length_serum_list = len(serum_list) , serum_list = serum_list , plasma_sample_data = plasma_sample_data , serum_sample_data = serum_sample_data)

if __name__ == '__main__':
    app.run(debug=True)
