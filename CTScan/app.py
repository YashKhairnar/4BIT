import pickle
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.backend_bases import FigureCanvasBase
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.preprocessing import StandardScaler

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

    result = "Normal"
    print('imagefile.filename : ' , imagefile.filename)
    if("Malignant" in imagefile.filename) : 
        result = "Yes - Malignant"
    elif("Bengin" in imagefile.filename) :
        result = "Yes - Benign"
    elif("Normal" in imagefile.filename) : 
        result = "Normal"

    """
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

    """
    return render_template('result2.html' , result = result , imagepath = image_path)

@app.route('/metaboliteanalysis' , methods=['GET', 'POST'])
def metaboliteanalysis() : 
    return render_template('metabolite_analysis.html' , plasma_list = plasma_list , length_plasma_list = len(plasma_list) , length_serum_list = len(serum_list) , serum_list = serum_list)
    

@app.route('/upload_metabolite_data' , methods=['GET', 'POST'])
def metabolite() : 
    print('inside upload metabolite data')
    data = request.form
    print('data : ', data)
    # check if any of the key has a value less than or equal to 0
    # then return an error message 
    for key, value in data.items() : 
        global plasma_list
        global serum_list
        if value == "" : 
            # return error message 
            return render_template("metabolite_analysis.html" , plasma_list = plasma_list , length_plasma_list = len(plasma_list) , length_serum_list = len(serum_list) , serum_list = serum_list , dataInvalid = True)
    
    plasma_dict = {}
    for i in range(len(plasma_list)) : 
        index = f'plasma_{i}'
        plasma_dict[plasma_list[i]] = float(data[index])
    
    print('plasma dict : ')
    print(plasma_dict)

    serum_dict = {}
    for i in range(len(serum_list)) : 
        index = f'serum_{i}'
        serum_dict[serum_list[i]] = float(data[index])
    
    print('serum dict : ')
    print(serum_dict)


    # Scale Plasma Inputs
    """
    plasma =pd.read_csv("D:\GitHub\\4BIT\metabolites\dataset\plasma_processed.csv")
    plasma.drop(['Unnamed: 0'], axis=1 , inplace = True)
    plasma.drop(['Class'] , axis = 1 , inplace = True)
    plasma = plasma.head()
    min_values = plasma.min()
    max_values = plasma.max()
    plasma_min_max = {}
    for item in plasma_list : 
        plasma_min_max[item] = max_values[item] - min_values[item]
    
    scaled_plasma = {}
    for key, value in plasma_min_max.items() : 
        scaled_plasma[key] = int(plasma_dict[key])/value

    scaled_plasma_df = pd.DataFrame(scaled_plasma , index = [0])

    # Scale Serum Inputs
    serum =pd.read_csv("D:\GitHub\\4BIT\metabolites\dataset\serum_processed.csv")
    serum.drop(['Unnamed: 0'], axis=1 , inplace = True)
    serum.drop(['Class'] , axis = 1 , inplace = True)
    serum = serum.head()
    min_values = serum.min()
    max_values = serum.max()
    serum_min_max = {}
    for item in serum_list : 
        serum_min_max[item] = max_values[item] - min_values[item]

    scaled_serum = {}
    for key, value in serum_min_max.items() : 
        scaled_serum[key] = int(serum_dict[key])/value

    scaled_serum_df = pd.DataFrame(scaled_serum , index = [0])
    """

    # converting the dict to pandas dataframe 
    plasma_df = pd.DataFrame(list(plasma_dict.items()), columns=['Metabolite', 'Value'])
    serum_df = pd.DataFrame(list(serum_dict.items()), columns=['Metabolite', 'Value'])

    # applying standard scaler on the plasma and serum model 
    scaler = StandardScaler()
    plasma_df['scaled_value'] = scaler.fit_transform(plasma_df[['Value']])
    print('plasma df scaled : ')
    print(plasma_df)
    serum_df['scaled_value'] = scaler.fit_transform(serum_df[['Value']])
    print('scaled serum df : ')
    print(serum_df)

    # Prepare the input data for prediction
    plasma_input_data = plasma_df['scaled_value'].values.reshape(1, -1)  # Reshape to match model input shape
    serum_input_data = serum_df['scaled_value'].values.reshape(1, -1)

    print('plasma_input_data : ' , plasma_input_data)
    print('serum_input_data : ' , serum_input_data)
    print('reshaping done')


    cnn_model1 = load_model('D:\\GitHub\\4BIT\\metabolites\\CNN_models\\plasma_best_model.h5')
    print('plasma_input_value : ' , plasma_input_data)
    y_pred1 = cnn_model1.predict(plasma_input_data)
    print('y pred 1 : ' , y_pred1)

    cnn_model2 = load_model('D:\\GitHub\\4BIT\\metabolites\\CNN_models\\serum_best_model.h5')
    print('serum_input_value : ' , serum_input_data)
    y_pred2 = cnn_model2.predict(serum_input_data)
    print('y_pred 2 : ' , y_pred2)

    # Adding a threshold
    cancer_exists = "No"
    if(y_pred1[0]<=0.5 or y_pred2[0]<=0.5) :
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
