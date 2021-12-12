from flask import Flask,render_template,redirect,request,send_from_directory
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

modelo_file = "model.h5"
modelo = load_model(modelo_file)

app = Flask(__name__,template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

def makePredictions(path):
  img = Image.open(path) 
  img_d = img.resize((224,224))
  rgbimg=None
  if len(np.array(img_d).shape)<3:
    rgbimg = Image.new("RGB", img_d.size)
    rgbimg.paste(img_d)
  else:
      rgbimg = img_d
  rgbimg = np.array(rgbimg,dtype=np.float64)
  rgbimg = rgbimg.reshape((1,224,224,3))
  predictions = modelo.predict(rgbimg)
  a = int(np.argmax(predictions))
  if a == 1:
    a = "Pneumonia"
  else:
    a = "Normal"
  return a

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        if 'img' not in request.files:
            return render_template('home.html', filename="amostra.jpg", message="Please upload an file")
        f = request.files['img'] 
        filename = secure_filename(f.filename) 
        if f.filename == '':
            return render_template('home.html', filename="amostra.jpg", message="Coloque seu raio-X")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html', filename="amostra.jpg", message="please upload an image with .png or .jpg/.jpeg extension")
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files) == 1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            files.remove("amostra.jpg")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER']+'/'+file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('home.html', filename=f.filename, message=predictions,show=True)
    return render_template('home.html', filename='amostra.jpg')

if __name__=="__main__":
    app.run(debug=True)