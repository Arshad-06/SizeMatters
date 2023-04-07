from flask import Flask,render_template, request
from werkzeug.utils import secure_filename
from skimage import io
from sklearn.cluster import KMeans
import math
import numpy as np
import os

UPLOAD_FOLDER = os.path.join('static','uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
  return render_template("index.html")

@app.route("/compress", methods = ["GET","POST"])
def kMeansCompress():
    file = request.files['file']
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    image = io.imread(save_path)
    print("Image Read Successful!")
    rows = image.shape[0]
    cols = image.shape[1]
    print(image.shape)
    image = image.reshape(rows*cols, 3)
    print("Image Reshaping Complete!")
    k = 64
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(image)
    print("K-Means Fitting Complete!")
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
    compressed_image = compressed_image.reshape(rows, cols, 3)
    print("Image Compression Complete!")
    io.imsave(os.path.join(app.config['UPLOAD_FOLDER'],'compressed_image.png'), compressed_image)
    og_size = (rows * cols * 24)/(8 * 1024)
    comp_size = ((24 * k) + (rows * cols * math.ceil(math.log2(k))))/(8 * 1024)
    return render_template('results.html', og_size = og_size, comp_size = comp_size)
  
if __name__ == "__main__":
    app.run()