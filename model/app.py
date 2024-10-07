import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# 加载预训练的 VGG16 模型
model = load_model('./models/VGG16_model.keras')

# 图像预处理函数
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 添加批量维度
    return img_array

# 预测函数
def predict_breed(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    dog_breeds = ['Affenpinscher', 'Afghan_hound', 'Airedale_terrier', 'Akita', 'Alaskan_malamute', ...]  # 品种列表
    return dog_breeds[predicted_class[0]]

# 路由：主页
@app.route('/')
def index():
    return render_template('index.html')

# 路由：处理预测请求
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    image = Image.open(file)
    breed = predict_breed(image)
    
    return render_template('result.html', breed=breed)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
