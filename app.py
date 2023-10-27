import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("leaf_disease_model.h5")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# 画像のファイル名が許可された拡張子かどうかをチェック
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def classify_leaf():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            img = Image.open(file).resize((128, 128))
            img = np.array(img)
            img = img / 255.0  # 画像データを正規化
            img = np.expand_dims(img, axis=0)  # バッチの次元を追加
            result = model.predict(img)

            if result[0][0] > 0.5:
                classification = "健康な葉"
            else:
                classification = "病気の葉"
            
            return render_template("index.html", result=classification)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=False)