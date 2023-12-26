import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)


class InputImage:
    _ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    def __init__(self, file) -> None:
        if not self._allowed_file_format(file.filename):
            raise Exception("無効なファイル形式です")
        self.image = Image.open(file)

    def _allowed_file_format(self, filename: str) -> bool:
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self._ALLOWED_EXTENSIONS
        )

    def resize(self, size: tuple) -> Image:
        return self.image.resize(size)

    def convertRGB(self) -> Image:
        return self.image.convert("RGB")


class CNN:
    _IMAGE_SIZE = (128, 128)
    _CLASS_1 = "健康な葉"
    _CLASS_0 = "病気の葉"
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _MODEL_FILE = os.path.join(_BASE_DIR, "model/leaf_disease_model.h5")

    def __init__(self):
        if not os.path.exists(self._MODEL_FILE):
            raise Exception("モデルファイルが存在しません")
        self.model = load_model(self._MODEL_FILE)

    def _prepare_input(self, image: InputImage) -> np.array:
        _image = image.convertRGB().resize(self._IMAGE_SIZE)
        _image = np.array(_image) / 255.0
        _image = np.expand_dims(_image, axis=0)
        return _image

    def predict(self, input_image: InputImage) -> str:
        _input = self._prepare_input(input_image)
        _result = self.model.predict(_input)
        return self._CLASS_1 if _result[0][0] > 0.5 else self._CLASS_0


@app.route("/", methods=["GET", "POST"])
def classify_leaf():
    try:
        if request.method == "POST":
            input_image = InputImage(request.files["file"])
            cnn = CNN()
            result = cnn.predict(input_image)
            return render_template("index.html", result=f"分類結果 : {result}")
        else:
            return render_template("index.html", result=None)
    except Exception as e:
        return render_template("index.html", result=f"エラー : {str(e)}")


if __name__ == "__main__":
    app.run(debug=False)
