from flask import Flask, request, jsonify
from model import FurnitureClassifierV1
from data_preprocessing import FurnitureTransform
from torch import load, no_grad
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

m = FurnitureClassifierV1()
m.load_state_dict(load("./model.pk"))
m.eval()
trans = FurnitureTransform(256, crop=False)
im_trans = transforms.ToTensor()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file uploaded."}), 400
    try:
        im = Image.open(request.files['file'].stream)
    except Exception:
        return jsonify({"error": "Invalid file."}), 400
    im.load()
    inp = im_trans(im.convert("RGB"))*255
    inp = trans(inp)
    with no_grad():
        out = m.evaluate(inp)
    return jsonify({"prediction": ["bed", "chair", "sofa"][out]}), 200

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
