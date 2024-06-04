import onnxruntime as ort
import numpy as np
import onnx
import matplotlib.pyplot as plt
from tqdm import tqdm

model = onnx.load('./best.onnx')
print([input.name for input in model.graph.input])
print([output.name for output in model.graph.output])

#get model class names with onnx names metadata
import json
import ast

class_dict = ast.literal_eval(model.metadata_props[-1].value)
print(class_dict)
del model

sess = ort.InferenceSession("./best.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

from PIL import Image

#classification case  - yolov8n-cls
def classify_image(image_path) -> str:
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 640))
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    img = np.ascontiguousarray(img)
    ort_inputs = {sess.get_inputs()[0].name: img}
    ort_outs = sess.run(None, ort_inputs)
    result = np.argmax(ort_outs[0])
    return class_dict[result]

print(classify_image('./lelah_8.png'))
    