from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import onnxruntime as ort
import numpy as np
import onnx
#get model class names with onnx names metadata
import base64
import io
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from PIL import Image

class ImageRequest(BaseModel):
    base64_img_str: str
model = onnx.load('./model/best.onnx')
print([input.name for input in model.graph.input])
print([output.name for output in model.graph.output])

class_dict = {0: """Bayi sepertinya mengalami kelelahan.\nPerbanyak istirahat dan makan yang cukup.""", 
              1: 'Bayi menangis tanda terdapat keadaan yang tidak nyaman.\nHal ini bisa disebabkan rasa nyeri atau sesak pada bayi.\nHubungi dokter segera!', 
              2: 'Bayi mengalami demam.\nSegera beri obat penurun panas', 
              3: 'Bayi dalam foto terlihat sehat.'}
print(class_dict)
del model

sess = ort.InferenceSession("./model/best.onnx", providers=['CPUExecutionProvider'])


#classification case  - yolov8n-cls
def classify_image(base64_img_str) -> str:
    imgdata = base64.b64decode(base64_img_str)
    img = Image.open(io.BytesIO(imgdata)).convert('RGB')
    img = img.resize((640, 640))
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    img = np.ascontiguousarray(img)
    ort_inputs = {sess.get_inputs()[0].name: img}
    ort_outs = sess.run(None, ort_inputs)
    print(ort_outs)
    if np.max(ort_outs[0]) < 0.4:
        return "Gambar yang Anda masukkan kurang sesuai. Mohon gunakan gambar bayi yang jelas!"
    result = np.argmax(ort_outs[0])
    return class_dict[result]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/classify/")
async def classify_image_api(image: ImageRequest):
    try:
        result = classify_image(image.base64_img_str)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

### html page with upload button, convert image to base64 and send to API
@app.get("/classify/", response_class=HTMLResponse)
def read_form():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Classifier</title>
  <script>
    function uploadImage() {
      const fileInput = document.getElementById('image-upload');
      const reader = new FileReader();

      reader.onload = function(e) {
        const imageData = e.target.result;
        // Send image data to the API
        fetch('/classify/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            base64_img_str: imageData.split(',')[1] // Extract base64 data after comma
          })
        })
        .then(response => response.json())
        .then(data => {
          const resultElement = document.getElementById('classification-result');
          resultElement.textContent = `Classification Result: ${data.result}`;
          // Display the image
            const previewElement = document.getElementById('image-preview');
            previewElement.innerHTML = `<img src="${imageData}" alt="Uploaded Image" style="max-width: 300px">`;
        })
        .catch(error => {
          console.error('Error:', error);
          alert('An error occurred during classification.');
        });
      };

      reader.readAsDataURL(fileInput.files[0]);
    }
  </script>
</head>
<body>
  <h1>Upload an Image for Classification</h1>
  <input type="file" id="image-upload" accept="image/*" onchange="uploadImage()">
  <br>
  
  <span id="image-preview"></span>
  <br>
  <span id="classification-result"></span>
</body>
</html>

    """
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)