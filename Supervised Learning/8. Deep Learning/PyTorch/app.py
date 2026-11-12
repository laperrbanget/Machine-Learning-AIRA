import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. Definisikan ulang class model kamu (sama persis seperti di notebook)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Muat model
# Pastikan file 'model.pth' ada di folder yang sama dengan app.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# 3. Daftar nama kelas
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 4. Fungsi prediksi
def predict(image):
    # Preprocessing: gambar dari user (PIL) -> grayscale -> resize 28x28 -> tensor
    # Gradio memberi gambar dalam format numpy array atau PIL, tergantung type.
    # Kita asumsikan inputnya PIL Image.
    if image is None:
        return "No image provided"
    
    # Konversi ke grayscale dan resize
    img = Image.fromarray(image).convert('L').resize((28, 28))
    
    # Ubah ke numpy, normalisasi 0-1, lalu ke tensor (1, 1, 28, 28)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
    
    # Prediksi
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)
    
    return class_names[predicted.item()], float(conf.item())

# 5. Buat antarmuka Gradio
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Gambar Pakaian", type="numpy"),
    outputs=[
        gr.Label(label="Prediksi Kelas"),
        gr.Number(label="Confidence Score")
    ],
    title="Fashion-MNIST Classifier",
    description="Upload gambar pakaian (atau gambar dari internet) untuk diprediksi jenisnya."
)

if __name__ == "__main__":
    interface.launch()