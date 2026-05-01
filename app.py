import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Modeli Yukle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLS = 5
CLS_NAMES = ['Ayrshire', 'Brown Swiss', 'Holstein Friesian', 'Jersey', 'Red Dane']

# Modeli tanimla
base = models.efficientnet_b0(weights=None)
in_f = base.classifier[-1].in_features
base.classifier[-1] = nn.Linear(in_f, NUM_CLS)
model = base.to(DEVICE)

# Kaydedilen agirliklari yukle
checkpoint = torch.load('efficientnet_cattle.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Goruntu on isleme adimlari (egitimdeki gibi)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # ToTensor [0, 1] araligina getirir
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya secilmedi'}), 400

    try:
        img = Image.open(file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(img_tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        
        results = []
        for i, prob in enumerate(probs):
            results.append({
                'class': CLS_NAMES[i],
                'probability': float(prob)
            })
        
        # Olasiliga gore sirala (en yuksek en ustte)
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
