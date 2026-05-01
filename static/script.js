const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultCard = document.getElementById('resultCard');
const loadingOverlay = document.getElementById('loadingOverlay');
const topClass = document.getElementById('topClass');
const topConfidence = document.getElementById('topConfidence');
const barsContainer = document.getElementById('barsContainer');

let selectedFile = null;

// Sürükle-bırak olayları
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            selectedFile = file;
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = function() {
                imagePreview.src = reader.result;
                previewContainer.style.display = 'flex';
                analyzeBtn.disabled = false;
                resultCard.style.display = 'none'; // Yeni resim seçildiğinde eski sonucu gizle
            }
        } else {
            alert('Lütfen sadece resim dosyası yükleyin (JPEG, PNG).');
        }
    }
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Dosya seçme diyaloğunu tetiklememek için
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.style.display = 'none';
    analyzeBtn.disabled = true;
    resultCard.style.display = 'none';
});

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Yükleniyor ekranını göster
    document.querySelector('.main-content').style.position = 'relative';
    loadingOverlay.style.display = 'flex';
    resultCard.style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            alert('Hata: ' + data.error);
        }
    } catch (error) {
        alert('Sunucuya bağlanırken bir hata oluştu: ' + error);
    } finally {
        // Yükleniyor ekranını gizle
        loadingOverlay.style.display = 'none';
    }
});

function displayResults(data) {
    // En yüksek tahmin
    const top = data.top_prediction;
    topClass.textContent = top.class;
    topConfidence.textContent = (top.probability * 100).toFixed(2) + '%';

    // Barları temizle ve yeniden oluştur
    barsContainer.innerHTML = '';
    
    data.predictions.forEach((pred, index) => {
        const percentage = (pred.probability * 100).toFixed(2);
        
        const barItem = document.createElement('div');
        barItem.className = 'bar-item';
        
        barItem.innerHTML = `
            <div class="bar-info">
                <span>${pred.class}</span>
                <span>${percentage}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: 0%"></div>
            </div>
        `;
        
        barsContainer.appendChild(barItem);
        
        // Animasyon için kısa bir gecikme ekle
        setTimeout(() => {
            barItem.querySelector('.bar-fill').style.width = `${percentage}%`;
        }, 100 + (index * 100));
    });

    resultCard.style.display = 'block';
    
    // Yumuşak kaydırma
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
