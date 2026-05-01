# CattleBreedClassifier

<<<<<<< HEAD
An end-to-end deep learning system designed to accurately classify five distinct cattle breeds using advanced Convolutional Neural Network (CNN) architectures and a modern Flask-based web interface.
=======
>>>>>>> 8d4239bba74d9d72303d37c310f2a84f064d3486

## About the Project

This project aims to automate cattle breed identification using computer vision. The system is trained to recognize five specific breeds: Ayrshire, Brown Swiss, Holstein Friesian, Jersey, and Red Dane. 

To achieve optimal performance under hardware constraints, multiple deep learning architectures were evaluated, including a Custom CNN built from scratch, alongside industry-standard models utilizing transfer learning: VGG16, ResNet50, MobileNetV2, and EfficientNetB0. Data augmentation techniques and learning rate scheduling were implemented to prevent overfitting and improve model generalization.

The final system features a modern, glassmorphism-styled web interface built with Flask, allowing users to upload images via drag-and-drop and receive instant, real-time prediction confidence bars.

## Key Features

- **Multi-Class Classification:** Accurately distinguishes between 5 cattle breeds.
- **Model Benchmarking:** Comprehensive comparison of model parameters, training times, and accuracy across 5 different architectures.
- **Best Performing Model:** EfficientNetB0 achieved the highest accuracy (88.46%) with optimal parameter efficiency (only 4M parameters).
- **Interactive Web Interface:** Modern, responsive UI with real-time prediction visualizations using Flask and vanilla JS.
- **Robust Training Pipeline:** Includes data augmentation (rotation, flipping, color jitter), early stopping, and learning rate reduction on plateau.

## Installation and Usage

To run this project locally, follow these steps:

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/CattleBreedClassifier.git
cd CattleBreedClassifier
```

2. **Set up a virtual environment (optional but recommended):**
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the web application:**
```bash
python app.py
```

5. **Access the interface:**
Open your web browser and navigate to `http://localhost:5000` to use the drag-and-drop classifier.

## Repository Structure

- `app.py`: Main Flask application server handling image uploads and inference.
- `cnn_assignment.py`: The core deep learning script containing model definitions, training loops, evaluation metrics, and plot generation.
- `save_model.py`: Utility script to fine-tune and export the best performing model (EfficientNetB0) for production use.
- `templates/` & `static/`: HTML, CSS (glassmorphism theme), and JavaScript files for the web frontend.
- `results/`: Directory containing generated performance graphs, confusion matrices, and error analysis visualizations.

## Technologies Used

- **Deep Learning Framework:** PyTorch, Torchvision
- **Web Backend:** Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Data Manipulation & Visualization:** NumPy, Scikit-learn, Matplotlib, Seaborn
