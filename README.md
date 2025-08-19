🥔 Potato Disease Detection (CNN + Flask)

This project uses a Convolutional Neural Network (CNN) to classify potato plant leaves into three categories:

✅ Healthy

🍂 Early Blight

🍁 Late Blight

The model is trained on the PlantVillage dataset and deployed using a Flask web application, allowing users to upload potato leaf images and get instant predictions.

🚀 Tech Stack

Python

TensorFlow / Keras

Flask

NumPy, Matplotlib

📂 Project Structure
Potato-disease-detection/
│
├── static/                # CSS, images, etc. for Flask
├── templates/             # HTML templates for Flask
├── training.ipynb         # Model training notebook
├── app.py                 # Flask app
├── model.h5               # Trained CNN model (if saved)
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

⚡ Features

Data preprocessing & CNN model training

Training history visualization (accuracy & loss plots)

Flask-based web app for real-time predictions

Upload potato leaf image → Get prediction instantly

🛠️ Installation & Setup

Clone the repository:

git clone https://github.com/Sandeep37-s/Potato-disease-detection.git
cd Potato-disease-detection


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows


Install dependencies:

pip install -r requirements.txt


Run the Flask app:

python app.py


Open in browser:

http://127.0.0.1:5000

📊 Dataset

The model is trained on the PlantVillage Dataset, specifically the potato leaf images.

📸 Screenshots

Add screenshots of your Flask app interface here (upload images in static/ and reference them).

🙌 Acknowledgements

Codebasics DL Playlist – for project guidance

PlantVillage Dataset – for providing high-quality leaf images


