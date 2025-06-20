🧠 Face Mask Detection using Deep Learning
A deep learning-based real-time face mask detection system built with TensorFlow, Keras, and MobileNetV2 architecture. The system can classify faces with and without masks from images or video feeds.

📁 Dataset
The model is trained on a custom dataset with two classes:
with_mask 😷
without_mask 😐
You can add or use other open datasets like the Kaggle Face Mask Detection Dataset if required.

🧰 Technologies Used
Python
TensorFlow / Keras
OpenCV
MobileNetV2 (Transfer Learning)
Matplotlib
Scikit-learn
Jupyter Notebook

 Installation
1. Clone the repository:
git clone https://github.com/Arpit-ve/Mask_Detection.git
cd face-mask-detection

2. Install required libraries:
 pip install -r requirements.txt

3. Prepare your dataset folder:
   dataset/
├── with_mask/
└── without_mask/

4. Run the training script in Jupyter Notebook:
 face_mask_training.ipynb

📈 Model Training
Uses MobileNetV2 for transfer learning (pretrained on ImageNet).
Trains only the top layers while freezing base layers initially.
Implements data augmentation for better generalization.

📊 After training, you'll get:
A trained model: mask_detector.model
A performance plot: plot.png
Classification report on test data.

🧪 Evaluation (Sample Output)

              precision    recall  f1-score   support

   with_mask       0.97      0.95      0.96       138
without_mask       0.96      0.98      0.97       140

    accuracy                           0.96       278

    🚀 Future Improvements
Real-time mask detection using OpenCV and webcam.
Flask web app or streamlit interface for deployment.
Alert system (sound/email) for unmasked face detection.

📸 Screenshots


📂 Project Structure
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── face_mask_training.ipynb
├── mask_detector.model
├── plot.png
└── README.md

🧑‍💻 Author
Arpit Verma
🔗 LinkedIn | 🐍 BCA Student | 💻 Tech Enthusiast | 📸 Photographer

📜 License
This project is licensed under the MIT License.



