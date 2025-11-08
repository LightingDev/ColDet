<p align="center">
  <img src="LOGO.png" alt="Logo" width="670"/>
</p>


<p align="center">
  <a href="https://github.com/LightingDev/ColDet/">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  <a href="https://huggingface.co/spaces/theguywhosucks/coldet-demo/">
    <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  </a>
  <a href="https://huggingface.co/spaces/theguywhosucks/coldet-demo/">
    <img src="https://img.shields.io/badge/Demo-1E90FF?style=for-the-badge&logo=googlechrome&logoColor=white"/>
  </a>
</p>



⸻

🎯 ColDet — Color-Based Detection with Deep Learning

ColDet is a sleek, lightweight deep learning tool designed to detect the dominant color in the center of an image. It leverages a .h5 Keras model for quick, cross-platform deployment.

⸻

🧠 Key Features
	•	Center Color Detection: Focuses on the image center to determine the main color.
	•	Fast & Lightweight: Compact .h5 model for instant predictions.
	•	Real-Time Applications: Perfect for robotics, automation, or color-based sorting.

⸻

⚙️ How It Works
	1.	Input: Provide an image through the demo or your local script.
	2.	Processing: The system crops the central area, normalizes pixel values, and feeds it into the trained .h5 model.
	3.	Prediction: Outputs the dominant color at the image center with a confidence score.

⸻

🚀 Try the Live Demo

Upload your own images or experiment with sample images to see ColDet in action instantly.

⸻

🧩 Tech Stack

Technology	Purpose
TensorFlow / Keras	Model training and inference (.h5)
Python	Backend logic
Gradio	Demo UI
NumPy / OpenCV	Image preprocessing and color extraction


⸻

💾 Model Info
	•	File: coldet_model.h5
	•	Input: Image (H, W, 3)
	•	Output: Predicted color label + confidence
	•	Trained On: Labeled color sample dataset

⸻

🔧 Local Setup

git clone https://github.com/LightingDev/ColDet.git
cd ColDet
pip install -r requirements.txt
python main.py

Make sure your coldet_model.h5 file is placed in the project root before running.

⸻

📜 License

Apache 2.0 Lisence — Read LISENCE file for terms and conditions.

⸻

Designed for developers, researchers, and hobbyists seeking fast, intuitive color detection tools.