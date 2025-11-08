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

ColDet is a compact deep learning project that identifies the dominant color in the center of an image.
It’s powered by a simple Keras model saved in .h5 format, making it lightweight, easy to integrate, and fast to deploy on any platform that supports TensorFlow.

⸻

🧠 What It Does

ColDet looks at the center region of an input image and predicts what color is most present there.
You can use it for:
	•	Detecting object color in simple scenes
	•	Building color-based sorting systems
	•	Real-time color recognition in robotics or automation projects

⸻

⚙️ How It Works
	1.	Input:
You feed in an image through the model or demo interface.
	2.	Processing:
The system crops the central portion of the image, normalizes pixel values, and feeds it into a trained .h5 model.
	3.	Prediction:
The neural network predicts the main color at the image center (e.g., red, green, blue, etc.) and outputs the confidence score.

⸻

🚀 Try It Online

👉 Live Demo on Hugging Face￼
You can upload your own images or use sample ones to see how ColDet identifies the center color instantly.

⸻

🧩 Tech Stack
	•	TensorFlow / Keras — model training and inference (.h5 format)
	•	Python — backend logic
	•	Gradio — for the demo UI
	•	NumPy / OpenCV — preprocessing and color extraction

⸻

💾 Model Info
	•	Model file: coldet_model.h5
	•	Input: Image (H, W, 3)
	•	Output: Predicted color label + confidence score
	•	Trained on: A dataset of labeled color samples

⸻

🔧 Run Locally

git clone https://github.com/LightingDev/ColDet.git
cd ColDet
pip install -r requirements.txt
python main.py

Place your coldet_model.h5 file in the project root before running.

⸻

📜 License

Apache2.0 License — Read LISENCE file for terms and conditions.