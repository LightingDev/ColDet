from keras.models import load_model  # TensorFlow backend required
import cv2  # OpenCV for webcam
import numpy as np  # For numerical operations

# ⚙️ Setup
np.set_printoptions(suppress=True)  # Disable scientific notation

# Load model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# 🎥 Initialize Webcam
camera = cv2.VideoCapture(0)

# 🔁 Live Prediction Loop
while True:
    # Capture frame from webcam
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize to model input size (224x224)
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image_resized)

    # Preprocess image for model
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1  # Normalize to [-1, 1]

    # Predict
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction
    print(f"Class: {class_name[2:].strip()} | Confidence: {np.round(confidence_score * 100, 2)}%")

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
