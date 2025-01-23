import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Load the fixed model
model = load_model("fixed_keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to preprocess the image
def preprocess_frame(frame):
    # Resize the frame to 224x224
    size = (224, 224)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.asarray(image)
    
    # Normalize the image data to -1 to 1
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Add batch dimension
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    data = preprocess_frame(frame)

    # Make predictions
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Display the prediction on the frame
    text = f"{class_name}: {confidence_score:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the prediction using OpenCV
    cv2.imshow("Webcam Prediction", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
