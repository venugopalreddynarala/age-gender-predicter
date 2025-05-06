
Real-Time Age and Gender Detection using OpenCV
This project detects human faces in real-time from webcam feed and predicts the age group and gender using deep learning models in OpenCV.

<!-- Replace with your actual demo gif if available -->

📌 Features
Real-time face detection with OpenCV DNN

Age & gender prediction using pretrained Caffe models

Optimized to detect once every 2 seconds for performance

Preprocessing for better accuracy (histogram equalization)

Works from webcam or static image

🔧 Requirements
Python 3.7+

OpenCV (with DNN module)

Install dependencies using pip:

bash
Copy code
pip install opencv-python opencv-contrib-python
📁 Model Files
Download the following pretrained models and place them in the project directory:

Model Type	File	Download Link
Face Detector	opencv_face_detector_uint8.pb	Download
Face Config	opencv_face_detector.pbtxt	Download
Age Net	age_net.caffemodel	Download
Age Proto	age_deploy.prototxt	Download
Gender Net	gender_net.caffemodel	Download
Gender Proto	gender_deploy.prototxt	Download

🚀 How to Run
📷 Use Webcam (Default)
bash
Copy code
python detect_age_gender.py
🖼 Use Image
bash
Copy code
python detect_age_gender.py --image path_to_image.jpg
Press Q to quit the webcam window.

🧪 Output
The script displays a live window with:

Face bounding box

Predicted gender (Male/Female)

Predicted age range (e.g., 25–32)

Sample output in terminal:

makefile
Copy code
Detected: Female, (25-32)
Detected: Male, (38-43)
📂 Project Structure
bash
Copy code
.
├── detect_age_gender.py       # Main Python script
├── README.md
├── age_net.caffemodel
├── age_deploy.prototxt
├── gender_net.caffemodel
├── gender_deploy.prototxt
├── opencv_face_detector.pbtxt
└── opencv_face_detector_uint8.pb
🙌 Credits
Age/Gender models: LearnOpenCV

Face detector: OpenCV DNN module

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.