mkdir -p models/face_recognition
# shellcheck disable=SC2164
cd models/face_recognition
wget https://raw.githubusercontent.com/ageitgey/face_recognition_models/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat
wget https://raw.githubusercontent.com/ageitgey/face_recognition_models/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat

# shellcheck disable=SC2103
cd ..
# shellcheck disable=SC2164
mkdir -p mask_detector && cd mask_detector
wget https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/mask_detector.model
