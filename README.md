## FaceRecognizer

A face recognition algorithm written in Python (haarcascade_frontalface algorithm).

Real-Time detection & prediction of subjects/persons in video recording with built-in camera.

If there is any intruder (known/ unknown subjects) attack, it automatically posts on your Facebook timeline to notify you and your friends/neighbors.

## Preview
Screen-shot while detecting & predicting subject(s) in real-time video.

![alt tag](../master/Preview1.png)

Screen-shot of Facebook post regarding intruder attack.

![alt tag](../master/Preview2.png)

## Usage of available files

|File Name|Used to|
|---------|-------|
|build_csv.py|build a csv file with paths to train images with labels.|
|detect_save_images.py|detect faces and save cropped images to output folder.|
|face_detect_recognition.py|detect & recognize faces in images and display them.|
|**face_detrec_video.py**|detect & recognize intruders in real-time video and notifies on facebook|
|face_recognition.py|train eigen_model & recognise faces in pre-cropped images|
|resize_rotate_images.py|pre-process images and saves output to output_folder|
