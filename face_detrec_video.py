'''
Sarath Kumar
sarathsmartzs[at]gmail[dot]com
SACET, Chirala

    Real-Time detection & prediction of subjects/persons in
        video recording by in-built camera.
    If there is any intruder (trained/ unknown subjects) attack, it posts on your
        facebook timeline to notify you and your friends/ neighbours.

Working:
    Takes images stored in first path and traines faceRecognizer models.
    Then starts recording video from camera and shows detected subjects.

Usage:
    face_detrec_video.py <full/path/to/root/images/folder>

Takes one argument:
    1. Input folder which contains sub-folders of subjects/ persons.
        There should be images saved in subfolders which are used to train.
'''

import cv2
#import cv2.cv as cv
import numpy as np
import os
import sys, time
import requests , facebook
import face_recognition
import dlib
import datetime
from twilio.rest import Client


def get_images(path, size):
    '''
    path: path to a folder which contains subfolders of for each subject/person
        which in turn cotains pictures of subjects/persons.

    size: a tuple to resize images.
        Ex- (256, 256)
    '''
    sub= 0
    images, labels= [], []
    people= []

    for subdir in os.listdir(path):
        for image in os.listdir(path+ "/"+ subdir):
            img= cv2.imread(path+os.path.sep+subdir+os.path.sep+image, cv2.IMREAD_GRAYSCALE)
            img= cv2.resize(img, size)

            images.append(np.asarray(img, dtype= np.uint8))
            #nbr = int(os.path.split(image)[1].split(".")[0].replace("faces-", ""))
	    #for (x,y,w,h) in faces:
		#images.append(image[q:q+s, p:p+r])
		#labels.append(nbr)
            labels.append(sub)

            #cv2.imshow("win", img)
            #cv2.waitKey(10)

        people.append(subdir)
        sub+= 1

    return [images, labels, people]

def detect_faces(image):
    '''
    Takes an image as input and returns an array of bounding box(es).
    '''
    frontal_face= cv2.CascadeClassifier("face.xml")
    bBoxes= frontal_face.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
        #,flags=cv2.CASCADE_SCALE_IMAGE)

    return bBoxes

def train_model(path):
    '''
    Takes path to images and train a face recognition model
    Returns trained model and people
    '''
    [images, labels, people]= get_images(sys.argv[1], (256, 256))
    #print([images, labels])

    labels= np.asarray(labels, dtype= np.int32)

    # initializing eigen_model and training
    print("Initializing eigen FaceRecognizer and training...")
    sttime= time.clock()
    eigen_model= cv2.face.EigenFaceRecognizer_create()
    eigen_model.read("face.xml")
    #eigen_model= cv2.createEigenFaceRecognizer()
    eigen_model.train(images, labels)
    print("\tSuccessfully completed training in "+ str(time.clock()- sttime)+ " Secs!")

    return [eigen_model, people]

def majority(mylist):
    '''
    Takes a list and returns an element which has highest frequency in the given list.
    '''
    myset= set(mylist)
    ans= mylist[0]
    ans_f= mylist.count(ans)

    for i in myset:
        if mylist.count(i)> ans_f:
            ans= i
            ans_f= mylist.count(i)

    return ans

def post_on_facebook(name, counter, picture_name):
    '''
    Takes name of intruder and posts on your facebok timeline.
    You need to get access_token from facebook GraphAPI and paste it below.
    '''
    
    token= "EAACEdEose0cBANd5Glvtj19SqrCULnU8C4Bdf846KfZCvFHed6KOk0Yx5G8uQjbCh1ZBxtpZC8l6bZCjlbwDgylXAIkaIjZBQAPpXbnE84fp3UblE63xrdgAidBwmGsXKz05qxZCYc2M0Pu6elatR23c5bsQmcS1ZBOsSBCfdRTV7CKnM10x2EChXlfyBW2PMFfkZCeufh50QgZDZD"
    url= "https://graph.facebook.com/me/feed"

    graph= facebook.GraphAPI(access_token= token)
    if name =="sarath":
	pass
       #my_message1= "Sarath is not in his room at present and '"+ intruder+ "' entered into his room without permission."
    else:
	    my_message1= "Sarath is not in his room at present and Someone has entered into his room without permission."
	    my_message2= "PS: This is automatically posted by 'intruder alert system' built by Sarath!\n"
	    final_message= my_message1+"\n\n"+my_message2+ "\n"+ str(counter)

	    #post on facebook using requests.
	    # params= {"access_token": token, "message": final_message}
	    # posted= requests.post(url, params)
	    #if str(posted)== "<Response [200]>":
	    #print("\tSuccessfully posted on your timeline.")
	    #else:
	    #print("\tPlease check your token and its permissions.")
	    #print("\tYou cannot post same message more than once in a single POST request.")

	    #post on facebook using python GraphAPI
	    # U can get the below details from the twilio.com account
	    
	    client = Client("AC5f50d4d72f0d2c0f4f2d67ea3d50a40f", "0c429349b366f65b5b04f60c96dce796")

	    # change the "from_" number to your Twilio number and the "to" number
	    # to the phone number you signed up for Twilio with, or upgrade your
	    # account to send SMS to any phone number
	    client.messages.create(to="+91123456789",
                                from_="+14843928163 ",
	                        body="Hi! Someone has been entered into your room. Check your Facebook timeline!")
	    graph.put_photo(image= open(picture_name), message= final_message)



if __name__== "__main__":
    if len(sys.argv)!= 2:
        print("Wrong number of arguments! See the usage.\n")
        print("Usage: face_detrec_video.py <full/path/to/root/images/folder>")
        sys.exit()

    arg_one= sys.argv[1]
    eigen_model, people= train_model(arg_one)

    #starts recording video from camera and detects & predict subjects
    
    cap= cv2.VideoCapture(0)#sarath'''
    sarath_image = face_recognition.load_image_file("sarath.jpg")
    sarath_face_encoding = face_recognition.face_encodings(sarath_image)[0]
    # Load a second sample picture and learn how to recognize it.
    praveen_image = face_recognition.load_image_file("praveen.JPG")
    praveen_face_encoding = face_recognition.face_encodings(praveen_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        sarath_face_encoding,praveen_face_encoding
    ]
    known_face_names = [
        "sarath","praveen"
    ]


    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True


    counter= 0
    last_20= [0 for i in range(20)]
    final_5= []
    box_text= ""

    while(True):
        # Grab a single frame of video
        ret, frame = cap.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time

        bBoxes= detect_faces(rgb_small_frame)
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
  
                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        # Display the results
        for (p, q, r, s), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            p *= 4
            q *= 4
            r *= 4
            s *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (s, p), (q, r), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (s, r - 35), (q, r), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (s + 6, r - 6), font, 1.0, (255, 255, 255), 1)#sarath'''




        #for bBox in bBoxes:
        #   (p,q,r,s)= bBox
        #   cv2.rectangle(frame, (p,q), (p+r,q+s), (225,0,0), 2)

        #   rgb_small_frame= small_frame[q:q+s, p:p+r]
        #   rgb_small_frame= cv2.resize(rgb_small_frame, (256, 256))

            #[predicted_label, predicted_conf]= eigen_model.predict(np.asarray(rgb_small_frame))
            #last_20.append(predicted_label)
            #last_20= last_20[1:]

            '''
            counter modulo x: changes value of final label for every x frames
            Use max_label or predicted_label as you wish to see in the output video.
                But, posting on facebook always use max_label as a parameter.
            '''

            #cv2.putText(frame, name, (p-20, q-5), font, 1.3, (25,0,225), 2)

            if counter%10== 0:
                max_label= majority(last_20)
                #box_text= format("Person: ")#+ people[max_label])
                #box_text= format("Person: "+ people[predicted_label])

                if name== "sarath":
                	cv2.rectangle(frame, (s, p), (q, r), (0, 0, 255), 2)
                	cv2.putText(frame, name, (s + 6, r - 6), font, 1.0, (255, 255, 255), 1)

                	pass
                	
                else:
                	cv2.rectangle(frame, (s, p), (q, r), (0, 0, 255), 2)
                	cv2.putText(frame, name, (s + 6, r - 6), font, 1.0, (255, 255, 255), 1)

		        if counter> 20:
		            print("Will post on facebook timeline if this counter reaches to 2: "+ str(len(final_5)+ 1))
		            final_5.append(max_label)       #it always takes max_label into consideration
		            if len(final_5)== 2:
		                final_label= majority(final_5)
		                print("Intruder entered ")#+ people[final_label])
		                print("Posting on your facebook timeline...")
		                picture_name= "frame.jpg"
		                cv2.imwrite(picture_name, frame)
		                post_on_facebook(name, counter, picture_name)
		                final_5= []
				break
				#continue



        cv2.imshow("Video Window", frame)
        counter+= 1

        if (cv2.waitKey(1) & 0xFF== 27):
            break

    cv2.destroyAllWindows()
    
