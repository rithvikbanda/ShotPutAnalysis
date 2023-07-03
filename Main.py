#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import csv
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[263]:


pip install youtube_dl


# In[264]:


import youtube_dl

# Set the YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=eiTEuw5REPY"

# Set the options for downloading
options = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Set the format of the downloaded video
    'outtmpl': 'test1.mp4'  # Set the output file name
}

# Download the video
with youtube_dl.YoutubeDL(options) as ydl:
    ydl.download([youtube_url])

print("Video downloaded successfully.")


# In[ ]:


import cv2
import mediapipe as mp
import pandas as pd
import csv

mp_pose = mp.solutions.pose

video_file = 'Glide.mp4'

cap = cv2.VideoCapture(video_file)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_count = 0



    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Recolor the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make the pose detection
        results =  pose.process(image)

        # Recolor the frame back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        frame_count = 0
        class_name = 'Glide'
       
            
        try:
                landmark = results.pose_landmarks.landmark
                pose_row = list(np.array([[frame_count, landmark.x, landmark.y, landmark.z, landmark.visibility]]))
            
                row = pose_row
                row.insert(0,class_name)
        except:
                print("error")
                pass

        # Display the frame
        cv2.imshow('Mediapipe Feed', image)

        # Increment the frame count
        frame_count += 1

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


#if results.pose_landmarks:
       #    landmarks = results.pose_landmarks.landmark

        #   if landmarks is not None:
         #      for landmark in landmarks:
                   # Append the landmark data and action label to the CSV
           #        with open('coords.csv', mode='a', newline='') as f:
            #           csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
             #          csv_writer.writerow([frame_count,landmark.x, landmark.y, landmark.z, landmark.visibility, 'glide'])
       


# In[3]:


row


# In[102]:


import cv2
import mediapipe as mp
import numpy as np
import csv

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_file = 'Stand9.mp4'

cap = cv2.VideoCapture(video_file)
# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_count = 0  # Initialize frame count
    class_name = 'Stand'
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Concate rows
            row = pose_row

            # Append frame count
         
            row.insert(0, frame_count)
            row.insert(0, class_name)
            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

            frame_count += 1  # Increment frame count
        except:
            pass

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[12]:


len(row)


# In[13]:


row.insert(0, class_name)


# In[10]:


num_coords = len(results.pose_landmarks.landmark)


# In[11]:


num_coords


# In[35]:


landmarks = ['class', 'frame_count']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


# In[6]:


landmarks


# In[37]:


with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[16]:


df = pd.read_csv('coords.csv')


# In[17]:


df.head()


# In[106]:


df[df['class']=='Stand']


# In[18]:


x = df.drop('class', axis=1)
y = df['class']


# In[19]:


x


# In[20]:


y


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


# In[120]:





# In[21]:


from sklearn.linear_model import LogisticRegression, RidgeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline


# In[22]:


pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier())
}


# In[23]:


list(pipelines.values())[0]


# In[26]:


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model


# In[27]:


fit_models


# In[28]:


fit_models['rf'].predict(X_test)


# In[29]:


import pickle


# In[30]:


for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))


# In[31]:


with open('throw_tracker', 'wb') as f:
    pickle.dump(fit_models['rf'], f)


# In[32]:


with open('throw_tracker', 'rb') as f:
    model = pickle.load(f)


# In[33]:


model


# In[1]:


import cv2
import mediapipe as mp
import numpy as np
import csv

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_file = 'Spin10.mp4'

cap = cv2.VideoCapture(video_file)
# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_count = 0  # Initialize frame count
    class_name = 'Stand'
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Concate rows
            row = pose_row
            row.insert(0, frame_count)
            
            
            x = pd.DataFrame([row])
            throw_tracker_class = model.predict(x)[0]
            throw_tracker_prob = model.predict_proba(x)[0]
            print(throw_tracker_class, throw_tracker_prob)


            # Append frame count
         
         #   row.insert(0, frame_count)
         #   row.insert(0, class_name)
            # Export to CSV
         #   with open('coords.csv', mode='a', newline='') as f:
          #      csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
           #     csv_writer.writerow(row)

            text = f"Throw Type: {throw_tracker_class}"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame_count += 1  # Increment frame count
        except:
            print('error')
            pass

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[40]:


x


# In[41]:


throw_tracker_class = model.predict(x)[0]


# In[42]:


feature_names


# In[204]:


sklearn - ver


# In[ ]:




