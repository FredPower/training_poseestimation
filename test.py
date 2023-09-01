import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a,b,c):
     a = np.array(a)
     b = np.array(b)
     c = np.array(c)

     rad = np.arctan2(c[1]-b[1], c[0]-b[0])-np.arctan2(a[1]-b[1], a[0]-b[0])
     angle = np.abs(rad*180.0/np.pi)
     if angle>180.0:
          angle = 360-angle
     return angle

## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture('video3.mp4')
angle=0
while cap.isOpened():
     # read frame
     _, frame = cap.read()
     try:
         # resize the frame for portrait video
         frame = cv2.resize(frame, (600, 600))
         # convert to RGB
         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         #Angle display
         angle_display = (0,0,255)
         
         # process the frame for pose detection
         pose_results = pose.process(frame_rgb)
         #print(pose_results.pose_landmark)
         
         #
         try:
              landmarks = pose_results.pose_landmarks.landmark
              shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
              elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
              wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
              angle = calculate_angle(shoulder,elbow,wrist)
         except:
              pass
         #
         # draw skeleton on the frame
         mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(255,0,0),thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0,255,0),thickness=2, circle_radius=2))

         if angle < 120:
              angle_display = (0,255,0)
         cv2.putText(frame,str(int(angle)),(50,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,angle_display,2,cv2.LINE_4)
         cv2.imshow('Output', frame)
     except:
         print("error-ended")
         break
    
     if cv2.waitKey(1) == ord('q'):
          break
          
cap.release()
cv2.destroyAllWindows()