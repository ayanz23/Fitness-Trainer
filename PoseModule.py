import cv2
import mediapipe as mp
import numpy as np

# Class to detect poses in images
class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        # Initialize parameters for pose detection
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe drawing and pose modules
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth, 
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        # Convert the image from BGR (OpenCV format) to RGB (MediaPipe format)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image to find pose landmarks
        self.results = self.pose.process(imgRGB)
        # Check if landmarks are found and if drawing is enabled
        if self.results.pose_landmarks and draw:
            # Draw landmarks on the image
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img  # Return the image with drawn landmarks

    def findPosition(self, img, draw=True, angles=None):
        # Initialize a list to hold landmark positions
        lmList = []
        # Check if pose landmarks are detected
        if self.results.pose_landmarks:
            # Iterate through each landmark
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Get image dimensions
                h, w, c = img.shape
                # Calculate the coordinates of the landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Append the landmark ID and coordinates to the list
                lmList.append([id, cx, cy])
                # Draw a circle
                if draw: 
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    
                    # Display angles next to specific keypoints if provided
                    if angles is not None:
                        hip_angle, knee_angle, ankle_angle = angles
                        # Display hip angle next to hip keypoint (id 23 or 24)
                        if id == 23 or id == 24:
                            if hip_angle is not None:
                                cv2.putText(img, f"{int(hip_angle)}", (cx + 15, cy), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        # Display knee angle next to knee keypoint (id 25 or 26)
                        elif id == 25 or id == 26:
                            if knee_angle is not None:
                                cv2.putText(img, f"{int(knee_angle)}", (cx + 15, cy), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        # Display ankle angle next to ankle keypoint (id 27 or 28)
                        elif id == 27 or id == 28:
                            if ankle_angle is not None:
                                cv2.putText(img, f"{int(ankle_angle)}", (cx + 15, cy), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return lmList  # Return the list of landmark positions

def main():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)  # Change to 0 for webcam
    detector = poseDetector()  # Create an instance of the poseDetector class
    while True:
        # Read a frame from the webcam
        success, img = cap.read()
        # Check if the frame was successfully captured
        if not success:
            print("Failed to access webcam.")
            break

        # Find and draw pose landmarks on the image
        img = detector.findPose(img)
        # Get the positions of the landmarks
        lmList = detector.findPosition(img, draw=False)

        # Display the image with pose tracking
        cv2.imshow("Live Webcam Pose Tracking", img)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()