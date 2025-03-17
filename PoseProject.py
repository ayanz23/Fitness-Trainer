import cv2
import numpy as np
import PoseModule as pm

def calculate_angle(a, b, ref_pt=np.array([0, 0])):
    # Convert 3D points to 2D by taking only x and y coordinates
    a = np.array([a[0], a[1]])  # Take only x,y coordinates
    b = np.array([b[0], b[1]])  # Take only x,y coordinates
    ref_pt = np.array([ref_pt[0], ref_pt[1]])

    a_ref = a - ref_pt
    b_ref = b - ref_pt

    cos_theta = (np.dot(a_ref, b_ref)) / (1.0 * np.linalg.norm(a_ref) * np.linalg.norm(b_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    degree = (180 / np.pi) * theta
    return int(degree)
 
def squat(keypoints):
    """Evaluate if the pose matches a proper squat and provide feedback."""
    if len(keypoints) < 29:  # Check if there are enough keypoints
        return ["Not enough keypoints detected"], None, None, None  # Not enough keypoints for evaluation

    # Fetch relevant landmarks for the right and left sides of the body
    r_shoulder, r_hip, r_knee, r_ankle, l_foot = keypoints[12], keypoints[24], keypoints[26], keypoints[28], keypoints[31]
    l_shoulder, l_hip, l_knee, l_ankle, r_foot = keypoints[11], keypoints[23], keypoints[25], keypoints[27], keypoints[32]
    nose = keypoints[0]  # Get nose position

    # Calculate the angle between the nose and shoulders
    offset_angle = calculate_angle(l_shoulder, r_shoulder, nose)

    # Check if the angle indicates a front view
    if offset_angle > 35:  # Threshold angle for front view detection
        return ["Front view detected, please show side view"], None, None, None  # Prompt for side view

    # Determine which side the user is facing
    if (abs(l_foot[2]- l_shoulder[2]) > abs(r_foot[2]- r_shoulder[2])):  # Facing right IRL
        shoulder, hip, knee, ankle = l_shoulder, l_hip, l_knee, l_ankle
    else:  # Facing left IRL
        shoulder, hip, knee, ankle = r_shoulder, r_hip, r_knee, r_ankle

    # Calculate angles with the vertical for feedback
    hip_vertical_angle = calculate_angle(np.array([shoulder[1], shoulder[2]]), np.array([hip[1], 0]), np.array([hip[1], hip[2]]))
    knee_vertical_angle = calculate_angle(np.array([hip[1], hip[2]]), np.array([knee[1], 0]), np.array([knee[1], knee[2]]))
    ankle_vertical_angle = calculate_angle(np.array([knee[1], knee[2]]), np.array([ankle[1], 0]), np.array([ankle[1], ankle[2]]))

    # Initialize an empty list for feedback
    feedback_list = []
    
    # Add feedback based on calculated angles
    if hip_vertical_angle < 20:
        feedback_list.append("Bend Forwards")
    elif hip_vertical_angle > 45:
        feedback_list.append("Bend Backwards")
    
    if 50 < knee_vertical_angle < 80:
        feedback_list.append("Lower Your Hips")
    elif knee_vertical_angle > 95:
        feedback_list.append("Squat Too Deep")
    
    if ankle_vertical_angle > 30:
        feedback_list.append("Knee Falling Over Toe")
    
    # If no specific feedback, add "Good Squat" only if in a squat position
    if not feedback_list and 70 <= knee_vertical_angle <= 95:
        feedback_list.append("Good Squat")
    
    return feedback_list, hip_vertical_angle, knee_vertical_angle, ankle_vertical_angle

def main():
    """Main function to perform fitness pose recognition."""
    camera = cv2.VideoCapture(0)  # Initialize webcam capture
    pose_detector = pm.poseDetector()  # Create an instance of the pose detector
    
    # Initialize counter variables
    squat_count = 0
    is_squatting = False
    
    while True:
        ret, frame = camera.read()  # Read a frame from the webcam
        if not ret:
            print("Error accessing the webcam. Exiting...")
            break

        frame = pose_detector.findPose(frame, draw=True)
        
        feedback_list, hip_angle, knee_angle, ankle_angle = [], None, None, None
        view_prompt = ""
        
        # First get landmarks without drawing to calculate angles
        landmarks = pose_detector.findPosition(frame, draw=True)
        
        if landmarks:
            feedback_list, hip_angle, knee_angle, ankle_angle = squat(landmarks)

            if isinstance(feedback_list, list) and any("Front view detected" in item for item in feedback_list):
                view_prompt = feedback_list[0]  # Show side view prompt
                feedback_list = []  # Clear feedback list when front view is detected
            else:
                view_prompt = ""
                
            # Now draw landmarks with angles
            landmarks = pose_detector.findPosition(frame, draw=True, angles=(hip_angle, knee_angle, ankle_angle))
            
            # Count squats based on knee angle and side view
            if knee_angle is not None and view_prompt == "":
                if knee_angle >= 55 and knee_angle <= 80 and not is_squatting:
                    is_squatting = True
                elif knee_angle > 80 and is_squatting:
                    is_squatting = False
                    squat_count += 1  # Increment only if side view is shown

        # Define text properties for consistency
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)  # White color for all text
        text_thickness = 2
        text_shadow_color = (0, 0, 0)  # Black shadow for better visibility
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Display counter in top right
        counter_text = f"Count: {squat_count}"
        counter_size = cv2.getTextSize(counter_text, font, 1, text_thickness)[0]
        # Add shadow for better visibility
        cv2.putText(frame, counter_text, (w - counter_size[0] - 19, 51), 
                    font, 1, text_shadow_color, text_thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, counter_text, (w - counter_size[0] - 20, 50), 
                    font, 1, text_color, text_thickness, cv2.LINE_AA)

        # Display welcome message with shadow
        cv2.putText(frame, "Welcome to Fitness Trainer. Perform Squats.", (19, 51),
                    font, 1, text_shadow_color, text_thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, "Welcome to Fitness Trainer. Perform Squats.", (20, 50),
                    font, 1, text_color, text_thickness, cv2.LINE_AA)
        
        # Display view prompt with shadow if needed
        if view_prompt:
            cv2.putText(frame, view_prompt, (19, 101),
                        font, 1, text_shadow_color, text_thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, view_prompt, (20, 100),
                        font, 1, text_color, text_thickness, cv2.LINE_AA)
        
        # Display feedback list with shadow
        feedback_text = "Feedback: "
        if feedback_list:
            feedback_text += ", ".join(feedback_list)
        
        cv2.putText(frame, feedback_text, (19, 151),
                    font, 0.8, text_shadow_color, text_thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, feedback_text, (20, 150),
                    font, 0.8, text_color, text_thickness, cv2.LINE_AA)
        
        # Display controls at the bottom of the screen
        controls_text = "Controls: Q - Quit | R - Reset Counter"
        controls_size = cv2.getTextSize(controls_text, font, 0.7, text_thickness)[0]
        # Add shadow for better visibility
        cv2.putText(frame, controls_text, (w//2 - controls_size[0]//2 - 1, h - 29),
                    font, 0.7, text_shadow_color, text_thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, controls_text, (w//2 - controls_size[0]//2, h - 30),
                    font, 0.7, text_color, text_thickness, cv2.LINE_AA)

        cv2.imshow("Fitness Pose Detection", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            squat_count = 0  # Reset counter when 'r' is pressed

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
