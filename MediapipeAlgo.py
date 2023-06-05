import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import datetime
mp_face_detector = mp.solutions.face_detection
def mpDnnDetectFaces(image, mp_face_detector, display = True):
    '''
    This function performs face(s) detection on an image using mediapipe deep learning based face detector.
    Args:
        image:            The input image with person(s) whose face needs to be detected.
        mp_face_detector: The mediapipe's face detection function required to perform the detection.
        display:          A boolean value that is if set to true the function displays the original input image, 
                          and the output image with the bounding boxes, and key points drawn, and also confidence 
                          scores, and time taken written and returns nothing.
    Returns:
        output_image: A copy of input image with the bounding box and key points drawn and also confidence scores written.
        results:      The output of the face detection process on the input image.
    '''
    
    # Get the height and width of the input image.
    image_height, image_width, _ = image.shape
    
    # Create a copy of the input image to draw bounding box and key points.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the current time before performing face detection.
    start = time()
    
    # Perform the face detection on the image.
    results = mp_face_detector.process(imgRGB)
    
    # Get the current time after performing face detection.
    end = time()

    # Check if the face(s) in the image are found.
    if results.detections:

        # Iterate over the found faces.
        for face_no, face in enumerate(results.detections):

            # Draw the face bounding box and key points on the copy of the input image.
            mp_drawing.draw_detection(image=output_image, detection=face, 
                                      keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                   thickness=-1,
                                                                                   circle_radius=image_width//115),
                                      bbox_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=image_width//180))
            
            # Retrieve the bounding box of the face.
            face_bbox = face.location_data.relative_bounding_box
            
            # Retrieve the required bounding box coordinates and scale them according to the size of original input image.
            x1 = int(face_bbox.xmin*image_width)
            y1 = int(face_bbox.ymin*image_height)

            
            # Draw a filled rectangle near the bounding box of the face.
            # We are doing it to change the background of the confidence score to make it easily visible
            cv2.rectangle(output_image, pt1=(x1, y1-image_width//20), pt2=(x1+image_width//16, y1) ,
                          color=(0, 255, 0), thickness=-1)
            
            # Write the confidence score of the face near the bounding box and on the filled rectangle. 
            cv2.putText(output_image, text=str(round(face.score[0], 1)), org=(x1, y1-25), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=image_width//700, color=(255,255,255), 
                        thickness=image_width//200)
            
    # Check if the original input image and the output image are specified to be displayed.
    if display:
        
        # Write the time take by face detection process on the output image. 
        cv2.putText(output_image, text='Time taken: '+str(round(end - start, 2))+' Seconds.', org=(10, 65),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=image_width//700, color=(0,0,255),
                    thickness=image_width//500)
        
        # Display the original input image and the output image.
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off');
        
    # Otherwise
    else:
        
        # Return the output image and results of face detection.
        return output_image, results
    
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.2, 0.2)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    img = mpDnnDetectFaces(imgS,mp,display=True)
    cv2.imshow('Webcam', img)
    cv2.waitKey(100)