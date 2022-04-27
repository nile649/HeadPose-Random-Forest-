import cv2
import mediapipe as mp

class FaceDetector:
  def __init__(self):
    self.mp_face_detection = mp.solutions.face_detection
  def __call__(self,image):
      # For static images:
      with self.mp_face_detection.FaceDetection(
          model_selection=1, min_detection_confidence=0.5) as face_detection:
      #     image = cv2.imread(file)
          # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
          results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

          annotated_image = image.copy()
          for face_no, face in enumerate(results.detections):
              face_data = face.location_data
          data = face_data.relative_bounding_box
          if len(image.shape)==3:
              h, w, c = image.shape
          else:
              h,w = image.shape
          xleft = data.xmin*w
          xleft = int(xleft)
          xtop = data.ymin*h
          xtop = int(xtop)
          xright = data.width*w + xleft
          xright = int(xright)
          xbottom = data.height*h + xtop
          xbottom = int(xbottom)
          if len(image.shape)==3:
              return image[xtop:xbottom,xleft:xright,:]
          else:
              return image[xtop:xbottom,xleft:xright]
