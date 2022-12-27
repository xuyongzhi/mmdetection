import cv2
import numpy as np


cap= cv2.VideoCapture('rtsp://admin:123456@192.168.20.6:554/media/video1')
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import urllib3
# import numpy as np
#
# stream = urllib3.urlopen('http://192.168.20.6:80/video_feed')
# bytes = ''
# while True:
#     bytes += stream.read(1024)
#     a = bytes.find(b'\xff\xd8')
#     b = bytes.find(b'\xff\xd9')
#     if a != -1 and b != -1:
#         jpg = bytes[a:b+2]
#         bytes = bytes[b+2:]
#         img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#         cv2.imshow('Video', img)
#         if cv2.waitKey(1) == 27:
#             exit(0)