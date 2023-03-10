from ultralytics import YOLO


try:
    model = YOLO('best.pt')
    # predict on test video
    results = model.predict(source='test.mp4', save=True)

    # predict on live webcam
    # results = model.predict(source = '0', show =True)

except AttributeError:
    print('No such attribute')




