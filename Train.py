import cv2
import tesseract_OCR
import bbox


def Train(filename='Sample.jpeg',coordinates = 'yolo.txt',classes = 'classes.txt',dict={}):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    class_list,location = bbox.get_bbox(img,coordinates,classes)     #getting bounding box and respective class
    for i in range(0,len(location)):
        dict[class_list[i]]= ''
    for i in range(len(location)):
        x,y,w,h = location[i]
        #print(class_list[i],' ',img[int(y):int(y + h), int(x):int(x + w)].shape, ' ', x, ' ', w,' ',y,' ',h)
        image, line_conf,sentence= tesseract_OCR.img_data(img[int(y):int(y + h), int(x):int(x + w)])
        dict[class_list[i]] = dict[class_list[i]] +sentence + '\n'
    print(dict)
    return dict



