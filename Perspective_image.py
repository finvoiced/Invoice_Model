import cv2
import numpy as np
import Skew_correction as skew

orignal_img = cv2.imread('../Images/invoice4.jpeg')

def corner_detection(blank_with_shape): #,blank_corners =[image.shape,(0,image.shape[1]),(image.shape[0],0),(0,0)]):
    corners = cv2.goodFeaturesToTrack(blank_with_shape, 10, 0.01, 50)

    #####getting 4 extreme corners
    pts1 = []
    maxq1 = 0
    minq2 = 1
    maxq3 = 0
    maxq4 = 0

    for corner in corners:
            point = list(corner.ravel())
            # if point in blank_corners:
            #     print('popped point = ', point)
            #     continue
            pts1.append(point)
            norm_point = [(point[0]/blank_with_shape.shape[1]), (point[1]/blank_with_shape.shape[0])]
            print(norm_point, ' ',point)
            if norm_point[0] <= 0.5 and norm_point[1]<=0.5:
                print('q2')
                add_norm_point = norm_point[0]+ norm_point[1]
                if add_norm_point < minq2:
                    minq2 = add_norm_point
                    cornerq2 = point
            elif norm_point[0] >= 0.5 and norm_point[1]>=0.5:
                add_norm_point = norm_point[0] + norm_point[1]
                print('q4', add_norm_point)
                if add_norm_point >maxq4:
                    maxq4 = add_norm_point
                    cornerq4 = point
            elif norm_point[0] > 0.5 and norm_point[1] < 0.5:
                print('q1')
                subtract_norm_point = norm_point[0] - norm_point[1]
                if subtract_norm_point > maxq1:
                    maxq1 = subtract_norm_point
                    cornerq1 = point
            elif norm_point[0]<0.5 and norm_point[1] > 0.5:
                print('q3')
                subtract_norm_point = norm_point[1] - norm_point[0]
                if subtract_norm_point > maxq3:
                    maxq3 = subtract_norm_point
                    cornerq3 = point


    arranged_pts1 = [cornerq4,cornerq3,cornerq1,cornerq2]
    return arranged_pts1

def process_img_camera(orignal_img):
    cv2.imshow('orignal',orignal_img)
    print('Image Processing....',' ', 'Image information:\n')
    print('shape = ',orignal_img.shape)
    #denoising
    temp = cv2.fastNlMeansDenoising(orignal_img, h=7)
    #cv2.imshow('denoise',temp)

    img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    #removing shadows###############################
    rgb_planes = cv2.split(img)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes) #better result

    cv2.imshow('shadows_out_norm.png', result_norm)

    image = result_norm
    #################################################
    #Morphological Transform
    kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            )
    opened = image
    for i in range(10):
        closed = cv2.morphologyEx(
                    opened,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations = 1
                )
        opened = cv2.morphologyEx(
                    closed,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations = 2
                )


    #Canny edge detection
    canny = cv2.Canny(opened, 50, 100)
    cv2.imshow('canny',canny)

    image2 = image.copy()

    #Hough LinesP
    linesP = cv2.HoughLinesP(canny,5, np.pi / 180,255, minLineLength= 100, maxLineGap= 50)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if (l[0]-l[2]) == 0:
                cv2.line(image2, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.INTER_LINEAR_EXACT)
                continue
            theta = np.arctan((l[1]-l[3])/(l[0]-l[2]))
            theta = np.degrees(theta)
            bool = 10.0<np.abs(theta)<80.0

            if bool == False:
                cv2.line(image2, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.INTER_LINEAR_EXACT)

    cv2.imshow('Hough Lines', image2)

    contours, _ = cv2.findContours(image2, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    blank = np.ones_like(image)
    blank = blank*255
    blank3 = blank.copy()
    contours3, _ = cv2.findContours(blank3, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blank3,contours3 , -1, (0, 0, 0), 4)

    #sorting through contours - based on Area

    if len(contours) != 0:
        area_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        contour = area_contours[1]
        percent_area = cv2.contourArea(area_contours[1])/(img.shape[0]*img.shape[1])
        print('percent area=',percent_area)
        if percent_area < .50:
            contour = area_contours[0]
            percent_area = cv2.contourArea(area_contours[0]) / (img.shape[0] * img.shape[1])
            print('new percent area=',percent_area)
        #insert new contour finder if still less than .50. Make assumptions.
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True),True, approxCurve= 4)
        if len(approx)==4:
            cv2.drawContours(blank, [approx], 0, (0, 255, 255), 4)
        else:
            #to check if corner generated lines at ninety
            approx = cv2.approxPolyDP(approx, 0.02 * cv2.arcLength(contour, True), True, approxCurve=4)
            cv2.drawContours(blank, [approx], 0, (0, 255, 255), 4)
    cv2.imshow('rectangle',blank)

    #########drawing 4 corners
    # blank_corners = corner_detection(blank3)
    # print('corner of white image = ', blank_corners)
    arranged_pts1 = corner_detection(blank)#,blank_corners=blank_corners)
    arranged_pts1 = np.array(arranged_pts1)

    for point in arranged_pts1:
        cv2.circle(blank, (int(point[0]),int(point[1])), 20, (0, 255, 0), 3)
    cv2.imshow('detected corners circled',blank)

    ##width and Height of result
    wi = np.max([np.abs(arranged_pts1[0]- arranged_pts1[1]),np.abs(arranged_pts1[2]- arranged_pts1[3])])
    wi = np.array(wi)
    w = wi*wi
    X = int(np.sqrt(np.sum(w)))
    hi = np.max([np.abs(arranged_pts1[0]- arranged_pts1[3]),np.abs(arranged_pts1[1]- arranged_pts1[3])])
    hi = np.array(hi)
    h = hi*hi
    Y = int(np.sqrt(np.sum(h)))
    print('width,height = ',wi, hi)

    print('arranged corners= ' ,arranged_pts1)

    #Perspective Transform
    pts2 = np.float32(([X, Y], [0, Y], [X, 0], [0, 0]))
    matrix = cv2.getPerspectiveTransform(arranged_pts1, pts2)
    fit_invoice = cv2.warpPerspective(result_norm, matrix, [X,Y])

    # skew correction

    angle, rotated = skew.correct_skew(fit_invoice)
    print('skew angle=',angle)

    #######
    cv2.imshow('aligned result', rotated)
    cv2.imwrite('invoice4_processed.png', rotated)

    cv2.waitKey(0)

def process_img_scanned(orignal_img):
    cv2.imshow('orignal', orignal_img)
    print('Image Processing....', ' ', 'Image information:\n')
    print('shape = ', orignal_img.shape)
    # denoising
    temp = cv2.fastNlMeansDenoising(orignal_img, h=7)

    img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    # removing shadows###############################
    rgb_planes = cv2.split(img)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)  # better result

    cv2.imshow('shadows_out_norm.png', result_norm)

    return result_norm