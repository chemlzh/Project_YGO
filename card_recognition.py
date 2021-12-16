import sys, os
import cv2, imutils
import math
import numpy as npy
from numpy.core.numeric import cross

def get_pic_list(pic_dir: str):
    # To get a list of pictures stored at pic_dir
    pic_list = []
    pic_list_temp = glob.glob(pic_dir + "*.[jpg][jpeg][png]")
    for p in pic_list_temp: 
        pic_list.append(os.path.basename(p))
    return pic_list

def get_pic(pic_dir: str, pic_file: str):
    # To read and attain the picture suitable for openCV
    # Note that pic_file must have a suffix like '.jpg', '.png', etc.
    if os.path.isfile(pic_dir + pic_file) == True:
        return cv2.imread(pic_dir + pic_file, 
                                        cv2.IMREAD_UNCHANGED)
    else: return None

def show_cv_pic(pic: npy.ndarray, pic_name: str):
    # To show a picture in window
    cv2.imshow(pic_name, pic); cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_cv_pic(pic: npy.ndarray, pic_dir: str, 
                        pic_name: str, suffix: str = ".jpg"):
    # To save a picture at pic_dir with the name pic_name
    cv2.imwrite(pic_dir + pic_name + suffix, pic, 
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

def get_corner_region(rect: npy.ndarray, scale: float = 6):
    # To get four corner regions of a rectangle
    hex_w = npy.zeros(4, dtype = "int")
    hex_h = npy.zeros(4, dtype = "int")
    coord_hex = npy.zeros((4, 4, 2), dtype = "int")
    for i in range(0, 4):
        hex_w[i] = round((rect[(i + 1) % 4][0] - rect[i][0]) / scale)
        hex_h[i] = round((rect[(i + 1) % 4][1] - rect[i][1]) / scale)
    for i in range(0, 4):
        coord_hex[i][0] = rect[i]
        coord_hex[i][1] = [rect[i][0] + hex_w[i], 
                                        rect[i][1] + hex_h[i]]
        coord_hex[i][2] = [rect[i][0] + hex_w[i] - hex_w[i - 1], 
                                        rect[i][1] + hex_h[i] - hex_h[i - 1]]
        coord_hex[i][3] = [rect[i][0] - hex_w[i - 1], 
                                        rect[i][1] - hex_h[i - 1]]
    return coord_hex

def card_recognition(pic: npy.ndarray, 
            recog_pic_dir: str = ".\\recognized_pics\\",
            recog_pic_batch_name: str = "recognized",
            recog_res_pic_dir: str = ".\\recognition_result_pics\\",
            recog_res_pic_name: str = "recognition_result"):
    # To detect and recognize Yugioh card
    
    # Initialization for standard Yugioh card
    std_h = 580; std_w = 400
    std_contour_area = std_h * std_w

    # Calculate red-green (RG) and yellow-blue (YB) part of the picture
    (pic_B, pic_G, pic_R) = cv2.split(pic.astype("float"))
    RG = npy.absolute(pic_R - pic_G)
    YB = npy.absolute(0.5 * (pic_R + pic_G) - pic_B)

    # Gaussian blur and gray scale
    gaussian_blur_pic = cv2.GaussianBlur(pic, (3, 3), 0)
    # show_cv_pic(gaussian_blur_pic, "gaussian_blur")
    gray_pic = cv2.cvtColor(gaussian_blur_pic, cv2.COLOR_RGB2GRAY)
    # show_cv_pic(gray_pic, "gray_scale")

    # Canny border detection
    canny_pic = cv2.Canny(gray_pic, 50, 125)
    # canny_pic = cv2.Canny(gaussian_blur_pic, 50, 125)
    # show_cv_pic(canny_pic, "canny")

    # Binarization
    ret, binary_pic = cv2.threshold(canny_pic, 127, 255, 
                                                        cv2.THRESH_BINARY)
    # show_cv_pic(binary_pic, "binarization")

    # Dilation and erosion
    kernel = npy.ones((5, 5), npy.uint8)
    dilated_pic = cv2.dilate(binary_pic, kernel, iterations = 3)
    # show_cv_pic(dilated_pic, "dilation")
    eroded_pic = cv2.erode(dilated_pic, kernel, iterations = 3)
    # show_cv_pic(eroded_pic, "eroding")

    # Generate contours
    contours = cv2.findContours(eroded_pic, cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, 
                                    reverse = True)
    
    # Generate candidate rectangle
    candidate_rect = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        if ((len(approx) == 4) and (cv2.contourArea(c) 
                                                >= 0.75 * std_contour_area)): 
            candidate_rect.append(approx)
    
    recog_res_pic = pic.copy()
    # Extract four corner regions and calculate color info
    for cnt in range(0, len(candidate_rect)):
        cr = candidate_rect[cnt]
        cv2.polylines(recog_res_pic , [cr.astype("int")], True, (0, 255, 0), 5)
        # print(cr[0][0], cr[1][0], cr[2][0], cr[3][0])
        coord_corner = get_corner_region(cr[:, 0], 6)
        color_coord = npy.zeros((4, 7), dtype = "float32")
        for i in range(0, 4):
            # Generate mask
            mask = npy.zeros(pic.shape[: 2], dtype = "uint8")
            mask = cv2.fillPoly(mask, [coord_corner[i]], (255, 255, 255))
            # show_cv_pic(mask, "mask")
            # masked_pic = cv2.bitwise_and(pic, pic, mask = mask)
            # show_cv_pic(masked_pic, "masked_pic")

            # Color info includes mean and deviation of R, G, B 
            (mean, std) = cv2.meanStdDev(pic, mask = mask)

            # In addition, colorfulness is also calculated
            (mean_RG, std_RG) = cv2.meanStdDev(RG, mask = mask)
            (mean_YB, std_YB) = cv2.meanStdDev(YB, mask = mask)
            mean_root = npy.sqrt((mean_RG ** 2) + (mean_YB ** 2))
            std_root = npy.sqrt((std_RG ** 2) + (std_YB ** 2))
            colorfulness = std_root + (0.3 * mean_root)
            # print((mean, std))
            
            # Generate color coordination (color vector)
            color_coord[i] = npy.concatenate((mean, std, 
                                              colorfulness), axis = 0).T

        # Search two corner with closest color coordination
        min_color_dis = float("inf")
        min_side_dis = float("inf")
        min_pos = -1
        for i in range(0, 4):
            color_dis = npy.linalg.norm(color_coord[(i + 1) % 4] - 
                                                            color_coord[i])
            side_dis = npy.linalg.norm(cr[:, 0][(i + 1) % 4] - cr[:, 0][i])
            if (color_dis < min_color_dis):
                if ((side_dis < min_side_dis) or math.isclose(side_dis, rel_tol = 5e-3)):
                    min_color_dis = color_dis
                    # min_side_dis = side_dis
                    min_pos = i

        # To distinguish left-up corner and right-up corner
        # The initial program use RGB deviation and colorfulness

        # color_dev1 = npy.linalg.norm(color_coord[min_pos - 1][3: 6])
        # color_dev2 = npy.linalg.norm(color_coord[(min_pos + 2) % 4][3: 6])
        # color_dev1 = color_coord[min_pos - 1][6]
        # color_dev2 = color_coord[(min_pos + 2) % 4][6]

        # Now the program use cross product of mid-point vector and up-side vector
        midp_down = (cr[:, 0][min_pos] + cr[:, 0][(min_pos + 1) % 4]) / 2
        midp_up =(cr[:, 0][min_pos - 1] + cr[:, 0][(min_pos + 2) % 4]) / 2
        vec_midp = midp_up - midp_down
        vec1 = cr[:, 0][min_pos - 1] - midp_up
        vec2 = cr[:, 0][(min_pos + 2) % 4] - midp_up

        # Attain accurate relation between coordination and corner
        # if (color_dev1 > color_dev2):
        if (npy.cross(vec_midp, vec1) > npy.cross(vec_midp, vec2)):
            cal_rect = npy.float32([cr[:, 0][min_pos - 2], cr[:, 0][min_pos - 1],
                                                cr[:, 0][min_pos], cr[:, 0][(min_pos + 1) % 4]])
        else:
            cal_rect = npy.float32([cr[:, 0][(min_pos + 3) % 4], cr[:, 0][(min_pos + 2) % 4],
                                                cr[:, 0][(min_pos + 1) % 4], cr[:, 0][min_pos]])
        des_rect = npy.float32([[0, 0], [std_w - 1, 0],
                                        [std_w - 1, std_h - 1], [0, std_h - 1]])

        # Perspective Transform
        M = cv2.getPerspectiveTransform(cal_rect, des_rect)
        des = cv2.warpPerspective(pic, M, (std_w, std_h))
        # show_cv_pic(des, "perspective_transform")

        save_cv_pic(des, recog_pic_dir, recog_pic_batch_name 
                                + "_" + str(cnt), ".jpg")

    save_cv_pic(recog_res_pic, recog_res_pic_dir, 
                            recog_res_pic_name, ".jpg")

def main():
    os.chdir(sys.path[0])
    overlaid_pic_dir = ".\\overlaid_pics\\"
    overlaid_pic_name = "overlaid_pic_example"
    overlaid_pic = get_pic(overlaid_pic_dir, overlaid_pic_name + ".jpg")
    card_recognition(overlaid_pic, ".\\recognized_pics\\", 
                                    "recognized", ".\\recognition_result_pics\\", 
                                    "recognition_result")

if __name__ == "__main__":
    main()