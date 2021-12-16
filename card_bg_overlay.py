import sys, os, glob
import cv2
import math
import numpy as npy
from numpy.random import randint

def get_pic_list(pic_dir: str):
    # To get a list of pictures stored at pic_dir
    pic_list = []
    pic_list_temp = glob.glob(pic_dir + "*.[jpg][jpeg][png]")
    for p in pic_list_temp: 
        pic_list.append(os.path.basename(p))
    return pic_list

def gen_pic_name(pic_name: str, pic_suffix: str = ".jpg"):
    # To generate picture filename with suffix
    return pic_name + pic_suffix

def gen_rand_pic_single(origin_pic_list: list):
    # To generate the name of a random picture
    rand_id = randint(0, len(origin_pic_list))
    return origin_pic_list[rand_id]

def gen_rand_pic_list(origin_pic_list: list, pic_num: int):
    # To generate the name of a list of random pictures
    rand_pic_list = []
    for i in range(0, pic_num):
        rand_id = randint(0, len(origin_pic_list))
        rand_pic_list.append(origin_pic_list[rand_id])
    return rand_pic_list

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

def add_alpha(pic: npy.ndarray):
    # To add an alpha channel for the picture
    b_channel, g_channel, r_channel = cv2.split(pic)
    alpha_channel = npy.ones(b_channel.shape, 
                                            dtype = b_channel.dtype) * 255
    pic_with_alpha = cv2.merge((b_channel, g_channel, 
                                                r_channel, alpha_channel))
    return pic_with_alpha

def rotate_pic(pic: npy.ndarray, deg: float):
    # To rotate a picture
    if pic.shape[2] == 3: pic = add_alpha(pic)
    (pic_h, pic_w) = pic.shape[:2]
    pic_center = (pic_w // 2, pic_h // 2)
    M = cv2.getRotationMatrix2D(pic_center, deg, 1)
    cos_deg = npy.abs(M[0, 0]); sin_deg = npy.abs(M[0, 1])
    new_h = int(pic_w * sin_deg + pic_h * cos_deg)
    new_w = int(pic_h * sin_deg + pic_w * cos_deg)
    M[0, 2] += (new_w - pic_w) / 2
    M[1, 2] += (new_h - pic_h) / 2
    pic_rotated = cv2.warpAffine(pic, M, (new_w, new_h),
                           borderMode = cv2.BORDER_CONSTANT,
                           borderValue = (255, 255, 255, 0))
    return pic_rotated

def get_coord_rotated(coord: npy.ndarray, 
                                    center: npy.ndarray, deg: float):
    # To attain the coordinate of four corners of a rotated rectangle
    # Note that the rotating direction of get_coord_rotated
    # It is oppsite to rotate_pic (or cv2.getRotationMatrix2D)
    cos_d = npy.cos(deg * npy.pi / 180)
    sin_d = npy.sin(deg * npy.pi / 180)
    coord_rotated = npy.zeros(coord.shape, dtype = "float")
    coord_rotated[:, 0] = center[0] + (cos_d * (coord[:, 0] - center[0]) 
                                                        - sin_d * (coord[:, 1] - center[1]))
    coord_rotated[:, 1] = center[1] + (sin_d * (coord[:, 0] - center[0]) 
                                                        + cos_d * (coord[:, 1] - center[1]))
    return coord_rotated.astype("int")

def get_coord_translated(coord: npy.ndarray,
                                        trans_vect: npy.ndarray):
    # To attain the coordinate of four corners of a translated rectangle
    coord_translated = npy.zeros(coord.shape, dtype = "int")
    coord_translated[:, 0] = coord[:, 0] + trans_vect[0]
    coord_translated[:, 1] = coord[:, 1] + trans_vect[1]
    return coord_translated

def overlay_pic(pic: npy.ndarray, bg: npy.ndarray,
                        ol_x: int, ol_y: int, with_alpha: bool):
    # To overlay a picture on a specific background
    # Can choose whether to add an alpha channel or not
    if (pic.shape[2] == 3): pic = add_alpha(pic)
    if (bg.shape[2] == 3): bg = add_alpha(bg)
    (pic_h, pic_w) = pic.shape[:2]
    (bg_h, bg_w) = bg.shape[:2]
    ol_mx = ol_x + pic_w - 1; ol_my = ol_y + pic_h - 1
    if ((ol_x < 0) or (ol_mx >= bg_w)):
        return None
    elif ((ol_y < 0) or (ol_my >= bg_h)):
        return None
    else:
        alpha_pic = pic[0: pic_h, 0: pic_w, 3] / 255.0
        alpha_bg = 1 - alpha_pic
        for ch in range(0, 4):
            bg[ol_y: ol_my + 1, ol_x: ol_mx + 1, ch] = (
                pic[0: pic_h, 0: pic_w, ch] * alpha_pic
                + bg[ol_y: ol_my + 1, ol_x: ol_mx + 1, ch] * alpha_bg)
        if (with_alpha == True): return bg
        else: return bg[:, :, 0: 3]

def overlay_one_card_chaos(card_pic_dir: str = ".\\pics\\", 
                                card_pic_name: str = "90809975.jpg", 
                                bg_pic_dir: str = ".\\background\\", 
                                bg_pic_name: str = "ref_background_black.jpg", 
                                overlaid_pic_dir: str = ".\\overlaid_pics\\",
                                overlaid_pic_name: str = "overlaid_pic_example"):
    # To overlay a card on a specific background
    # And then save the overlaid picture
    card = get_pic(card_pic_dir, card_pic_name)
    # show_cv_pic(card, "input")
    (card_h, card_w) = card.shape[:2]
    card_minlen = min(card_h, card_w)
    card_maxlen = max(card_h, card_w)
    bg = get_pic(bg_pic_dir, bg_pic_name)
    (bg_h, bg_w) = bg.shape[:2]
    bg_minlen = min(bg_h, bg_w)
    bg_maxlen = min(bg_h, bg_w)
    if (bg_minlen < card_minlen):
        print("Failed to overlay card!"); return
    elif ((bg_minlen == card_minlen) 
          and (bg_maxlen < card_maxlen)):
        print("Failed to overlay card!"); return
    else:
        card_rotated = rotate_pic(card, randint(0, 360))
        # show_cv_pic(card_rotated, "rotated")
        overlaid_pic = None
        while (overlaid_pic is None):
            overlaid_pic = overlay_pic(card_rotated, bg, 
                                        randint(0, bg_w), randint(0, bg_h), True)
        # show_cv_pic(overlaid_pic, "overlaid")
        save_cv_pic(overlaid_pic, overlaid_pic_dir, overlaid_pic_name, ".jpg")
        print("Successfully overlay card!")

def overlay_multiple_cards_order(card_pic_dir: str = ".\\pics\\", 
                                            card_pic_name: list = ["90809975.jpg"], 
                                            bg_pic_dir: str = ".\\background\\", 
                                            bg_pic_name: str = "ref_background_black.jpg", 
                                            overlaid_pic_dir: str = ".\\overlaid_pics\\",
                                            overlaid_pic_name: str = "overlaid_pic_example",
                                            ins_h_init: int = 0, ins_w_init: int = 0, 
                                            blank_h: int = 20, blank_w: int = 20,
                                            orientation: float = 0):
    # To overlay a list of cards with a specific direction on background
    # And then save the overlaid picture
    # Therefore you can choose insertion position pic[ins_h_init][ins_w_init]
    # You can also choose blank between external rectangles of rotated cards
    # Note that the ndarray use the expression pic[h][w] to describe a pixel
    # In Cartesian coordination it stands for point (w, h)
    bg = get_pic(bg_pic_dir, bg_pic_name)
    (bg_h, bg_w) = bg.shape[:2]
    overlaid_pic = bg
    (ins_h, ins_w) = (ins_h_init + blank_h, ins_w_init + blank_w)
    (border_h, border_w) = (ins_h_init + blank_h, ins_w_init + blank_w)
    for cname in card_pic_name:
        card = get_pic(card_pic_dir, cname)
        card_rotated = rotate_pic(card, orientation)
        (card_rotated_h, card_rotated_w) = card_rotated.shape[:2]
        if (border_w + card_rotated_w + blank_w >= bg_w):
            if (border_h + card_rotated_h + blank_h >= bg_h):
                print("Failed to overlay card!"); break
            else:
                ins_h = border_h; ins_w = ins_w_init + blank_w
                overlaid_pic = overlay_pic(card_rotated, overlaid_pic, 
                                                            ins_w, ins_h, True)
                ins_w += (blank_w * 2 + card_rotated_w)
                border_h += (blank_h * 2 + card_rotated_h)
                border_w = ins_w_init + blank_w * 2 + card_rotated_w
                # show_cv_pic(overlaid_pic, "overlaid")
                print("Successfully overlay card!")
        else:
            overlaid_pic = overlay_pic(card_rotated, overlaid_pic, 
                                                            ins_w, ins_h, True)
            ins_w += (blank_w * 2 + card_rotated_w)
            border_w += (blank_w * 2 + card_rotated_w)
            border_h = max(ins_h + blank_h * 2 + card_rotated_h, border_h)
            # show_cv_pic(overlaid_pic, "overlaid")
            print("Successfully overlay card!")
    save_cv_pic(overlaid_pic, overlaid_pic_dir, overlaid_pic_name, ".jpg")

def overlay_multiple_cards_chaos(card_pic_dir: str = ".\\pics\\", 
                                            card_pic_name: list = ["90809975.jpg"], 
                                            bg_pic_dir: str = ".\\background\\", 
                                            bg_pic_name: str = "ref_background_black.jpg", 
                                            overlaid_pic_dir: str = ".\\overlaid_pics\\",
                                            overlaid_pic_name: str = "overlaid_pic_example",
                                            max_trial: int = 100):
    # To overlay a list of cards with random directions on background
    # And then save the overlaid picture
    # You can set max_trial for trial times of every card
    # This function ban overlapped cards
    # If you want overlapped cards, use overlay_multiple_cards_lunacy instead
    bg = add_alpha(get_pic(bg_pic_dir, bg_pic_name))
    (bg_h, bg_w) = bg.shape[:2]
    bg_minlen = min(bg_h, bg_w)
    bg_maxlen = min(bg_h, bg_w)
    used_area = npy.zeros(bg.shape, dtype = "uint8")
    for cpn in card_pic_name:
        card = get_pic(card_pic_dir, cpn)
        # show_cv_pic(card, "input")
        (card_h, card_w) = card.shape[:2]
        card_minlen = min(card_h, card_w)
        card_maxlen = max(card_h, card_w)
        if (bg_minlen < card_minlen):
            print("Failed to overlay card!"); return
        elif ((bg_minlen == card_minlen) and 
              (bg_maxlen < card_maxlen)):
            print("Failed to overlay card!"); return
        else:
            cnt = 0
            while (cnt <= max_trial):
                deg = randint(0, 360) # deg = 45
                ins_w = randint(0, bg_w); ins_h = randint(0, bg_h)
                origin_coord = npy.asarray([[0, 0], [card_w - 1, 0], 
                                                [card_w -1, card_h - 1], [0, card_h - 1]])
                rotated_coord = get_coord_rotated(origin_coord, 
                                                npy.asarray([card_w / 2, card_h / 2]), -deg)
                delta_w = -npy.min(rotated_coord[:, 0])
                delta_h = -npy.min(rotated_coord[:, 1])
                translated_coord = get_coord_translated(rotated_coord,
                                                    npy.asarray([ins_w + delta_w, ins_h + delta_h]))
                # translated_coord = npy.asarray([[283, 0], [693, 410], [410, 693], [0, 283]])
                if ((npy.max(translated_coord[:, 0]) >= bg_w) or
                    (npy.max(translated_coord[:, 1]) >= bg_h)): 
                    cnt += 1; continue
                mask = npy.zeros(bg.shape[: 2], dtype = "uint8")
                mask = cv2.fillPoly(mask, [translated_coord], (255, 255, 255, 255))   
                # show_cv_pic(mask, "mask")
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(used_area[:,:,3], 
                                                                                                            mask = mask)
                if (max_val > 0):
                    cnt += 1; continue
                else:
                    # masked_pic = cv2.bitwise_and(used_area, used_area, 
                    #                                                     mask = mask)
                    # show_cv_pic(masked_pic, "masked_pic")
                    card_rotated = rotate_pic(card, deg)                    
                    used_area = overlay_pic(card_rotated, used_area, 
                                                            ins_w, ins_h, True)
                    print("Successfully overlay card!")
                    # show_cv_pic(used_area, "overlaid")
                    break
            if (cnt > max_trial): 
                print("Reach max trial counts, failed to overlay card!")
                continue
    overlaid_pic = overlay_pic(used_area, bg, 0, 0, True)
    save_cv_pic(overlaid_pic, overlaid_pic_dir, overlaid_pic_name, ".jpg")

def overlay_multiple_cards_lunacy(card_pic_dir: str = ".\\pics\\", 
                                            card_pic_name: list = ["90809975.jpg"], 
                                            bg_pic_dir: str = ".\\background\\", 
                                            bg_pic_name: str = "ref_background_black.jpg", 
                                            overlaid_pic_dir: str = ".\\overlaid_pics\\",
                                            overlaid_pic_name: str = "overlaid_pic_example"):
    # To overlay a list of cards with random directions on background
    # And then save the overlaid picture
    # This function does allow overlapped cards
    bg = get_pic(bg_pic_dir, bg_pic_name)
    (bg_h, bg_w) = bg.shape[:2]
    bg_minlen = min(bg_h, bg_w)
    bg_maxlen = min(bg_h, bg_w)
    for cpn in card_pic_name:
        card = get_pic(card_pic_dir, cpn)
        # show_cv_pic(card, "input")
        (card_h, card_w) = card.shape[:2]
        card_minlen = min(card_h, card_w)
        card_maxlen = max(card_h, card_w)
        if (bg_minlen < card_minlen):
            print("Failed to overlay card!"); return
        elif ((bg_minlen == card_minlen) 
              and (bg_maxlen < card_maxlen)):
            print("Failed to overlay card!"); return
        else:
            card_rotated = rotate_pic(card, randint(0, 360))
            # show_cv_pic(card_rotated, "rotated")
            overlaid_pic = None
            while (overlaid_pic is None):
                overlaid_pic = overlay_pic(card_rotated, bg, 
                                           randint(0, bg_w), randint(0, bg_h), True)
            # show_cv_pic(overlaid_pic, "overlaid")
            bg = overlaid_pic
            print("Successfully overlay card!")
    save_cv_pic(bg, overlaid_pic_dir, overlaid_pic_name, ".jpg")

def main():
    os.chdir(sys.path[0])
    card_pic_list = get_pic_list(".\\pics\\")
    bg_pic_list = get_pic_list(".\\background\\")
    overlay_multiple_cards_order(".\\pics\\", gen_rand_pic_list(card_pic_list, 10), 
                                    ".\\background\\", gen_rand_pic_single(bg_pic_list), 
                                    ".\\overlaid_pics\\", "overlaid_pic_example", 0, 0, 20, 20, -90)
                               
if __name__ == "__main__":
    main()

"""
Example:
Overlay one card on one background randomly:
    overlay_one_card_chaos(".\\pics\\", gen_rand_pic_single(card_pic_list), 
                                    ".\\background\\", gen_rand_pic_single(bg_pic_list), 
                                    ".\\overlaid_pics\\", "overlaid_pic_example")

Overlay multiple cards on one background in order:
    overlay_multiple_cards_order(".\\pics\\", gen_rand_pic_list(card_pic_list, 10), 
                                    ".\\background\\", gen_rand_pic_single(bg_pic_list), 
                                    ".\\overlaid_pics\\", "overlaid_pic_example", 0, 0, 20, 20, 90)

Overlay multiple cards on one background randomly, overlap cards banned:
    overlay_multiple_cards_chaos(".\\pics\\", gen_rand_pic_list(card_pic_list, 10), 
                                    ".\\background\\", gen_rand_pic_single(bg_pic_list), 
                                    ".\\overlaid_pics\\", "overlaid_pic_example", 100)

Overlay multiple cards on one background randomly, overlap cards permitted:
    overlay_multiple_cards_lunacy(".\\pics\\", gen_rand_pic_list(card_pic_list, 10), 
                                    ".\\background\\", gen_rand_pic_single(bg_pic_list), 
                                    ".\\overlaid_pics\\", "overlaid_pic_example")

"""