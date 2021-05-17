import cv2
import numpy as np
import glob
import os


def segment_mask_by_point(mask, list_point, org_image):
    mask_cp = np.copy(mask)
    cv2.line(mask_cp, lb_points[0], lb_points[1], 0, 2)
    cnts, hierarchy = cv2.findContours(
        mask_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hand_mask = np.zeros(mask.shape, dtype=np.uint8)
    forearm_mask = np.zeros(mask.shape, dtype=np.uint8)

    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    cnts.reverse()

    if len(cnts) == 0:
        print('no contours found!...')
        return hand_mask, forearm_mask
    if len(cnts) == 1:
        cv2.drawContours(hand_mask, cnts, 0, 255, -1)
        # return hand_mask, forearm_mask
    else:
        # lb_points[2] is target point for hand
        # check if (X,Y) inside the contour
        d = cv2.pointPolygonTest(cnts[0], lb_points[2], True)
        # print('d = ', d)
        if (d >= 0):  # lb_point2 in cnts[0] ==> cnts[0] is hand
            cv2.drawContours(hand_mask, [cnts[0]], 0, 255, -1)
            cv2.drawContours(forearm_mask, [cnts[1]], 0, 255, -1)
        else:
            cv2.drawContours(forearm_mask, [cnts[0]], 0, 255, -1)
            cv2.drawContours(hand_mask, [cnts[1]], 0, 255, -1)

    img_show = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_show[hand_mask > 100] = (0, 255, 0)
    img_show[forearm_mask > 100] = (255, 0, 0)

    # print(img.shape)
    # print(org_image.shape)

    try:
        img_show = cv2.addWeighted(img_show, 0.2, org_image, 0.8, 20)
    except Exception:
        pass

    # cv2.drawContours(img_show, cnts, 0, (0,0,255), -1)
    # cv2.drawContours(img_show, cnts, 1, (0,255,255), -1)
    # img_show = cv2.resize(img_show, (640, 480))

    return hand_mask, forearm_mask, img_show


lb_points = []
# Create the Event Capturing Function


def capture_event(event, x, y, flags, params):
    # Check if the event was left click
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(lb_points) < 3:
            lb_points.append((x, y))
        print('add point: ', (x, y), 'num: ', len(lb_points))


log_file = 'right_seg_log.txt'
choose_file = 'choosed_data_file_name/f_right_data_file_500.txt'
hand_label_mask_dir = 'right_hand_labels/'
forearm_label_mask_dir = 'right_forearm_labels/'
org_image_dir = '../Hand_data_labeled_full/'

if not os.path.isdir(hand_label_mask_dir):
    os.makedirs(hand_label_mask_dir)
if not os.path.isdir(forearm_label_mask_dir):
    os.makedirs(forearm_label_mask_dir)


window_name = 'label_here'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 600)
cv2.moveWindow(window_name, 200, 200)
cv2.setMouseCallback(window_name, capture_event)

# window_name2 = 'cut'
# cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name2, 720, 600)
# cv2.moveWindow(window_name2, 1000, 200)

# with open(choose_file, 'r') as f:
    # image_names = f.readlines()
image_names = glob.glob('data_right_hand/*.png')
# image_names = ['data_right_hand/47_017_1.png']

total = len(image_names)
i = 0
for file_name in image_names:
    print(file_name)
    # set the mouse settin function
    file_name = file_name.strip('\r\n')
    if not os.path.isfile(file_name):
        i += 1
        print('no file named %s , next' % file_name)
        print('Total %d/%d' %(i, total))
    img = cv2.imread(file_name, 0)
    mask = img > 100
    mask = 255*mask.astype('uint8')

    org_path = org_image_dir + os.path.splitext(os.path.basename(file_name))[0][:-2] + '.png'
    # print('number point: ', len(list_points_cp))
    if os.path.isfile(org_path):
        org_img = cv2.imread(org_path)
    else:
        org_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)


    while True:

        mask_color = np.zeros(org_img.shape, dtype=np.uint8)
        mask_color[img > 100] = (0, 180, 180)
        img_show = cv2.addWeighted(org_img, 0.8, mask_color, 0.2, 20)

        key = cv2.waitKey(20)
        if key == 8 or key == ord('c'):  # press backspace or 'c'
            print('remove latest click point!')
            if len(lb_points) > 0:
                lb_points.pop()
            print('lb points: ', lb_points)
        if key == ord('x'):
            print('clear lb points!')
            lb_points.clear()

        show = np.zeros(org_img.shape, dtype=np.uint8)
        if len(lb_points) >= 3:
            _, _, img_show = segment_mask_by_point(mask, lb_points, org_img)


        if len(lb_points) >= 2:
            cv2.line(img_show, lb_points[0], lb_points[1], (0, 255, 0), 5)

        for point in lb_points:
            cv2.circle(img_show, point, 7, (0, 0, 255), 12)

        cv2.imshow(window_name, img_show)

        if key == 13 or key == ord('q') or key == 9:
            if len(lb_points) < 3:
                print('not enought point for processing!...')
            else:
                print('list point choose: ', lb_points)
                hand_mask, forearm_mask, _ = segment_mask_by_point(
                    mask, lb_points, org_img)

                cv2.imwrite(hand_label_mask_dir +
                            os.path.basename(file_name), hand_mask)
                cv2.imwrite(forearm_label_mask_dir +
                            os.path.basename(file_name), forearm_mask)
                with open(log_file, 'a') as f:
                    f.write(os.path.basename(file_name) +
                            '-' + str(lb_points) + '\n')
                    f.close
                i += 1
                print('Total %d/%d' %(i, total))
                break
        if key == 27:
            i += 1
            print('skip this image')
            print('Total %d/%d' %(i, total))
            break

    lb_points.clear()

