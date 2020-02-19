# -*- coding:utf-8 -*-
import cv2
from absl.flags import FLAGS
import time
import os
import numpy as np
import tensorflow as tf
import glob
from absl import logging
from .utils.utils import im_resize_and_pad
from .models.model import psenet_model
from .models.metrics import iou
import imgaug as ia
import imgaug.augmenters as iaa
from .pse import pse
import matplotlib.pyplot as plt
import pytesseract
import cv2
import tqdm


def show(img):
    cv2.imshow("img", img)
    cv2.waitKey()


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    logging.info('Find {} images'.format(len(files)))
    return files


def test_iou(model, folder):
    img = cv2.imread(os.path.join(folder, folder.split('/')[-1] + "2.jpg"))
    img = im_resize_and_pad(img, FLAGS.input_h, FLAGS.input_w)
    h, w, c = img.shape
    img = np.reshape(img, (1, h, w, c))

    seg_maps = model.predict(img)
    seg_5 = seg_maps[0, :, :, 5]
    show(seg_5)
    list_masks = []
    for k in range(6):
        list_masks.append(cv2.imread(os.path.join(folder, "mask_" + str(k) + ".png"), cv2.IMREAD_UNCHANGED))
    gt = np.array(list_masks)
    gt = np.transpose(gt, (1, 2, 0))
    h, w, c = gt.shape
    gt = np.reshape(gt, (1, h, w, c))
    res = iou(seg_maps, gt, 1)

    return res


def detect(seg_maps, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio=1, show=False):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    kernals = []

    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    seg_maps[:, :, 5] = np.where(seg_maps[..., 5] > 0.5, one, zero)
    for k in range(5):
        seg_maps[..., k][np.where(seg_maps[..., 5] == 0)] = 0

    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1] - 1, -1, -1):
        kernal = np.where(seg_maps[..., i] > thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh * ratio
    start = time.time()
    mask_res, label_values = pse(kernals, min_area_thresh)
    mask_res = np.array(mask_res)

    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

    img_pse = mask_res_resized.astype(np.uint8) * 5
    if show:
        for k in range(6):
            show(seg_maps[:, :, k] * 255)
        show(img_pse)
    boxes = []
    for label_value in label_values:
        points = np.argwhere(mask_res_resized == label_value)
        points = points[:, (1, 0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return np.array(boxes), kernals


def eval(model_name, input_dir, output_dir, output_text=True, read_text = False,input_h=1000, input_w=1000,
         padding=True, gpu_list="0", show=False, test_iou=False, output_feature_maps=False):
    """
    Fonction de prédiction des zones de texte, qui prend un dossier en input, puis pour chaque image calcule les
    masques des zones de texte à l'aide du réseau de neurones resnet+pyramidal, puis applique l'algorithme PSE
    pour reconstruire les zones de texte ground truth.
    :param model_name:
    :param input_dir:
    :param output_dir:
    :param output_text:
    :param input_h:
    :param input_w:
    :param padding:
    :param gpu_list:
    :param show:
    :param test_iou:
    :param output_feature_maps:
    :return:
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    dirname = os.path.dirname(__file__)
    checkpoint_path = os.path.join(dirname, "logs", model_name, "best_model.h5")

    inputs = tf.keras.layers.Input(shape=(input_h, input_w, 3))
    outputs = psenet_model(inputs)
    inference_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    inference_model.load_weights(checkpoint_path)

    if test_iou:
        res = test_iou(inference_model, "./test_data/img-26")
        print(res)
    else:
        text_idx = 0
        for img_path in tqdm.tqdm(glob.glob(os.path.join(input_dir, "*.jpg"))):
            raw_img = cv2.imread(img_path)
            ratio = 1
            if padding:
                img, orientation, amount = im_resize_and_pad(raw_img, input_h, input_w, output_amounts=True)
            else:
                img = cv2.resize(raw_img.astype(np.uint8), (input_h, input_w))
            h, w, c = img.shape
            img = np.reshape(img, (1, h, w, c))
            res = inference_model.predict(img)
            if output_feature_maps:
                # Utilisé pour montrer certaines feature maps
                inputs = tf.keras.layers.Input(shape=(input_h, input_w, 3))
                outputs = psenet_model(inputs)
                temp_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
                temp_model.load_weights(checkpoint_path)
                temp_model.summary()
                last = temp_model.get_layer("conv2d_17").output
                F_beforelast = temp_model.get_layer("conv2d_16").output
                b5 = temp_model.get_layer("conv5_block3_out").output
                p5 = temp_model.get_layer("conv2d_9").output

                b4 = temp_model.get_layer("conv4_block5_out").output
                p4 = temp_model.get_layer("conv2d_12").output

                b3 = temp_model.get_layer("conv3_block3_out").output
                p3 = temp_model.get_layer("conv2d_13").output

                b2 = temp_model.get_layer("conv2_block2_out").output
                p2 = temp_model.get_layer("conv2d_14").output

                b1 = temp_model.get_layer("conv1_relu").output
                p1 = temp_model.get_layer("conv2d_15").output

                end_points = [last, F_beforelast, b5, p5, b4, p4, b3, p3, b2, p2, b1, p1]
                for i, ep in enumerate(end_points):
                    visualization_model = tf.keras.models.Model(inputs=inputs, outputs=ep)

                    feature_maps = visualization_model.predict(img)

                    # plot all 64 maps in an 8x8 squares

                    if i == 0:
                        square = 3
                        minusligns = 1
                    else:
                        square = 16
                        minusligns = 0
                    ix = 1
                    for _ in range(square):
                        for _ in range(square - minusligns):
                            # specify subplot and turn of axis
                            ax = plt.subplot(square - minusligns, square, ix)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # plot filter channel in grayscale
                            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
                            ix += 1
                    # show the figure
                    print(i)
                    plt.show()

            boxes, kernels = detect(seg_maps=res, image_w=w, image_h=h, show=show)
            # show_score_geo(img, kernels, img)
            if boxes is not None:
                boxes = boxes.reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio
                boxes[:, :, 1] /= ratio
                boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
                boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)
                # res_file = os.path.join(
                #     output_dir, "res01.txt")

                img = img[0, :, :]
                im_save = np.copy(img)
                list_polygons = []
                for i in range(len(boxes)):
                    # to avoid submitting errors
                    box = boxes[i]
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    # if np.linalg.norm(box[0] - box[1]) >200 or np.linalg.norm(box[3] - box[0]) >250 :
                    #     continue
                    list_polygons.append(ia.Polygon(box))
                    # cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)

                psoi = ia.PolygonsOnImage(list_polygons, shape=img.shape)
                if padding:
                    if orientation == "h":
                        aug_pipeline = iaa.Sequential([
                            iaa.Crop(
                                px=(0, amount[1], 0, amount[0]),
                                keep_size=False
                            ),
                            iaa.Resize({'height': raw_img.shape[0], 'width': raw_img.shape[1]}),
                        ])
                    else:
                        aug_pipeline = iaa.Sequential([
                            iaa.Crop(
                                px=(amount[0], 0, amount[1], 0),
                                keep_size=False
                            ),
                            iaa.Resize({'height': raw_img.shape[0], 'width': raw_img.shape[1]}),
                        ])

                else:
                    aug_pipeline = iaa.Sequential([
                        iaa.Resize({'height': raw_img.shape[0], 'width': raw_img.shape[1]}),
                    ])

                #Resize l'image de taille input_h*input_w et ses polygones détectés à la taille de l'image originale
                img_aug, psoi_aug = aug_pipeline(image=im_save, polygons=psoi)
                psoi_aug = psoi_aug.remove_out_of_image().clip_out_of_image()
                img_output_path = os.path.join(output_dir, img_path.split('/')[-1].split(".")[0])
                text_output_path = os.path.join(img_output_path, "texte")
                if not (os.path.isdir(img_output_path)):
                    os.makedirs(img_output_path)
                if not (os.path.isdir(text_output_path)):
                    os.makedirs(text_output_path)
                if output_text:
                    for poly in psoi_aug.polygons:
                        x_min = min(poly.xx_int)
                        x_max = max(poly.xx_int)
                        y_min = min(poly.yy_int)
                        y_max = max(poly.yy_int)
                        extracted_rectangle = raw_img[y_min:y_max, x_min:x_max]
                        custom_oem_psm_config = r'--oem 1 --psm 12'
                        if read_text :
                            img_name = str(text_idx)+ "~"+ str(pytesseract.image_to_string(extracted_rectangle, config=custom_oem_psm_config,
                                                                              lang='fra'))
                        else :
                            img_name = "img~" +str(text_idx)
                        cv2.imwrite(os.path.join(text_output_path,  img_name + ".png"),
                                        extracted_rectangle)
                        text_idx += 1
                img_aug = psoi_aug.draw_on_image(raw_img, alpha_points=0, alpha_face=0.02, color_lines=(255, 0, 0),
                                                 size_lines=2)
                # cv2.imwrite(img_output_path, img)
                cv2.imwrite(os.path.join(img_output_path, img_path.split('/')[-1]), img_aug)


if __name__ == '__main__':
    # TODO
    print("TODO")
