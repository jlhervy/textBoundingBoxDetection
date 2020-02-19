import sys
import cv2
import pyclipper
import os
import shutil
from itertools import repeat
from .. import parameters
import multiprocessing as mp
import glob
import numpy as np
from ..utils.utils import del_allfile, convert_label_to_id, im_resize_and_pad
from absl import flags
from itertools import starmap
from ..data.generate_tfrecord import write_image_annotation_pairs_to_tfrecord

flags.DEFINE_string('ds', 'dataset6-noncadre6k/', 'Directory of the raw dataset ')
flags.DEFINE_string('datad', 'dataset_gen/', 'Directory where the dataset \
                                                will be generated')
flags.DEFINE_boolean('padding', True, 'whether to pad or not')
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def read_txt(file):
    with open(file,'r',encoding='utf-8') as f :
        lines = f.read()
    lines = lines.split('\n')
    gtbox =[]
    for line in lines:
        if(line==''):
            continue
        pts = line.split(',')[0:8]
        #convert str to int
        x1 = round(float(pts[0]))
        y1 = round(float(pts[1]))
        x2 = round(float(pts[2]))
        y2 = round(float(pts[3]))
        x3 = round(float(pts[4]))
        y3 = round(float(pts[5]))
        x4 = round(float(pts[6]))
        y4 = round(float(pts[7]))

        gtbox.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    return gtbox

def read_dataset():
    files = glob.glob(os.path.join(FLAGS.ds, "labels",'*.txt'))
    dataset={}
    for file in files:
        basename = '.'.join(os.path.basename(file).split('.')[:-1])
        imgname = os.path.join(FLAGS.ds, 'imgs'  ,basename+'.jpg')
        gtbox = read_txt(file)
        dataset[imgname] = gtbox
    return dataset


def cal_di(pnt, m, n):
    '''
    calculate di pixels for shrink the original polygon pnt
    Arg:
        pnt : the points of polygon [[x1,y1],[x2,y2],...]
        m : the minimal scale ration , which the value is (0,1]
        n : the number of kernel scales
    return di_n [di1,di2,...din]
    '''

    area = cv2.contourArea(pnt)
    perimeter = cv2.arcLength(pnt, True)

    ri_n = []
    for i in range(1, n):
        ri = 1.0 - (1.0 - m) * (n - i) / (n - 1)
        ri_n.append(ri)

    di_n = []
    for ri in ri_n:
        di = area * (1 - ri * ri) / perimeter
        di_n.append(di)

    return di_n


def shrink_polygon(pnt, di_n):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(pnt, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    shrink_pnt_n = []
    for di in di_n:
        shrink_pnt = pco.Execute(-int(di))
        shrink_pnt_n.append(shrink_pnt)
    return shrink_pnt_n



def gen_dataset(data):
    imgname, gtboxes = data[0]
    dst_dir = data[1]

    basename = '.'.join(os.path.basename(imgname).split('.')[:-1])
    if not(os.path.exists(os.path.join(dst_dir, basename))):
        os.mkdir(os.path.join(dst_dir, basename))
    img = cv2.imread(imgname)

    if FLAGS.padding :
        padded_img = im_resize_and_pad(img, 1000, 1000)
    else:
        padded_img = img
    cv2.imwrite(os.path.join(dst_dir, basename, basename + '.jpg'), padded_img)

    labels = np.ones((parameters.n, img.shape[0], img.shape[1], 3))
    labels = labels * 255

    gtboxes = np.array(gtboxes)
    # shrink 1.0
    for gtbox in gtboxes:
        cv2.drawContours(labels[parameters.n - 1], [gtbox], -1, (0, 0, 255), -1)

    # shrink n-1 times
    for gtbox in gtboxes:
        di_n = cal_di(gtbox, parameters.m, parameters.n)
        shrink_pnt_n = shrink_polygon(gtbox, di_n)
        for id, shrink_pnt in enumerate(shrink_pnt_n):
            cv2.drawContours(labels[id], np.array(shrink_pnt), -1, (0, 0, 255), -1)

    # convert labelimage to id
    enc = []
    for idx, label in enumerate(labels):
        npy = convert_label_to_id(parameters.label_to_id, label)
        if FLAGS.padding :
            mask_padded = im_resize_and_pad(npy*255, 500, 500, black_padding=True)
        else :
            dim_resize = (500 , 500)
            mask_padded = cv2.resize((npy*255).astype(np.uint8), dim_resize)
        cv2.imwrite(os.path.join(dst_dir, basename, "mask_" + str(idx)+ ".png"), mask_padded)



def create_dataset():
    data = read_dataset()
    train_data = {key: data[key] for i, key in enumerate(data) }

    del_allfile(FLAGS.traind)
    gen_dataset(train_data, FLAGS.datad)



if __name__ == '__main__':
    data = read_dataset()
    del_allfile(FLAGS.datad)
    print(len(data))
    with mp.Pool(processes=18) as pool :
        pool.map(gen_dataset, zip(data.items(), repeat("./dataset_gen")))

    l = glob.glob("dataset_gen/*")

    # Separate l in several fixed length lists

    nb_img_per_chunk = 6000
    chunks = [l[x:x + nb_img_per_chunk] for x in range(0, len(l), nb_img_per_chunk)]

    for chunk_nb in range(len(chunks)):
        output_name = "../dataset_chunk_" + str(chunk_nb) + ".tfrecord"
        write_image_annotation_pairs_to_tfrecord(chunks[chunk_nb], output_name)


