import cv2
from timm.data import Dataset
from timm.data.transforms import LetterBoxResize
import json
import os
import shutil
from PIL import Image
from eval_map import eval_map
import logging

def save_info():
    dataset_eval = Dataset("/home/data/classification/action/datav3/valid")

    gts_info = {}
    for filename in dataset_eval.filenames():
        h, w, _ = cv2.imread(filename).shape
        target_class = filename.split('/')[-2]

        file_name = filename.split('/')[-1].split('.')[0]
        gts_info[file_name] = [target_class, w, h]
    with open("./txts/v3-info-valid.json", "w", encoding="utf-8") as f:
        json.dump(gts_info, f)


def remove_redundant_ckpt(ckpt_path):
    for root, dirs, files in os.walk(ckpt_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".pth.tar"):
                if file != "model_best.pth.tar":
                    os.remove(os.path.join(root, file))


def save_padding_image(raw_img_path, pad_img_path):
    for root, dirs, files in os.walk(raw_img_path):

        for f in files:
            if f.endswith("csv") or f.endswith("txt"):
                continue

            if root.split('/')[-2] == "valid":
                raw_path = os.path.join(root, f)

                img = Image.open(raw_path).convert("RGB")
                img = LetterBoxResize(224, interpolation="bicubic")(img)

                dst_path = os.path.join(pad_img_path, root.split('/')[-2], root.split('/')[-1])

                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)

                img.save(os.path.join(dst_path, f))


def find_img():
    label_dict1 = {'normal': [], 'calling': [], 'smoking': []}
    label_dict2 = {'normal': [], 'calling': [], 'smoking': []}

    res1 = r'fusions_svm/result_C_testB-1.json'
    res2 = r'fusions_svm/result_C_testB-2.json'

    fv1 = open("fusions_svm/v1.json", "w", encoding="utf-8")
    fv2 = open("fusions_svm/v2.json", "w", encoding="utf-8")
    dets_info1 = {}
    dets_info2 = {}

    f1 = open(res1, 'r', encoding="utf-8")
    data1 = json.load(f1)
    for v1 in data1:
        label_dict1[v1['category']].append(v1['image_name'])

    f2 = open(res2, 'r', encoding="utf-8")
    data2 = json.load(f2)
    for v2 in data2:
        label_dict2[v2['category']].append(v2['image_name'])

    diff_sm = set(label_dict1['smoking']) ^ set(label_dict2['smoking'])
    print(len(diff_sm))
    diff_nor = set(label_dict1['normal']) ^ set(label_dict2['normal'])
    print(len(diff_nor))
    diff_ca = set(label_dict1['calling']) ^ set(label_dict2['calling'])
    print(len(diff_ca))

    dst_root = './fusions_svm/diff'

    with open("./txts/t-info-diff.json", 'r', encoding="utf-8") as f:
        shape_dict = json.load(f)

    for da in diff_sm:
        src_path = os.path.join('/home/data/classification/action/new_data/testB/', da)
        name = da.split('.')[0]

        if da in label_dict1["smoking"]:
            dst_path = os.path.join(dst_root, "res1/smoking")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(src_path, os.path.join(dst_path, da))

            for da1 in data1:
                if da1['image_name'] == da:
                    dets_info1[name] = ["smoke", float(da1['score']), shape_dict[name][1], shape_dict[name][2]]

        if da in label_dict2["smoking"]:
            dst_path = os.path.join(dst_root, "res2/smoking")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(src_path, os.path.join(dst_path, da))

            for da2 in data2:
                if da2['image_name'] == da:
                    dets_info2[name] = ["smoke", float(da2['score']), shape_dict[name][1], shape_dict[name][2]]

        shutil.copy(src_path, os.path.join(dst_root, 'total', da))


    for da in diff_ca:
        src_path = os.path.join('/home/data/classification/action/new_data/testB/', da)
        name = da.split('.')[0]

        if da in label_dict1["calling"]:
            dst_path = os.path.join(dst_root, "res1/calling")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(src_path, os.path.join(dst_path, da))

            for da1 in data1:
                if da1['image_name'] == da:
                    dets_info1[name] = ["phone", float(da1['score']), shape_dict[name][1], shape_dict[name][2]]

        if da in label_dict2["calling"]:
            dst_path = os.path.join(dst_root, "res2/calling")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(src_path, os.path.join(dst_path, da))

            for da2 in data2:
                if da2['image_name'] == da:
                    dets_info2[name] = ["phone", float(da2['score']), shape_dict[name][1], shape_dict[name][2]]

        shutil.copy(src_path, os.path.join(dst_root, 'total', da))

    for da in diff_nor:
        src_path = os.path.join('/home/data/classification/action/new_data/testB/', da)
        name = da.split('.')[0]

        if da in label_dict1["normal"]:
            dst_path = os.path.join(dst_root, "res1/normal")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(src_path, os.path.join(dst_path, da))

            for da1 in data1:
                if da1['image_name'] == da:
                    dets_info1[name] = ["normal", float(da1['score']), shape_dict[name][1], shape_dict[name][2]]

        if da in label_dict2["normal"]:
            dst_path = os.path.join(dst_root, "res2/normal")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(src_path, os.path.join(dst_path, da))

            for da2 in data2:
                if da2['image_name'] == da:
                    dets_info2[name] = ["normal", float(da2['score']), shape_dict[name][1], shape_dict[name][2]]
        shutil.copy(src_path, os.path.join(dst_root, 'total', da))

    # dets_info1 = sorted(dets_info1, key=lambda k:k[0], reverse=True)
    json.dump(dets_info1, fv1)
    json.dump(dets_info2, fv2)


def modify_probs(json_path):
    dets_info = {}
    with open("./txts/v2-info-test.json", 'r', encoding="utf-8") as f:
        shape_dict = json.load(f)

    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

        for data in json_data:
            data['score'] = data['score'] * 1.1
            if data['score'] > 1.0:
                data['score'] = 1.0

            name = data['image_name'].split('.')[0]
            dets_info[name] = [data['category'], float(data['score']), shape_dict[name][1], shape_dict[name][2]]

    with open('infer/result-modify.json', 'w', encoding="utf-8") as out_file:
        json.dump(json_data, out_file, indent=4)

    with open("infer/v2-modify.json", "w", encoding="utf-8") as f:
        json.dump(dets_info, f, indent=4)

    map, each_ap = eval_map(detFolder="infer/v2-modify.json",
                            gtFolder="txts/v2-info-test.json", return_each_ap=True)
    print('Valid mAP: {}, each ap: {}'.format(round(map, 4), each_ap))


if __name__ == "__main__":
    # save_info()
    modify_probs("infer/result.json")
    # find_img()
    # save_padding_image("/home/data/classification/action/new_data/", "/home/data/classification/action/new_data_pad/")
# remove_redundant_ckpt("./output_new")
