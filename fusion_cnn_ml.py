import torch
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import os
from tqdm import tqdm
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from timm.data.transforms import LetterBoxResize
from timm.models import create_model
from timm.data import Dataset, create_loader
from timm.data.transforms_factory import get_test_transform
import glob
import cv2
import shutil
import json
import time
from eval_map import eval_map


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


def save_feature_batch(model_name, ckpt_path, feature_path, label_path, data_type="valid"):
    model = create_model(model_name.split('-')[-2], num_classes=3, checkpoint_path=ckpt_path)
    model.cuda().eval()
    print('..... Finished loading model! ......')
    img_size = int(model_name.split('-')[-1])
    interpolation = "bicubic"
    batch_size = 128

    dataset = Dataset(os.path.join(BASE, data_type))
    loader = create_loader(
        dataset,
        input_size=img_size,
        batch_size=batch_size,
        use_prefetcher=False,
        interpolation=interpolation,
        num_workers=8)

    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            out = model.forward_features(input.cuda())
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            feature = out2.view(out.size(0), -1)

            features.append(feature.cpu().numpy())
            labels.extend(target.cpu().numpy())
    features = np.array(np.vstack(features))

    pickle.dump(features, open(feature_path, 'wb'))
    pickle.dump(labels, open(label_path, 'wb'))
    print('CNN features obtained and saved.')


def save_feature_test(model_name, ckpt_path, feature_path, imgs):
    '''
    提取特征，保存为pkl文件
    '''
    model = create_model(model_name.split('-')[-2], num_classes=3, checkpoint_path=ckpt_path)
    model.eval().cuda()
    print('..... Finished loading model! ......')

    ## 特征的维度需要自己根据特定的模型调整，我这里采用的是哪一个我也忘了
    nb_features = 2048
    features = np.empty((len(imgs), nb_features))
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip().split(' ')[0]
        img = Image.open(img_path).convert('RGB')
        # bilinear bicubic
        img = get_test_transform(img_size=int(model_name.split('-')[-1]), interpolation="bicubic")(img).unsqueeze(
            0).cuda()

        with torch.no_grad():
            out = model.forward_features(img)
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            feature = out2.view(out.size(1), -1).squeeze(1)
        features[i, :] = feature.cpu().numpy()

    pickle.dump(features, open(feature_path, 'wb'))
    print('CNN features obtained and saved.')

def fusion_feature(dataset="train"):
    model_list = ["20200923-213438-hrnet_w48-288", "20200923-111323-gluon_seresnext101_32x4d-224",
                  "20200923-175500-gluon_seresnext101_64x4d-224"]
    fusion_features = []
    for ml in model_list:
        feature_path = './features/%s/%s_feature.pkl' % ("-".join(ml.split('-')[1:3]), dataset)
        fusion_features.append(pickle.load(open(feature_path, 'rb')))

    fusion_features = np.mean(fusion_features, axis=0)

    return fusion_features

def classifier_training(features, clas_fier, label_path):
    print('Pre-extracted features and labels found. Loading them ...')
    # features = pickle.load(open(feature_path, 'rb'))
    labels = pickle.load(open(label_path, 'rb'))

    if clas_fier == "LinearSVC":
        classifier = LinearSVC()  # 54.43
    elif clas_fier == "SVC":
        classifier = SVC(C=0.5, probability=True)  # 54.47
    elif clas_fier == "MLP":
        classifier = MLPClassifier()  # 54.53
    elif clas_fier == "RandomForest":
        classifier = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70,
                                            min_samples_split=5)  # 56.31
    elif clas_fier == "KNeighbors":
        classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4)  # 49.76
    elif clas_fier == "ExtraTrees":
        classifier = ExtraTreesClassifier(n_jobs=4, n_estimators=100, criterion='gini', min_samples_split=10,
                                          max_features=50, max_depth=40, min_samples_leaf=4)  # 0.9719
        # parameters = {"n_estimators": [50, 100, 200, 300, 400], "criterion": ("gini", "entropy"),
        #               "max_features": ("auto", "sqrt", "log2"), "max_depth": [20, 40, 60, 80, 100],
        #               "min_samples_split": [2, 4, 6, 8, 10, 12, 14], "min_samples_leaf": [1, 2, 3, 4, 5, 6]}
        # classifier = ExtraTreesClassifier(n_jobs=8, n_estimators=200, criterion='gini', min_samples_split=4,
        #                            max_features=50, max_depth=40, min_samples_leaf=6)  # 0.9718
        # classifier = ExtraTreesClassifier(n_jobs=8, n_estimators=200, criterion='gini', min_samples_split=6,
        #                            max_features=50, max_depth=40, min_samples_leaf=4)  # 0.9733
    else:
        classifier = GaussianNB()  # 49.35 预测概率值全1.0

    classifier.fit(features, labels)
    return classifier


def classifier_pred(classifier, shape_path, features, id, model_name, data_type="valid"):
    class_2_index = {0: 'normal', 1: 'phone', 2: 'smoke'}
    dets_info = {}

    # features = pickle.load(open(feature, 'rb'))
    ids = pickle.load(open(id, 'rb'))
    predict = classifier.predict(features)

    probs = classifier.predict_proba(features)
    prob_list = [round(prob[int(predict[i])], 4) for i, prob in enumerate(probs)]

    prediction = predict.tolist()
    total_pred_idx = [int(pred) for pred in prediction]
    total_true_idx = [int(label) for label in ids]

    with open("./txts/%s.json" % shape_path, 'r', encoding="utf-8") as f:
        shape_dict = json.load(f)

    dataset = Dataset(os.path.join(BASE, data_type))
    filenames = dataset.filenames()

    for i, filename in enumerate(filenames):
        name = filename.split('/')[-1].split('.')[0]
        dets_info[name] = [class_2_index[int(prediction[i])], prob_list[i], shape_dict[name][1], shape_dict[name][2]]

    with open("%s/%s.json" % (feature_path, shape_path.split('-')[0]), "w", encoding="utf-8") as f:
        json.dump(dets_info, f)
    accuracy = round(accuracy_score(total_true_idx, total_pred_idx), 4)

    test_map, ap_list = eval_map(detFolder="%s/v.json" % feature_path, gtFolder="txts/v-info-new.json",
                                 return_each_ap=True)
    print("Accuracy: %s, map: %.4f" % (accuracy, test_map))

    with open("weights/%s.json" % model_name, 'w', encoding="utf-8") as f:
        prob_dict = {}
        prob_dict["prob"] = probs
        prob_dict["model_weight"] = test_map
        prob_dict["label_weight"] = ap_list

        json.dump(prob_dict, f, cls=MyEncoder, indent=2)

    return accuracy, test_map


def classifier_test(model_path, feature, imgs):
    class_2_index = {0: 'normal', 1: 'calling', 2: 'smoking'}

    features = pickle.load(open(feature, 'rb'))
    classifier = joblib.load(model_path)
    predict = classifier.predict(features)

    probs = classifier.predict_proba(features)
    prob_list = [round(prob[int(predict[i])], 4) for i, prob in enumerate(probs)]
    prediction = predict.tolist()

    result_list = []
    clas_name = model_path.split('/')[-1].split('-')[0]
    with open('./infer/result-%s.json' % clas_name, 'w', encoding="utf-8") as out_file:
        for i in range(len(imgs)):
            filename = imgs[i].split('/')[-1].strip()
            name = class_2_index[int(prediction[i])]
            result_data = {"image_name": str(filename), "category": name, "score": prob_list[i]}
            result_list.append(result_data)
        json.dump(result_list, out_file, cls=MyEncoder, indent=4)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    BASE = "/home/data/classification/action/new_data/"

    # 训练好模型的保存位置
    # 20200923-213438-hrnet_w48-288
    # 20200923-111323-gluon_seresnext101_32x4d-224
    # 20200923-175500-gluon_seresnext101_64x4d-224
    model_name = "fusion"
    # save_name = '-'.join(model_name.split('-')[1:])
    # checkpoints = glob.glob('./output_new/%s' % model_name + '/*best*.pth.tar')[0]

    # #构建保存特征的文件夹
    feature_path = './features/111323-gluon_seresnext101_32x4d/'
    os.makedirs(feature_path, exist_ok=True)

    # sets = ["train", "valid", "test"]
    train_feature_path = feature_path + 'train_feature.pkl'
    train_label_path = feature_path + 'train_label.pkl'
    valid_feature_path = feature_path + 'valid_feature.pkl'
    valid_label_path = feature_path + 'valid_label.pkl'
    test_feature_path = feature_path + 'test_feature.pkl'
    # save_feature_batch(model_name, checkpoints, train_feature_path, train_label_path, "train")
    # save_feature_batch(model_name, checkpoints, valid_feature_path, valid_label_path, "valid")
    # save_feature_batch(model_name, checkpoints, test_feature_path, "test")
    train_features = fusion_feature(dataset="train")
    valid_features = fusion_feature(dataset="valid")

    # classifiers = ["SVC", "LinearSVC", "MLP", "RandomForest",
    # "KNeighbors", "ExtraTrees", "GaussianNB"]
    classifier = "MLP"
    stage = "train"
    if stage == "train":
        best_map = 0
        for i in range(150):
            save_path = feature_path + '%s.m' % classifier
            fier = classifier_training(train_features, classifier, train_label_path)

            acc, test_map = classifier_pred(fier, "v-info-new", valid_features, valid_label_path, model_name)
            # test_map = eval_map(detFolder="%s/v.json" % feature_path, gtFolder="txts/v-info-new.json")
            # print(classifier, test_map)
            test_map = round(test_map, 4)
            if test_map > best_map:
                if i != 0:
                    last_path = save_path.replace(classifier, classifier + "-%s" % best_map)
                    os.remove(last_path)
                best_map = test_map
                save_path = save_path.replace(classifier, classifier + "-%s" % test_map)
                joblib.dump(fier, save_path)
    elif stage == "valid":
        save_path = glob.glob('%s%s*.m' % (feature_path, classifier))[0]
        fier = joblib.load(save_path)

        classifier_pred(fier, "v-info-new", valid_feature_path, valid_label_path, save_name)

    elif stage == "eval":
        valid_model(model_name, checkpoints)
        test_map = eval_map(detFolder="%s/v.json" % feature_path, gtFolder="txts/v-info-new.json")
        print(test_map)
        # save_feature_batch(model_name, checkpoints, train_feature_path, train_label_path, "valid")
    else:
        save_path = glob.glob('%s%s*.m' % (feature_path, classifier))[0]
        classifier_test(save_path, test_feature_path, test_imgs)
