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
from timm.utils import MyEncoder
import glob
import cv2
import shutil
import json
import time
from eval_map import eval_map


def get_test_transform(img_size=0, interpolation="bicubic"):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        LetterBoxResize(img_size, interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])


def save_feature(model_name, ckpt_path, feature_path, label_path, imgs):
    '''
    提取特征，保存为pkl文件
    '''
    model = create_model(model_name.split('-')[-2], num_classes=3, checkpoint_path=ckpt_path)
    model.eval().cuda()
    print('..... Finished loading model! ......')

    ## 特征的维度需要自己根据特定的模型调整，我这里采用的是哪一个我也忘了
    nb_features = 2048
    features = np.empty((len(imgs), nb_features))
    labels = []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip().split(' ')[0]
        label = imgs[i].strip().split(' ')[1]
        img = Image.open(img_path).convert('RGB')
        # bilinear bicubic
        img = get_test_transform(img_size=int(model_name.split('-')[-1]), interpolation="bicubic")(img).unsqueeze(
            0).cuda()

        with torch.no_grad():
            # [1, 2048, 7, 7]
            out = model.forward_features(img)
            # [1, 2048, 1, 1]
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            # [2048]
            feature = out2.view(out.size(1), -1).squeeze(1)
        features[i, :] = feature.cpu().numpy()
        labels.append(label)

    pickle.dump(features, open(feature_path, 'wb'))
    pickle.dump(labels, open(label_path, 'wb'))
    print('CNN features obtained and saved.')


def save_feature_batch(model_name, ckpt_path, feature_path, label_path=None, data_type="valid"):

    model = create_model(model_name.split('-')[-2], num_classes=3, checkpoint_path=ckpt_path)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print('..... Finished loading model! ......')
    img_size = int(model_name.split('-')[-1])

    dataset = Dataset(os.path.join(BASE, data_type))
    loader = create_loader(
        dataset,
        input_size=img_size,
        batch_size=64,
        use_prefetcher=False,
        interpolation="bicubic",
        num_workers=8)

    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            if torch.cuda.is_available():
                input = input.cuda()
            out = model.forward_features(input)
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            feature = out2.view(out.size(0), -1)

            features.append(feature.cpu().numpy())
            labels.extend(target.cpu().numpy())

    features = np.array(np.vstack(features))

    pickle.dump(features, open(feature_path, 'wb'))
    if label_path is not None:
        pickle.dump(labels, open(label_path, 'wb'))
    print('CNN features obtained and saved.')


def save_feature_test(model_name, ckpt_path, feature_path, imgs):
    '''
    提取特征，保存为pkl文件
    '''
    model = create_model(model_name.split('-')[-2], num_classes=3, checkpoint_path=ckpt_path)
    model.eval().cuda()
    print('..... Finished loading model! ......')

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

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def valid_model(model_name, ckpt_path):

    # model = create_model(model_name.split('-')[-2], num_classes=3, checkpoint_path=ckpt_path)
    model = load_checkpoint(ckpt_path)
    model.cuda()

    img_size = int(model_name.split('-')[-1])
    interpolation = "bicubic"
    batch_size = 64

    dataset = Dataset(os.path.join(BASE, "valid"))
    # loader = create_loader(
    #     dataset,
    #     input_size=img_size,
    #     batch_size=batch_size,
    #     use_prefetcher=False,
    #     interpolation=interpolation,
    #     num_workers=4)
    loader = torch.utils.data.DataLoader(
        Dataset(root=os.path.join(BASE, "valid"),
                transform=get_test_transform(img_size)),
        batch_size=64,
        num_workers=8,
        drop_last=False,
    )
    print('..... Finished loading model! ......')
    class_2_index = {0: 'normal', 1: 'phone', 2: 'smoke'}

    with open("./txts/v-info-new.json", 'r', encoding="utf-8") as f:
        shape_dict = json.load(f)

    ## 特征的维度需要自己根据特定的模型调整，我这里采用的是哪一个我也忘了
    labels = []
    total_pred_idx = []
    dets_info = {}

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            output = model(input.cuda())

            prob = torch.max(torch.softmax(output, -1), -1)[0]
            idx = torch.max(torch.softmax(output, -1), -1)[1]
            pred_idx = idx.cpu().numpy()
            total_pred_idx.extend(pred_idx)

            for j in range(len(pred_idx)):
                filename = loader.dataset.filenames()[batch_idx * batch_size + j]
                name = filename.split('/')[-1].split('.')[0]

                dets_info[name] = [class_2_index[pred_idx[j]], float(prob[j]), shape_dict[name][1], shape_dict[name][2]]

            labels.extend(target.cpu().numpy())

    with open("%s/v.json" % (feature_path), "w", encoding="utf-8") as f:
        json.dump(dets_info, f)
    prec = accuracy_score(labels, total_pred_idx)
    print("%.4f" % prec)


def classifier_training(feature_path, clas_fier, label_path):
    print('Pre-extracted features and labels found. Loading them ...')
    features = pickle.load(open(feature_path, 'rb'))
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
        # classifier = ExtraTreesClassifier(n_jobs=4, n_estimators=100, criterion='gini', min_samples_split=10,
        #                                   max_features=50, max_depth=40, min_samples_leaf=4)  # 0.9719
        # parameters = {"n_estimators": [50, 100, 200, 300, 400], "criterion": ("gini", "entropy"),
        #               "max_features": ("auto", "sqrt", "log2"), "max_depth": [20, 40, 60, 80, 100],
        #               "min_samples_split": [2, 4, 6, 8, 10, 12, 14], "min_samples_leaf": [1, 2, 3, 4, 5, 6]}
        # classifier = ExtraTreesClassifier(n_jobs=8, n_estimators=200, criterion='gini', min_samples_split=4,
        #                            max_features=50, max_depth=40, min_samples_leaf=6)  # 0.9718
        classifier = ExtraTreesClassifier(n_jobs=8, n_estimators=200, criterion='gini', min_samples_split=6,
                                   max_features=50, max_depth=40, min_samples_leaf=4)  # 0.9733
    else:
        classifier = GaussianNB()  # 49.35 预测概率值全1.0

    # clf = GridSearchCV(classifier, parameters, n_jobs=8)
    classifier.fit(features, labels)
    return classifier


def classifier_pred(classifier, shape_path, feature, id, model_name, data_type="valid"):

    class_2_index = {0: 'normal', 1: 'phone', 2: 'smoke'}
    dets_info = {}

    features = pickle.load(open(feature, 'rb'))
    ids = pickle.load(open(id, 'rb'))
    predict = classifier.predict(features)

    # predicted_test_scores = classifier.decision_function(features)
    # probs = softmax(predicted_test_scores)
    # prob_list = [prob[int(predict[i])] for i, prob in enumerate(probs)]

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

    test_map, ap_list = eval_map(detFolder="%s/v.json" % feature_path, gtFolder="txts/v-info-new.json", return_each_ap=True)
    print("Accuracy: %s, map: %.4f" % (accuracy, test_map))

    with open("weights/%s-valid.json" % model_name, 'w', encoding="utf-8") as f:
        prob_dict = {}
        prob_dict["prob"] = probs
        prob_dict["model_weight"] = test_map
        prob_dict["label_weight"] = ap_list

        json.dump(prob_dict, f, cls=MyEncoder)

    return accuracy, round(test_map, 4)


def classifier_test(model_path, feature, data_type="test"):
    class_2_index = {0: 'normal', 1: 'calling', 2: 'smoking'}

    features = pickle.load(open(feature, 'rb'))
    classifier = joblib.load(model_path)
    predict = classifier.predict(features)

    probs = classifier.predict_proba(features)
    prob_list = [round(prob[int(predict[i])], 4) for i, prob in enumerate(probs)]
    prediction = predict.tolist()

    result_list = []
    clas_name = model_path.split('/')[-1].split('-')[0]
    dataset = Dataset(os.path.join(BASE, data_type))
    filenames = dataset.filenames()
    print(clas_name)
    with open('./infer/result-%s.json' % clas_name, 'w', encoding="utf-8") as out_file:
        for i in range(len(filenames)):
            filename = filenames[i].split('/')[-1].strip()
            name = class_2_index[int(prediction[i])]
            result_data = {"image_name": str(filename), "category": name, "score": prob_list[i]}
            result_list.append(result_data)
        json.dump(result_list, out_file, cls=MyEncoder, indent=4)


def save_test_probs(model_path, feature, save_name):

    features = pickle.load(open(feature, 'rb'))
    classifier = joblib.load(model_path)
    probs = classifier.predict_proba(features)

    with open('./weights/%s-valid.json' % save_name, 'r') as json_file:
        json_data = json.load(json_file)

    with open("weights/%s-testB.json" % save_name, 'w', encoding="utf-8") as f:
        prob_dict = {}
        prob_dict["prob"] = probs
        prob_dict["model_weight"] = json_data['model_weight']
        prob_dict["label_weight"] = json_data['label_weight']

        json.dump(prob_dict, f, cls=MyEncoder)

    print("save %s-testB probs successfully" % model_name)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    BASE = "/home/data/classification/action/new_data/"

    # 20201001-140815-hrnet_w44-288
    # 20200930-164220-ig_resnext101_32x8d-224
    # 20201007-192904-ig_resnext101_32x16d-320
    # 20201003-220359-ig_resnext101_32x16d-288
    model_list = [
        # '20201001-140815-hrnet_w44-288',
        # '20200930-164220-ig_resnext101_32x8d-224',
        '20201007-192904-ig_resnext101_32x16d-320',
        # '20201003-220359-ig_resnext101_32x16d-288'
    ]
    # model_list = ['20201012-184118-ig_seresnext101_32x8d-224']
    stage = "eval"
    for model_name in model_list:
        save_name = '-'.join(model_name.split('-')[1:])
        checkpoints = glob.glob('./output/%s' % model_name + '/*best*.pth.tar')[0]

        # #构建保存特征的文件夹
        feature_path = './features/%s/' % "-".join(model_name.split('-')[1:3])
        os.makedirs(feature_path, exist_ok=True)

        train_feature_path = feature_path + 'train_feature.pkl'
        train_label_path = feature_path + 'train_label.pkl'
        valid_feature_path = feature_path + 'valid_feature.pkl'
        valid_label_path = feature_path + 'valid_label.pkl'
        test_feature_path = feature_path + 'test_feature.pkl'
        # save_feature_batch(model_name, checkpoints, train_feature_path, train_label_path, "train")
        # save_feature_batch(model_name, checkpoints, valid_feature_path, valid_label_path, "valid")
        # save_feature_batch(model_name, checkpoints, test_feature_path, data_type="testB")

        # classifiers = ["SVC", "MLP", "RandomForest", "KNeighbors", "ExtraTrees", "GaussianNB"]
        classifier = "ExtraTrees"
        if stage == "train":
            best_map = 0
            for i in range(150):
                save_path = feature_path + '%s.m' % classifier
                fier = classifier_training(train_feature_path, classifier, train_label_path)

                acc, test_map = classifier_pred(fier, "v-info-new", valid_feature_path, valid_label_path, save_name)

                if test_map > best_map:
                    if i != 0:
                        last_path = save_path.replace(classifier, classifier+"-%s" % best_map)
                        os.remove(last_path)
                    best_map = test_map
                    save_path = save_path.replace(classifier, classifier+"-%s" % test_map)
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
            save_test_probs(save_path, test_feature_path, save_name)
            # classifier_test(save_path, test_feature_path)
