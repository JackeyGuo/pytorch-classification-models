import argparse
from predictor import *
from timm.data.dataset import Dataset, TestDataset
from eval_map import eval_map
import time
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--flag', default="testB", type=str, metavar='FLAG',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root-path', default='./fusions_svm', metavar='DIR',
                    help='path to root')

# noinspection PyTypeChecker
def dump_csv(predictor):
    results = []
    n_predictors = len(predictor.index2combine_name)
    path_json_dumps = ['%s/%s/result_%s_%s.json' % (args.root_path, predictor.index2combine_name[i],
                                                   predictor.index2policy[i], args.flag) for i in
                       range(n_predictors)]
    print('Start eval predictor...')
    predictions = predictor.fusion_prediction(top=1, return_with_prob=True)

    if len(results) == 0:
        for i in range(len(predictions)):
            results.append([])
    for index, prediction in enumerate(predictions):
        results[index] = prediction
    assert len(results) == len(path_json_dumps), 'The result length is not equal with path_json_dumps\'s.'

    for result, save_path in zip(results, path_json_dumps):
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if args.flag == 'valid':
            class_2_index = {0: 'normal', 1: 'phone', 2: 'smoke'}
            dets_info = {}
            pred_idx = []
            true_idx = []
            with open("./txts/v-info-new.json", 'r', encoding="utf-8") as f:
                shape_dict = json.load(f)
            for i, res in enumerate(result):
                name = dataset.filenames()[i].split('/')[-1].split('.')[0]
                dets_info[name] = [class_2_index[res[0][0]], float(res[0][1]), shape_dict[name][1],
                                   shape_dict[name][2]]
                pred_idx.append(res[0][0])

                target_class = dataset.filenames()[i].split('/')[-2]
                target_idx = list(class_2_index.keys())[list(class_2_index.values()).index(target_class)]

                true_idx.append(target_idx)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(dets_info, f)

            acc = accuracy_score(y_true=true_idx, y_pred=pred_idx)
            test_map = eval_map(detFolder=save_path, gtFolder="txts/v-info-new.json")
            print("acc score: %.4f, map score: %.4f" % (acc, test_map))
        else:
            import shutil
            dst_root = './infer/fusion1'
            if not os.path.exists(dst_root):
                os.makedirs(dst_root)

            class_2_index = {0: 'normal', 1: 'calling', 2: 'smoking'}
            result_list = []
            with open(save_path, 'w', encoding="utf-8") as out_file:
                filenames = dataset.filenames()
                for i, res in enumerate(result):
                    filename = filenames[i].split('/')[-1].strip()
                    name = class_2_index[res[0][0]]
                    result_data = {"image_name": str(filename), "category": name, "score": float(res[0][1])}
                    result_list.append(result_data)

                    # dst_path = os.path.join(dst_root, name)
                    # if not os.path.exists(dst_path):
                    #     os.makedirs(dst_path)
                    # shutil.copy(filenames[i], os.path.join(dst_path, filenames[i].split('/')[-1]))
                json.dump(result_list, out_file, indent=4)

        print('Dump %s finished.' % save_path)


if __name__ == '__main__':
    st = time.time()

    args = parser.parse_args()

    model_weight, label_weight, weights = {}, {}, {}
    predictions_dict = {}
    output_root = r'./output'
    data_root = r'/home/data/classification/action/new_data'
    model_list = []

    checkpoint_list = [
        # '20201003-220359-ig_resnext101_32x16d-288',
        '20201001-140815-hrnet_w44-288',
        '20200930-164220-ig_resnext101_32x8d-224',
        '20201007-192904-ig_resnext101_32x16d-320',
        # '20201012-184118-ig_seresnext101_32x8d-224',
        # '20200930-164220-ig_resnext101_32x8d-224'
    ]

    for checkpoint in checkpoint_list:
        name = '-'.join(checkpoint.split('-')[1:])
        model_list.append(name)

    for index, model_name in enumerate(model_list):
        img_size = int(model_name.split('-')[-1])

        dataset = Dataset(os.path.join(data_root, args.flag))

        with open('./weights/%s-%s.json' % (model_name, args.flag), 'r') as json_file:
            json_data = json.load(json_file)

            model_weight[model_name] = np.array([json_data['model_weight']])
            label_weight[model_name] = np.array([json_data['label_weight']])

            prediction_output = np.array(json_data["prob"])

        predictions_dict.update({model_name: prediction_output})

    if args.flag == 'valid':
        # ['A', 'B', 'C', 'D', 'E', 'P', 'M', 'MM', 'ML']
        INTEGRATED_POLICY = ['B', 'C', 'D']
    else:
        INTEGRATED_POLICY = ['C']
    predictor = IntegratedPredictor(model_list, [predictions_dict, model_weight, label_weight], args,
                                    policies=INTEGRATED_POLICY, all_combine=False)
    dump_csv(predictor)

    print("total time: %.4f" % (time.time()-st))