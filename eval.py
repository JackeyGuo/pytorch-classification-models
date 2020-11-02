import argparse
from timm.models import create_model
from timm.data import resolve_data_config
from predictor import *
from timm.data.dataset import Dataset, TestDataset
from timm.data.loader import create_loader
from timm.utils import MyEncoder
import glob
from eval_map import eval_map
import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

parser.add_argument('--flag', default="valid", type=str, metavar='FLAG',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--num-classes', type=int, default=3,
                    help='Number classes in dataset')
parser.add_argument('--no-prefetcher', action='store_true', default=True,
                    help='disable fast prefetcher')
parser.add_argument('--root-path', default='./fusions', metavar='DIR',
                    help='path to root')

# noinspection PyTypeChecker
def dump_csv(predictor, true_output):
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
            with open("./txts/v-info-new.json", 'r', encoding="utf-8") as f:
                shape_dict = json.load(f)
            for i, res in enumerate(result):
                name = dataset.filenames()[i].split('/')[-1].split('.')[0]
                dets_info[name] = [class_2_index[res[0][0]], float(res[0][1]), shape_dict[name][1],
                                   shape_dict[name][2]]
                pred_idx.append(res[0][0])
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(dets_info, f)

            acc = accuracy_score(y_true=true_output, y_pred=pred_idx)
            test_map = eval_map(detFolder=save_path, gtFolder="txts/v-info-new.json")
            print("acc score: %.4f, map score: %.4f" % (acc, test_map))
        else:
            class_2_index = {0: 'normal', 1: 'calling', 2: 'smoking'}
            result_list = []
            with open(save_path, 'w', encoding="utf-8") as out_file:
                filenames = dataset.filenames()
                for i, res in enumerate(result):
                    filename = filenames[i].split('/')[-1].strip()
                    name = class_2_index[res[0][0]]
                    result_data = {"image_name": str(filename), "category": name, "score": float(res[0][1])}
                    result_list.append(result_data)
                json.dump(result_list, out_file)
        print('Dump %s finished.' % save_path)


def perform_predict(predictor, loader, model_weight, label_weight, weights, save_weights=True):
    temp_weight = {}
    total_true_output = []
    total_pred_output = []
    total_pred_idx = []
    total_true_idx = []
    right_count = 0
    n_labels = np.zeros((3,)) + 1e-5
    n_right_labels = np.zeros((3,))
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(loader)):
            output = predictor(input.cuda())
            output_data = output.cpu().numpy()
            if save_weights:
                idx = torch.max(F.softmax(output, -1), -1)[1]
                target_idx = target.cpu().numpy()
                predict_idx = idx.cpu().numpy()

                total_pred_idx.extend(predict_idx)
                total_true_idx.extend(target_idx)
                for j in range(len(target_idx)):
                    # 统计预测中预测对的数量，相当于precision
                    n_labels[predict_idx[j]] += 1
                    # 统计真实中预测对的数量，相当于recall
                    # n_labels[target_idx[j]] += 1

                    if predict_idx[j] == target_idx[j]:
                        right_count += 1
                        n_right_labels[predict_idx[j]] += 1
                    total_true_output.append(target_idx[j])
                    total_pred_output.append(output_data[j])
            else:
                total_pred_output.extend(output_data)

    if save_weights:
        with open("./txts/v-info-new.json", 'r', encoding="utf-8") as f:
            shape_dict = json.load(f)

        filenames = dataset.filenames()
        dets_info = {}
        class_2_index = {0: 'normal', 1: 'phone', 2: 'smoke'}

        predictions = softmax(np.array(total_pred_output))
        probs = np.max(predictions, axis=-1)
        for i, filename in enumerate(filenames):
            name = filename.split('/')[-1].split('.')[0]
            dets_info[name] = [class_2_index[int(total_pred_idx[i])], probs[i], shape_dict[name][1],
                               shape_dict[name][2]]

        with open("fusions/fv.json", "w", encoding="utf-8") as f:
            json.dump(dets_info, f, cls=MyEncoder)
        accuracy = round(accuracy_score(total_true_idx, total_pred_idx), 4)

        test_map, ap_list = eval_map(detFolder="fusions/fv.json", gtFolder="txts/v-info-new.json",
                                     return_each_ap=True)
        print("Accuracy: %s, map: %.4f" % (accuracy, test_map))

        with open("weights/fusion_weights_map.json", 'w', encoding="utf-8") as f:
            weights[predictor.default_cfg['model_name']] = {}
            # weights[predictor.default_cfg['model_name']]["prob"] = predictions
            weights[predictor.default_cfg['model_name']]["model_weight"] = test_map
            weights[predictor.default_cfg['model_name']]["label_weight"] = ap_list

            json.dump(weights, f, cls=MyEncoder)

        model_weight[predictor.default_cfg['model_name']] = test_map
        label_weight[predictor.default_cfg['model_name']] = ap_list
    else:
        with open('./weights/fusion_weights_map.json', 'r') as json_file:
            json_data = json.load(json_file)
        model_weight[predictor.default_cfg['model_name']] = np.array(
            [json_data[predictor.default_cfg['model_name']]['model_weight']])
        label_weight[predictor.default_cfg['model_name']] = np.array(
            [json_data[predictor.default_cfg['model_name']]['label_weight']])

    return total_pred_output, total_true_output


if __name__ == '__main__':
    st = time.time()

    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    model_weight, label_weight, weights = {}, {}, {}
    predictions_dict = {}
    output_root = r'./output'
    data_root = r'/home/data/classification/action/new_data'
    model_list = []

    # checkpoint_list = ['20200923-111323-gluon_seresnext101_32x4d-224',
    #                    '20200923-213438-hrnet_w48-288',
    #                    '20200923-175500-gluon_seresnext101_64x4d-224']
    # checkpoint_list = ['20200928-210253-ig_resnext101_32x8d-224',
    #                    '20200923-111323-gluon_seresnext101_32x4d-224',
    #                    '20200928-091837-hrnet_w44-224']
    checkpoint_list = ['20201001-140815-hrnet_w44-288',
                       '20201007-192904-ig_resnext101_32x16d-320',
                       '20201003-220359-ig_resnext101_32x16d-288',
                       '20201005-183234-ig_resnext101_32x8d-320',
                       '20201012-184118-ig_seresnext101_32x8d-224',
                       '20201014-232454-ig_resnext101_32x16d-320']

    for checkpoint in checkpoint_list:
        name = '-'.join(checkpoint.split('-')[1:])
        model_list.append(name)

    for index, model_name in enumerate(model_list):
        img_size = int(model_name.split('-')[-1])
        if args.flag == "valid":
            save_weights = True
        else:
            save_weights = False
        dataset = Dataset(os.path.join(data_root, args.flag))

        checkpoint = glob.glob(os.path.join(output_root, checkpoint_list[index] + '/*best*.pth.tar'))[0]
        model = create_model('%s' % model_name.split('-')[-2], num_classes=args.num_classes, in_chans=3,
                             checkpoint_path='%s' % checkpoint)
        model = model.cuda()
        model.eval()

        config = resolve_data_config(vars(args), model=model)
        loader = create_loader(
            dataset,
            input_size=img_size,
            batch_size=args.batch_size,
            use_prefetcher=args.prefetcher,
            interpolation=config['interpolation'],
            mean=config['mean'],
            std=config['std'],
            num_workers=args.workers,
            crop_pct=config['crop_pct'])
        model.default_cfg['model_name'] = model_name

        prediction_output, true_output = perform_predict(model, loader, model_weight, label_weight, weights,
                                                         save_weights=save_weights)
        predictions_dict.update({model.default_cfg['model_name']: prediction_output})

    if args.flag == 'valid':
        # ['A', 'B', 'C', 'D', 'E', 'P', 'M', 'MM', 'ML']
        INTEGRATED_POLICY = ['B', 'C', 'D']
    else:
        INTEGRATED_POLICY = ['C']
    predictor = IntegratedPredictor(model_list, [predictions_dict, model_weight, label_weight], args,
                                    policies=INTEGRATED_POLICY, all_combine=True)
    dump_csv(predictor, true_output)

    print("total time: %.4f" % (time.time()-st))