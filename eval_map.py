import glob
import json
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    # files = [os.path.join(directory, f) for f in os.listdir(directory)]
    # files.sort()
    with open(directory, 'r', encoding="utf-8") as f:
        data_info = json.load(f)
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f, v in data_info.items():
        nameOfImage = str(f.split('/')[-1]).split('.')[0]
        # fh1 = open(f, "r")
        # for line in fh1:
        #     line = line.replace("\n", "")
        #     if line.replace(' ', '') == '':
        #         continue
        #     splitLine = line.split(" ")
        if isGT:
            # idClass = int(splitLine[0]) #class
            idClass = (v[0])  # class
            x = float(0)
            y = float(0)
            w = float(v[1])
            h = float(v[2])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.GroundTruth,
                format=bbFormat)
        else:
            # idClass = int(splitLine[0]) #class
            idClass = (v[0])  # class
            confidence = float(v[1])
            x = float(0)
            y = float(0)
            w = float(v[2])
            h = float(v[3])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.Detected,
                confidence,
                format=bbFormat)
        allBoundingBoxes.addBoundingBox(bb)
        if idClass not in allClasses:
            allClasses.append(idClass)
        # fh1.close()
    return allBoundingBoxes, allClasses


def eval_map(gtFolder=None, detFolder="", return_each_ap=False):
    currentPath = os.path.dirname(os.path.abspath(__file__))

    if gtFolder == None:
        gtFolder = os.path.join(currentPath, "txts/v-info.json")
    else:
        gtFolder = gtFolder

    gtFormat, detFormat = BBFormat.XYWH, BBFormat.XYWH
    # Groundtruth folder
    # detFolder = os.path.join(currentPath, detFolder)
    # Coordinates types
    gtCoordType, detCoordType = CoordinatesType.Absolute, CoordinatesType.Absolute
    imgSize = (0, 0)

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.75,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,  # ElevenPointInterpolation EveryPointInterpolation
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=None,
        showGraphic=False)

    each_ap = []
    # each detection is a class
    for metricsPerClass in detections:
        # Get metric values per each class
        ap = metricsPerClass['AP']
        # cl = metricsPerClass['class']
        totalPositives = metricsPerClass['total positives']

        each_ap.append(ap)
        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap

    mAP = acc_AP / validClasses

    if return_each_ap:
        return mAP, each_ap
    else:
        return mAP

# print(eval_map(detFolder="outputv2/20201029-230706-ig_resnext101_32x8d-224/v2.json", gtFolder="txts/v2-info-test.json", return_each_ap=True))