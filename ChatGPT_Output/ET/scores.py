import json
import torch

from torchmetrics.classification import MulticlassCalibrationError

# 1. Sample Data Scores
ans = json.load(open("./BBN_Sample_ans.json", "r"))

all_data_num = len(ans)

acc_o_correct = len([item for item in ans if item["isOpenCorrect"] == 1]) / all_data_num
acc_c_correct = len([item for item in ans if item["isCloseCorrect"] == 1]) / all_data_num

ifR4OETAuto = [item["ifR4OETAuto"] for item in ans if item["ifR4OETAuto"] != None]
ifR4OETManual = [item["ifR4OETManual"] for item in ans if item["ifR4OETManual"] != None]
ifR4OETFicManual = [1 - item["ifR4OETFicManual"] for item in ans if item["ifR4OETFicManual"] != None]

open_preds, open_target = list(), list()
for item in ans:
    open_preds.append([0.0, item["OConf"]])
    if item["isOpenCorrect"] == 1:
        open_target.append(1)
    else:
        open_target.append(0)
open_preds, open_target = torch.tensor(open_preds), torch.tensor(open_target)
metric = MulticlassCalibrationError(num_classes=2, n_bins=50, norm='l1')
open_calib = metric(open_preds, open_target)

ifR4CETAuto = [item["ifR4CETAuto"] for item in ans if item["ifR4CETAuto"] != None]
ifR4CETManual = [item["ifR4CETManual"] for item in ans if item["ifR4CETManual"] != None]
ifR4CETFicManual = [1 - item["ifR4CETFicManual"] for item in ans if item["ifR4CETFicManual"] != None]

id2label = eval(open("id2label.txt", "r").readline())
label2id = { l:i for i, l in enumerate(id2label) }

close_preds, close_target = list(), list()
for item in ans:
    p = [0.0] * len(id2label)
    ClosedET = item["ClosedET"]
    CConf = item["CConf"]
    if CConf == 0:
        continue
    try:
        p[label2id[ClosedET]] = CConf / 100
    except:
        continue
    close_preds.append(p)
    close_target.append(label2id[item["GroundTruth"]])
close_preds, close_target = torch.tensor(close_preds), torch.tensor(close_target)
metric = MulticlassCalibrationError(num_classes=len(id2label), n_bins=50, norm='l1')
close_calib = metric(close_preds, close_target)

OAMRCorrect = list()
for item in ans:
    if item["isOpenCorrect"] == 1:
        if item["ifR4OETAuto"] == item["ifR4OETManual"]:
            OAMRCorrect.append(1)
        else:
            OAMRCorrect.append(0)

CAMRCorrect = list()
for item in ans:
    if item["isCloseCorrect"] == 1:
        if item["ifR4CETAuto"] == item["ifR4CETManual"]:
            CAMRCorrect.append(1)
        else:
            CAMRCorrect.append(0)

right_oconf = [item["OConf"] for item in ans if item["isOpenCorrect"] == 1]
err_oconf = [item["OConf"] for item in ans if item["isOpenCorrect"] == 0]

right_cconf = [item["CConf"] for item in ans if item["isCloseCorrect"] == 1]
err_cconf = [item["CConf"] for item in ans if item["isCloseCorrect"] == 0]

print("= " * 15 + "SAMPLE" + " =" * 15, "\n")
print("### Open:")
print("acc_o_correct\t", acc_o_correct)
print("ifR4OAuto\t", sum(ifR4OETAuto)/len(ifR4OETAuto))
print("ifR4OManual\t", sum(ifR4OETManual)/len(ifR4OETManual))
print("OAMRCorrect\t", sum(OAMRCorrect)/len(OAMRCorrect))
print("ifR4OFicManual\t", sum(ifR4OETFicManual)/len(ifR4OETFicManual))
print("open_calib\t", open_calib.item())
print("- " * 30)
print("right_oconf\t", sum(right_oconf) / len(right_oconf))
print("err_oconf\t", sum(err_oconf) / len(err_oconf))

print()
print("### Close:")
print("acc_c_correct\t", acc_c_correct)
print("ifR4CAuto\t", sum(ifR4CETAuto)/len(ifR4CETAuto))
print("ifR4CManual\t", sum(ifR4CETManual)/len(ifR4CETManual))
print("CAMRCorrect\t", sum(CAMRCorrect)/len(CAMRCorrect))
print("ifR4CFicManual\t", sum(ifR4CETFicManual)/len(ifR4CETFicManual))
print("close_calib\t", close_calib.item())
print("- " * 30)
print("right_cconf\t", sum(right_cconf) / len(right_cconf))
print("err_cconf\t", sum(err_cconf) / len(err_cconf))

print()

# 2. The Whole Data Scores

print("= " * 15 + "WHOLE" + " =" * 15, "\n")
print("### Close:")
ans = json.load(open("./BBN_All_ans.json", "r"))

top3 = list()
top5 = list()
for line in ans:
    if line["groundTruth"] in line["Top3"]:
        top3.append(1)
    else:
        top3.append(0)
    
    if line["groundTruth"] in line["Top5"]:
        top5.append(1)
    else:
        top5.append(0)

print("Top3_recall\t", sum(top3) / len(top3))
print("Top5_recall\t", sum(top5) / len(top5))

rigth_ls = list()
error_ls = list()

for line in ans:
    if line["closePred"] == line["groundTruth"]:
        rigth_ls.append(line["CConf"])
    else:
        error_ls.append(line["CConf"])

print("right_cconf\t", sum(rigth_ls)/len(rigth_ls))
print("err_cconf\t", sum(error_ls)/len(error_ls))

close_preds, close_target = list(), list()
for item in ans:
    p = [0.0] * len(id2label)
    ClosedET = item["closePred"]
    CConf = item["CConf"]
    if CConf == 0:
        continue
    try:
        p[label2id[ClosedET]] = CConf / 100
    except:
        continue
    close_preds.append(p)
    close_target.append(label2id[item["groundTruth"]])
close_preds, close_target = torch.tensor(close_preds), torch.tensor(close_target)
metric = MulticlassCalibrationError(num_classes=len(id2label), n_bins=50, norm='l1')
close_calib = metric(close_preds, close_target)

print("close_calib\t", close_calib.item())