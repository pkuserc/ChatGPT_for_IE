import os
import json
from const import ROLE, EVENT_TYPE
import torch
from torchmetrics.classification import MulticlassCalibrationError


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def evaluate(preds_event, preds_arg, gold_event, gold_arg):
    assert len(preds_event) == len(gold_event)
    assert len(preds_arg) == len(gold_arg)

    # trigger
    for example_id in preds_event:
        preds_event[example_id] = list(set([tuple(i) for i in preds_event[example_id]]))
        gold_event[example_id] = [tuple(i) for i in gold_event[example_id]]


    pred_tri_num, gold_tri_num = 0, 0
    match_idn_num, match_cls_num, match_word_num = 0, 0, 0

    tri_correct_confidence = 0
    tri_incorrect_confidence = 0
    tri_if_reasonable_num = 0
    for example_id in preds_event:
        pred_tri_num += len(preds_event[example_id])
        gold_tri_num += len(gold_event[example_id])

    calibrate_record = []
    invalid_num = 0
    for example_id in preds_event:
        for pred_tri in preds_event[example_id]:
            trigger_start, trigger_end, event_type, trigger_word, tri_confidence, tri_if_reasonable = pred_tri
            match_idn = {item for item in gold_event[example_id] if item[0] == trigger_start and item[1] == trigger_end}
            match_word = [item for item in gold_event[example_id] if item[3] == trigger_word and item[2]==event_type]
            if match_word:
                match_word_num += 1
                tri_correct_confidence += tri_confidence
                if tri_if_reasonable:
                    tri_if_reasonable_num += 1
                if tri_confidence == 0:
                    invalid_num += 1
                    continue
                gold_label_idx = LABEL2ID[match_word[0][2]]
                pred_label_idx = LABEL2ID[event_type]
                calibrate_record.append([gold_label_idx, pred_label_idx, tri_confidence / 100])
            else:
                tri_incorrect_confidence += tri_confidence
                if event_type not in LABEL2ID:
                    invalid_num += 1
                    continue
                if tri_confidence == 0:
                    invalid_num += 1
                    continue
                gold_label_idx = LABEL2ID['None']
                pred_label_idx = LABEL2ID[event_type]
                calibrate_record.append([gold_label_idx, pred_label_idx, tri_confidence / 100])
            if match_idn:
                match_idn_num += 1
                match_cls = {item for item in match_idn if item[2] == event_type}
                if match_cls:
                    match_cls_num += 1
    
    print(f"gold_tri_num: {gold_tri_num}, pred_tri_num: {pred_tri_num}, match_idn_num: {match_idn_num}, match_cls_num: {match_cls_num}, match_word_num: {match_word_num}")

    tri_id_prec, tri_id_rec, tri_id_f = compute_f1(pred_tri_num, gold_tri_num, match_idn_num)
    tri_cls_prec, tri_cls_rec, tri_cls_f = compute_f1(pred_tri_num, gold_tri_num, match_cls_num)
    tri_word_prec, tri_word_rec, tri_word_f = compute_f1(pred_tri_num, gold_tri_num, match_word_num)
    print('Trigger Identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(tri_id_prec * 100.0, tri_id_rec * 100.0, tri_id_f * 100.0))
    print('Trigger Classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(tri_cls_prec * 100.0, tri_cls_rec * 100.0, tri_cls_f * 100.0))
    print('Trigger Word Cls: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(tri_word_prec * 100.0, tri_word_rec * 100.0, tri_word_f * 100.0))
    print('(Trigger Word Cls) Correct Mean Confidence: {:.2f}, Incorrect Mean Confidence: {:.2f}'.format(tri_correct_confidence / match_word_num, tri_incorrect_confidence / (pred_tri_num - match_word_num)))
    print(f' Auto Rate: {tri_if_reasonable_num/match_word_num}')

    # argument
    for example_id in preds_arg:
        preds_arg[example_id] = list(set([tuple(i) for i in preds_arg[example_id]]))
        gold_arg[example_id] = [tuple(i) for i in gold_arg[example_id]]


    pred_arg_num, gold_arg_num = 0, 0
    arg_idn_num, arg_class_num, arg_ic_num = 0, 0, 0

    arg_correct_confidence = 0
    arg_incorrect_confidence = 0
    arg_if_reasonable_num = 0
    for example_id in preds_arg:
        pred_arg_num += len(preds_arg[example_id])
        gold_arg_num += len(gold_arg[example_id])

    for example_id in preds_arg:
        for pred_arg in preds_arg[example_id]:
            start, end, event_type, role, text, arg_confidence, arg_if_reasonable = pred_arg
            gold_idn = {item for item in gold_arg[example_id] if item[0] == start and item[1] == end}
            gold_ic = [item for item in gold_arg[example_id] if item[2] == event_type and item[3] == role and item[4] == text]
            if gold_ic:
                arg_ic_num += 1
                arg_correct_confidence += arg_confidence
                if arg_if_reasonable:
                    arg_if_reasonable_num += 1
                if arg_confidence == 0:
                    invalid_num += 1
                    continue
                gold_label_idx = LABEL2ID[gold_ic[0][3]]
                pred_label_idx = LABEL2ID[role]
                calibrate_record.append([gold_label_idx, pred_label_idx, arg_confidence / 100])
            else:
                arg_incorrect_confidence += arg_confidence
                if role not in LABEL2ID:
                    invalid_num += 1
                    continue
                if arg_confidence == 0:
                    invalid_num += 1
                    continue
                gold_label_idx = LABEL2ID['None']
                pred_label_idx = LABEL2ID[role]
                calibrate_record.append([gold_label_idx, pred_label_idx, arg_confidence / 100])
            if gold_idn:
                arg_idn_num += 1
                gold_class = {item for item in gold_idn if item[2] == event_type and item[3] == role}
                if gold_class:
                    arg_class_num += 1
    
    print(f"gold_arg_num: {gold_arg_num}, pred_arg_num: {pred_arg_num}, arg_idn_num: {arg_idn_num}, arg_class_num: {arg_class_num}, arg_ic_num: {arg_ic_num}")

    role_id_prec, role_id_rec, role_id_f = compute_f1(pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(pred_arg_num, gold_arg_num, arg_class_num)
    role_ic_prec, role_ic_rec, role_ic_f = compute_f1(pred_arg_num, gold_arg_num, arg_ic_num)
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
    print('Role ic: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(role_ic_prec * 100.0, role_ic_rec * 100.0, role_ic_f * 100.0))
    print('(Role ic) Correct Mean Confidence: {:.2f}, Incorrect Mean Confidence: {:.2f}'.format(arg_correct_confidence / arg_ic_num, arg_incorrect_confidence / (pred_arg_num - arg_ic_num)))
    print(f' Auto Rate: {arg_if_reasonable_num/arg_ic_num}')

    # Compute Expected Calibration Error (ECE)
    print(invalid_num, len(calibrate_record))
    assert len(calibrate_record) == (pred_tri_num + pred_arg_num - invalid_num)
    label_idx, pred_idx, prob = zip(*calibrate_record)
    labels = torch.tensor(label_idx)
    preds = torch.zeros(len(calibrate_record), len(LABEL2ID), dtype=torch.float32)
    preds[range(len(calibrate_record)), pred_idx] = torch.tensor(prob)
    metric = MulticlassCalibrationError(num_classes=56, n_bins=50, norm='l1')
    result = metric(preds, labels)
    print('Expected Calibration Error: {:.5f}'.format(result))


def filter_invalid_answer(preds_event, preds_arg):
    "过滤掉不合法的输出，比如索引为负数的答案"
    
    def if_invalid_event(record):
        filter_words = ['unknown', 'Unknown', 'unspecified', 'not specified', 'not mentioned', 'None', 'none', 'NONE', 'not mentioned', 'not applicable', 'N/A']
        if not isinstance(record[2], str):
            return True
        elif not isinstance(record[3], str):
            return True
        elif not isinstance(record[0], int):
            return True
        elif not isinstance(record[1], int):
            return True
        elif not record[0]>=0:
            return True
        elif not record[1]>=0:
            return True
        elif record[2] in filter_words:
            return True
        elif record[3] in filter_words:
            return True
        return False

    def if_invalid_arg(argument):
        filter_words = ['None', 'NONE',"N/A"]
        if not isinstance(argument[2], str):
            return True
        elif not isinstance(argument[3], str):
            return True
        elif not isinstance(argument[4], str):
            return True
        elif not (isinstance(argument[0], int) and isinstance(argument[1], int)):
            return True
        elif not (argument[0]>=0 and argument[1]>=0):
            return True
        elif argument[2] in filter_words:
            return True
        elif argument[3] in filter_words:
            return True
        elif argument[4] in filter_words:
            return True
        return False
    
    count_event, count_arg = 0, 0

    for example_id in preds_event:
        for record in preds_event[example_id][::-1]:
            if if_invalid_event(record):
                preds_event[example_id].remove(record)
                count_event += 1
    for example_id in preds_event:
        for record in preds_event[example_id]:
            record[1] += 1   # end + 1 

    count_arg = 0
    for example_id in preds_arg:
        for argument in preds_arg[example_id][::-1]:
            if if_invalid_arg(argument):
                preds_arg[example_id].remove(argument)
                count_arg += 1
    for example_id in preds_arg:
        for argument in preds_arg[example_id]:
            argument[1] += 1


    return count_event, count_arg


def read_gold_example(path):
    gold_event, gold_arg = {}, {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        events = line['event']
        gold_event[line['id']] = []
        gold_arg[line['id']] = []
        for event in events:
            event_type = event['event_type']

            trigger_start = event['trigger']['start']
            trigger_end = event['trigger']['end']
            trigger_word = event['trigger']['text']
            gold_event[line['id']].append([trigger_start, trigger_end, event_type, trigger_word])

            for arg in event['arguments']:
                start = arg['start']
                end = arg['end']
                role = arg['role']
                text = arg['text']
                # print(text)
                gold_arg[line['id']].append([start, end, event_type, role, text])

    return gold_event, gold_arg


def get_vocab():
    all_labels = ['None']
    for label in EVENT_TYPE + ROLE:
        all_labels.append(label)
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for idx, label in enumerate(all_labels)}
    return label2id, id2label


def main(result_dir, gold_path):
    gold_event, gold_arg = read_gold_example(gold_path)
    

    preds_event, preds_arg = {}, {}
    # print(len(os.listdir(result_dir)))
    for file in os.listdir(result_dir):
        example_id = file[:-5]
        preds_event[example_id] = []
        preds_arg[example_id] = []
        file_path = os.path.join(result_dir, file)
        print(file)
        with open(file_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for event in res:
            event_type = event['event_type']
            trigger_start = event['start_word_index']
            trigger_end = event['end_word_index']
            trigger_word = event['trigger']
            tri_confidence = event['confidence']
            if 'if_reasonable' not in event:
                event['if_reasonable'] = 0
            else:
                tri_if_reasonable = event['if_reasonable']
            preds_event[example_id].append([trigger_start, trigger_end, event_type, trigger_word, tri_confidence, tri_if_reasonable])

            for arg in event['participants']:
                start = arg['start_word_index']
                end = arg['end_word_index']
                role = arg['role']
                text = arg['span']
                arg_confidence = arg['confidence']
                if 'if_reasonable' not in arg:
                    arg['if_reasonable'] = 0
                else:
                    arg_if_reasonable = arg['if_reasonable']
                preds_arg[example_id].append([start, end, event_type, role, text, arg_confidence, arg_if_reasonable])
    
    count_event, count_arg = filter_invalid_answer(preds_event, preds_arg)
    print(count_event, count_arg)
    evaluate(preds_event, preds_arg, gold_event, gold_arg)

LABEL2ID,ID2LABEL = get_vocab()
if __name__ == "__main__":
    result_dir = './Output/EE/Full_Testset/EE_E+_Closed'
    gold_path = './data/ACE05-E+/EE_E+_gold.json'
    main(result_dir, gold_path)
    