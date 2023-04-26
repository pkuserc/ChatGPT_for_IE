import os
import json
from const import EVENT_TYPE
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


def evaluate(preds, gold):
    assert len(preds) == len(gold)

    for example_id in preds:
        preds[example_id] = list(set([tuple(i) for i in preds[example_id]]))
        gold[example_id] = [tuple(i) for i in gold[example_id]]


    pred_tri_num, gold_tri_num = 0, 0
    match_idn_num, match_cls_num, match_word_num = 0, 0, 0
    correct_confidence = 0
    incorrect_confidence = 0
    if_reasonable_num = 0
    for example_id in preds:
        pred_tri_num += len(preds[example_id])
        gold_tri_num += len(gold[example_id])

    calibrate_record = []
    invalid_event_type = 0
    for example_id in preds:
        for pred_tri in preds[example_id]:
            start, end, event_type, trigger_word, confidence, if_reasonable = pred_tri
            match_idn = {item for item in gold[example_id] if item[0] == start and item[1] == end}
            match_word = [item for item in gold[example_id] if item[3] == trigger_word and item[2]==event_type]
            if match_word:
                match_word_num += 1
                correct_confidence += confidence
                if if_reasonable:
                    if_reasonable_num += 1
                if confidence == 0:
                    invalid_event_type += 1
                    continue
                gold_label_idx = LABEL2ID[match_word[0][2]]
                pred_label_idx = LABEL2ID[event_type]
                calibrate_record.append([gold_label_idx, pred_label_idx, confidence / 100])
            else:
                incorrect_confidence += confidence
                if event_type not in LABEL2ID:
                    invalid_event_type += 1
                    continue
                if confidence == 0:
                    invalid_event_type += 1
                    continue
                gold_label_idx = LABEL2ID['None']
                pred_label_idx = LABEL2ID[event_type]
                calibrate_record.append([gold_label_idx, pred_label_idx, confidence / 100])
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
    print('(Trigger Word Cls) Correct Mean Confidence: {:.2f}, Incorrect Mean Confidence: {:.2f}'.format(correct_confidence / match_word_num, incorrect_confidence / (pred_tri_num - match_word_num)))
    print(f' Auto Rate: {if_reasonable_num/match_word_num}')

    # Compute Expected Calibration Error (ECE)
    assert len(calibrate_record) == (pred_tri_num - invalid_event_type)
    print(invalid_event_type, len(calibrate_record))
    label_idx, pred_idx, prob = zip(*calibrate_record)
    labels = torch.tensor(label_idx)
    preds = torch.zeros(len(calibrate_record), len(LABEL2ID), dtype=torch.float32)
    preds[range(len(calibrate_record)), pred_idx] = torch.tensor(prob)
    metric = MulticlassCalibrationError(num_classes=34, n_bins=50, norm='l1')
    result = metric(preds, labels)
    print('Expected Calibration Error: {:.5f}'.format(result))


def filter_invalid_answer(preds):
    
    def if_invalid(record):
        filter_words = ['unknown', 'Unknown', 'unspecified', 'not specified', 'not mentioned', 'None', 'none', 'not mentioned', 'not applicable', 'N/A']
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
    
    count = 0
    for example_id in preds:
        for record in preds[example_id][::-1]:
            if if_invalid(record):
                preds[example_id].remove(record)
                count += 1
    for example_id in preds:
        for record in preds[example_id]:
            record[1] += 1      # end + 1  
    return count
        

def read_gold_example(path):
    gold = {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        events = line['events']
        gold[line['id']] = []
        for event in events:
            start = event['trigger']['start']
            end = event['trigger']['end']
            event_type = event['event_type'].replace(':','.')
            trigger_word = event['trigger']['text']
            gold[line['id']].append([start, end, event_type, trigger_word])

    return gold


def get_vocab():
    all_labels = ['None']
    for label in EVENT_TYPE:
        all_labels.append(label)
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for idx, label in enumerate(all_labels)}
    return label2id, id2label


def main(result_dir, gold_path):
    gold = read_gold_example(gold_path)
    
    preds = {}
    # print(len(os.listdir(result_dir)))
    for file in os.listdir(result_dir):
        example_id = file[:-5]
        preds[example_id] = []
        file_path = os.path.join(result_dir, file)
        # print(file)
        with open(file_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for event in res:
            start = event['start_word_index']
            end = event['end_word_index']
            event_type = event['event_type']
            trigger_word = event['trigger']
            confidence = event['confidence']
            if_reasonable = event['if_reasonable']
            preds[example_id].append([start, end, event_type, trigger_word, confidence, if_reasonable])
    
    invalid_arg_num = filter_invalid_answer(preds)
    print(invalid_arg_num)
    evaluate(preds, gold)


LABEL2ID,ID2LABEL = get_vocab()
if __name__ == "__main__":
    result_dir = './Output/ED/Full_Testset/ED_E+_Closed'
    gold_path = './data/ACE05-E+/ED_E+_gold.json'
    main(result_dir, gold_path)
    