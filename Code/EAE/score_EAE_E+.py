import os
import json
import spacy
from spacy.tokens import Doc
from const import ROLE, All_Valid_EntTypes
import torch
from torchmetrics.classification import MulticlassCalibrationError


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


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


def find_head(arg_start, arg_end, doc):
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <=arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head 
            break 
        else:
            cur_i = doc[cur_i].head.i
        
    arg_head = cur_i
    
    return (arg_head, arg_head)


def clean_span(tokens, start, end):
    if tokens[start].lower() in {'the', 'an', 'a'}:
        if start!=end:
            return (start+1, end)
    return start, end


def evaluate(preds, gold, only_head=False):
    for example_id in gold:
        for argument in gold[example_id]:
            if only_head:
                words = argument.pop(4)
                doc = nlp(' '.join(words))
                argument[3] -= 1
                argument[2], argument[3] = clean_span(words, argument[2], argument[3])
                argument[2], argument[3] = find_head(argument[2], argument[3], doc)
                assert argument[2] == argument[3]
                argument[1] = words[argument[2]]
                argument[3] += 1
            else:
                argument.pop(4)


    for example_id in preds:
        preds[example_id] = list(set([tuple(i) for i in preds[example_id]]))
        gold[example_id] = [tuple(i) for i in gold[example_id]]

    pred_arg_num, gold_arg_num = 0, 0
    arg_idn_num, arg_class_num, arg_ic_num = 0, 0, 0

    for example_id in preds:
        pred_arg_num += len(preds[example_id])
        gold_arg_num += len(gold[example_id])

    correct_confidence = 0
    incorrect_confidence = 0
    if_reasonable_num = 0
    calibrate_record = []
    invalid_role = 0
    for example_id in preds:
        for pred_arg in preds[example_id]:
            role, span, arg_start, arg_end, confidence, if_reasonable = pred_arg
            gold_idn = {item for item in gold[example_id] if item[2] == arg_start and item[3] == arg_end}
            gold_ic = [item for item in gold[example_id] if item[0] == role and item[1] == span]
            if gold_ic:
                arg_ic_num += 1
                correct_confidence += confidence
                if if_reasonable:
                    if_reasonable_num += 1
                if confidence == 0:
                    invalid_role += 1
                    continue
                gold_label_idx = LABEL2ID[gold_ic[0][0]]
                pred_label_idx = LABEL2ID[role]
                calibrate_record.append([gold_label_idx, pred_label_idx, confidence / 100])
            else:
                incorrect_confidence += confidence
                if role not in LABEL2ID:
                    invalid_role += 1
                    continue
                if confidence == 0:
                    invalid_role += 1
                    continue
                gold_label_idx = LABEL2ID['None']
                pred_label_idx = LABEL2ID[role]
                calibrate_record.append([gold_label_idx, pred_label_idx, confidence / 100])
            if gold_idn:
                arg_idn_num += 1
                gold_class = {item for item in gold_idn if item[0] == role}
                if gold_class:
                    arg_class_num += 1
    
    print(f"gold_arg_num: {gold_arg_num}, pred_arg_num: {pred_arg_num}, arg_idn_num: {arg_idn_num}, arg_class_num: {arg_class_num}, arg_ic_num: {arg_ic_num}")

    role_id_prec, role_id_rec, role_id_f = compute_f1(pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(pred_arg_num, gold_arg_num, arg_class_num)
    role_ic_prec, role_ic_rec, role_ic_f = compute_f1(pred_arg_num, gold_arg_num, arg_ic_num)
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
    print('Role ic: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(role_ic_prec * 100.0, role_ic_rec * 100.0, role_ic_f * 100.0))
    print('(Role ic) Correct Mean Confidence: {:.2f}, Incorrect Mean Confidence: {:.2f}'.format(correct_confidence / arg_ic_num, incorrect_confidence / (pred_arg_num - arg_ic_num)))
    print(f' Auto Rate: {if_reasonable_num/arg_ic_num}')

    # Compute Expected Calibration Error (ECE)
    assert len(calibrate_record) == (pred_arg_num - invalid_role)
    print(invalid_role, len(calibrate_record))
    label_idx, pred_idx, prob = zip(*calibrate_record)
    labels = torch.tensor(label_idx)
    preds = torch.zeros(len(calibrate_record), len(LABEL2ID), dtype=torch.float32)
    preds[range(len(calibrate_record)), pred_idx] = torch.tensor(prob)
    metric = MulticlassCalibrationError(num_classes=23, n_bins=50, norm='l1')
    result = metric(preds, labels)
    print('Expected Calibration Error: {:.5f}'.format(result))


def read_question(question_path):
    Event2Query = {}
    with open(question_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        event_arg, query = line.split(",")
        event, arg = event_arg.split("_")
        if event not in Event2Query:
            Event2Query[event] = []
        Event2Query[event].append((arg, query))
    
    return Event2Query


def filter_invalid_answer(preds, only_head=False):
    "过滤掉不合法的输出，比如索引为负数的答案；end_index加1（因为Prompt中明确说明了end_word_index应该是inclusive的）"
    
    def if_invalid(argument):
        filter_words = ['unknown', 'Unknown', 'unspecified', 'not specified', 'not mentioned', 'None', 'none', 'not mentioned', 'not applicable', 'N/A']
        if not isinstance(argument[1], str):
            return True
        elif not isinstance(argument[0], str):
            return True
        elif not (isinstance(argument[2], int) and isinstance(argument[3], int)):
            return True
        elif not (argument[2]>=0 and argument[3]>=0):
            return True
        elif [i for i in filter_words if i in argument[1]]:
            return True
        return False
    
    count = 0
    for example_id in preds:
        for argument in preds[example_id][::-1]:
            if if_invalid(argument):
                preds[example_id].remove(argument)
                count += 1
    for example_id in preds:
        for argument in preds[example_id]:

            if only_head:
                words = argument.pop(4)

                if argument[2]>=len(words):
                    argument[3] += 1
                    continue

                doc = nlp(' '.join(words))
                argument[2], argument[3] = find_head(argument[2], argument[3], doc)
                assert argument[2] == argument[3]
                argument[1] = words[argument[2]]
            else:
                argument.pop(4)
            argument[3] += 1
                
    return count


def join(word_list):
    res = ''
    for idx, word in enumerate(word_list):
        if idx==0:
            res = word
        else:
            if "'" in word:
                res += word
            elif word == '-':
                res += word
            elif word_list[idx-1] == '-':
                res += word
            else:
                res += (' '+word)
    return res
        

def read_gold_example(path):
    gold = {}
    gold_event = {}
    gold_words = {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        event_type = line['event']['event_type']
        gold[line['id']] = []
        gold_event[line['id']] = event_type
        gold_words[line['id']] = line['words']
        for arg in line['event']['argument']:
            role = arg['role']
            span = arg['text']
            # print(span)
            gold[line['id']].append([role, span, arg['start'] , arg['end'], line['words']])

    return gold, gold_event, gold_words


def get_vocab():
    all_labels = ['None']
    for label in ROLE:
        all_labels.append(label)
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for idx, label in enumerate(all_labels)}
    return label2id, id2label


def main(result_dir, gold_path, question_path, only_head=False):

    Event2Query = read_question(question_path)
    gold, gold_event, gold_words = read_gold_example(gold_path)

    preds = {}
    for file in os.listdir(result_dir):
        example_id = file[:-5]
        preds[example_id] = []
        file_path = os.path.join(result_dir, file)
        # print(file)
        with open(file_path, 'r') as f:
            res = json.load(f)
        question_num = len(res)
        questions = [f"Question{i+1}" for i in range(question_num)]
        event_type = gold_event[example_id]
        words = gold_words[example_id]
        all_role = [i[0] for i in Event2Query[event_type]]
        assert len(all_role) == question_num
        for idx, ques in enumerate(questions):
            answers = res[ques]
            for ans in answers:
                role = all_role[idx]
                span = ans['span']
                confidence = ans['confidence']
                if_reasonable = ans['if_reasonable']
                preds[example_id].append([role, span, ans['start_word_index'] , ans['end_word_index'], words, confidence, if_reasonable])
    

    invalid_arg_num = filter_invalid_answer(preds, only_head)
    print(invalid_arg_num)
    evaluate(preds, gold, only_head)


LABEL2ID,ID2LABEL = get_vocab()
if __name__ == "__main__":
    result_dir = './Output/EAE/Full_Testset/EAE_E+_Closed'
    question_path = './Code/description_queries_new.csv'
    gold_path = "./data/ACE05-E+/EAE_E+_gold.json"
    only_head = False
    main(result_dir, gold_path, question_path, only_head)
    