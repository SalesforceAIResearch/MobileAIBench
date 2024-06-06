import os
import json
import random
import tqdm
# import numpy as np
# from PIL import Image
from datasets import load_dataset
from utils.utils import image_to_base64_data_uri, pil_to_base64_data_uri


def data_processing(dataset, downsample_size=None, dataset_dir=None, split="test"):
    if dataset == "vqav2":
        data = process_data_vqav2(dataset_dir, split, downsample_size=downsample_size)
    elif dataset == "viswiz":
        data = process_data_viswiz(dataset_dir, split, downsample_size=downsample_size)
    elif dataset == "scienceqa":
        data = process_data_scienceqa(dataset_dir, split, downsample_size=downsample_size)
    elif dataset == "textvqa":
        data = process_data_textvqa(dataset_dir, split, downsample_size=downsample_size)
    elif dataset == "gqa":
        data = process_data_gqa(dataset_dir, split, downsample_size=downsample_size)
    else:
        raise ValueError(f"No dataset {dataset}")
    return data


def select_column(example, required_fields):
    return {key: example[key] for key in ['a', 'b']}


def process_data_vqav2(dataset_dir, split, downsample_size):
    quation_dir = os.path.join(dataset_dir, "vqa-v2/v2_OpenEnded_mscoco_val2014_questions.json")
    with open(quation_dir, 'r') as f:
        questions = json.load(f)['questions']
    question_dict = {str(q["question_id"]): q["question"] for q in questions}
    answer_dir = os.path.join(dataset_dir, "vqa-v2/v2_mscoco_val2014_annotations.json")
    with open(answer_dir, 'r') as f:
        annotations = json.load(f)['annotations']
    
    image_dir = os.path.join(dataset_dir, "vqa-v2/val2014")

    question_ids = [a['question_id'] for a in annotations]
    if downsample_size:
        question_ids = random.sample(question_ids, downsample_size)

    question_ids = set(question_ids)
    questions = {}
    answers = {}
    images = {}

    for a in annotations:
        q_id = a['question_id']
        if q_id in question_ids:
            i_id = str(a['image_id'])
            questions[str(q_id)] = {'image_id': i_id, "question": question_dict[str(q_id)]}
            answers[str(q_id)] = {'image_id': i_id, "multiple_choice_answer": a['multiple_choice_answer'], "answer": [answer['answer'] for answer in a['answers']]}
            image_path = os.path.join(image_dir, f"COCO_val2014_{i_id.zfill(12)}.jpg")
            images[i_id] = image_to_base64_data_uri(image_path)
    return questions, answers, images


def process_data_viswiz(dataset_dir, split, downsample_size):
    annotation_dir = os.path.join(dataset_dir, "viswiz/val.json")
    with open(annotation_dir, 'r') as f:
        annotations = json.load(f)
    image_ids = []
    for a in annotations:
        id_ = a['image'].split(".")[0]
        image_ids.append(id_)

    if downsample_size:
        image_ids = random.sample(list(image_ids), downsample_size)

    annotations = [a for a in annotations if a['image'].split(".")[0] in image_ids]
    for a in annotations:
        a['image_id'] = a.pop('image').split(".")[0]
    annotations = {a['image_id']: a for a in annotations}
    image_ids = set(image_ids)

    image_dir = os.path.join(dataset_dir, "viswiz/val")
    images = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            key = filename.split('.')[0]
            if key in image_ids:
                image_path = os.path.join(image_dir, filename)
                images[key] = image_to_base64_data_uri(image_path)
    return annotations, annotations, images


def process_data_scienceqa(dataset_dir, split, downsample_size):
    hf_dataset = load_dataset('derek-thomas/ScienceQA', cache_dir="dataset_dir")
    test_dataset = hf_dataset["validation"]
    valid_ids = []

    valid_ids = [i for i, img in enumerate(test_dataset['image']) if img is not None]
    if downsample_size:
        image_ids = random.sample(list(valid_ids), downsample_size)
    else:
        image_ids = valid_ids
    image_ids = set(image_ids)

    questions = {}
    answers = {}
    images = {}
    options_char = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, im in enumerate(tqdm.tqdm(test_dataset['image'], desc="Processing items")):
        if i not in image_ids:
            continue
        if test_dataset['hint'][i]:
            question_QCM = "Context: " + test_dataset['hint'][i] + "\n"
        else: 
            question_QCM = ""
        question_QCM += "Question: " + test_dataset['question'][i] + "\n"  # q part
        question_QCM += "Options: "
        for opt_n in range(len(test_dataset['choices'][i])):
            question_QCM += f"({options_char[opt_n]}): {test_dataset['choices'][i][opt_n]} "
        questions[str(i)] = {"image_id": str(i), "question": question_QCM}
        answers[str(i)] = {"image_id": str(i), "answer": options_char[test_dataset['answer'][i]]}
        images[str(i)] = pil_to_base64_data_uri(im)
    return questions, answers, images


def process_data_textvqa(dataset_dir, split, downsample_size):
    hf_dataset = load_dataset('textvqa', cache_dir="dataset_dir")
    test_dataset = hf_dataset["validation"]
    question_ids = list(test_dataset['question_id'])
    if downsample_size:
        question_ids = random.sample(question_ids, downsample_size)
    question_ids = set(question_ids)
    questions = {}
    answers = {}
    images = {}
    for i, data in enumerate(tqdm.tqdm(test_dataset, desc="Processing items")):
        q_id = data['question_id']
        if q_id not in question_ids:
            continue
        i_id = data['image_id']
        questions[str(q_id)] = {"image_id": str(i_id), "question": data['question']}
        answers[str(q_id)] = {"image_id": str(i_id), "answer": data['answers']}
        if i_id not in images:
            images[i_id] = pil_to_base64_data_uri(data['image'])
    return questions, answers, images


def process_data_gqa(dataset_dir, split, downsample_size):
    annotation_dir = os.path.join(dataset_dir, "gqa/testdev.json")
    image_dir = os.path.join(dataset_dir, "gqa/images")
    with open(annotation_dir, 'r') as f:
        annotations = json.load(f)
    
    question_ids = [a['question_id'] for a in annotations]
    if downsample_size:
        question_ids = random.sample(question_ids, downsample_size)

    question_ids = set(question_ids)
    questions = {}
    answers = {}
    images = {}

    for a in annotations:
        q_id = a['question_id']
        if q_id in question_ids:
            i_id = str(a['img_id'])
            questions[str(q_id)] = {'image_id': i_id, "question": a['sent']}
            answers[str(q_id)] = {'image_id': i_id, "answer": next(iter(a['label'].keys()))}
            image_path = os.path.join(image_dir, i_id + '.jpg')
            images[i_id] = image_to_base64_data_uri(image_path)
    return questions, answers, images
