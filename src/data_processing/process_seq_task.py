import pandas as pd
import json
import re
import argparse


label_columns = [
    '1-RAPPORT',
    '2-NEGOTIATE',
    '3-EMOTION',
    '4-LOGIC',
    '5-AUTHORITY',
    '6-SOCIAL',
    '7-PRESSURE',
    '8-NO-PERSUASION'
]


def genargs():
    p = argparse.ArgumentParser()
    p.add_argument('jsonl_path', type=str, help='Path to the jsonl file')
    p.add_argument('output_path', type=str, help='Path to the output file')
    return p.parse_args()


def open_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data


def gen_idx_map(text):

    start_idx = 0
    idx_map = {}

    for idx, i in enumerate(text.split('\n')):

        end_idx = start_idx + len(i)
        idx_map[idx] = (start_idx, end_idx)
        start_idx = end_idx
    return idx_map


def add_labels_to_text(text, labels):

    idx_map = gen_idx_map(text)
    splat_text = text.split('\n')
    for label in labels:
        _,  end_idx, label = label
        for idx, (s, e) in idx_map.items():
            if end_idx <= e:
                splat_text[idx] += ' - ' + label
                break
    return splat_text


def process_transcripts(jsonl, text_column):

    transcripts = []
    for idx, line in enumerate(jsonl):
        j = json.loads(line)
        text = j[text_column]
        labels = j['label']
        if len(labels) < 1:
            break
        transcripts.append(add_labels_to_text(text, labels))

    return transcripts


def process_line(line):

    split_line = line.split(' - ')

    if len(split_line) > 2:
        text, label = split_line[0], split_line[1:]

    elif len(split_line) < 2:
        text, label = line, '8-NO-PERSUASION'
    else:
        text, label = line.split(' - ')

    return text[10:], label


def update_labels(label_columns, data, labels: list):

    if not isinstance(labels, list):
        labels = [labels]

    for i in label_columns:
        if i in labels:
            data[i].append(1)
        else:
            data[i].append(0)
    return data


def generate_dataframe(jsonl, label_columns, text_column):

    transcripts = process_transcripts(jsonl, text_column)

    data = {i: [] for i in label_columns}
    data[text_column] = []

    for t in transcripts:
        for line in t:

            if re.match('Persuader:', line):
                text, label = process_line(line)
                data[text_column].append(text)
                data = update_labels(label_columns, data, label)

            else:
                continue
    df = pd.DataFrame(data)
    return df[[text_column] + label_columns]


if __name__ == '__main__':
    args = genargs()
    jsonl = open_jsonl(args.jsonl_path)
    df = generate_dataframe(jsonl, label_columns)
    df.to_csv(args.output_path, index=False)
