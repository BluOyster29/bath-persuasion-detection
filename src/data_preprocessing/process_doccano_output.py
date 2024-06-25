import json
import os
import pandas as pd
import argparse


def genargs():

    parser = argparse.ArgumentParser(
        description='Process Doccano output files'
    )

    parser.add_argument(
        'output_folder_path', type=str,
        help='Path to the folder containing the output files from Doccano')
    parser.add_argument(
                '--output_df_path', type=str,
                help='Path to dataframe output file'
    )

    args = parser.parse_args()
    return args


def read_jsonl_file(file_path, user):
    processed_list = []
    with open(file_path) as f:

        jline = f.readlines()

        for idx in range(len(jline)):
            j = json.loads(jline[idx])
            j['page'] = int(j['text'].split(' ')[1])
            j['user'] = user
            for i in j['label']:
                i.append(user)
            j['num_labels'] = len(j['label'])
            if len(j['label']) < 1:
                continue
            else:
                processed_list.append(j)

    return processed_list


def process_jsonlists(output_folder_path):
    dataset = []
    for file in os.listdir(output_folder_path):
        path = output_folder_path + '/' + file

        if path.endswith('.jsonl'):
            jsonl = read_jsonl_file(path, file)
            dataset.extend(jsonl)
    return dataset


def gen_root_df(output_folder_path):
    dataset = process_jsonlists(output_folder_path)
    df = pd.DataFrame(dataset)
    return df


def annotate_string(long_string, labels):
    # Split the long string by new lines to create the list of lines
    lines = long_string.split('\n')

    # Create a list to hold the annotated lines
    annotated_lines = []

    # Flatten the labels list with their original indexes
    all_labels = []
    for start, stop, label, user in labels:
        all_labels.append((start, stop, label, user))

    # Sort labels based on the start index
    all_labels.sort(key=lambda x: x[0])

    # Iterate over each line
    current_pos = 0
    for line in lines:
        line_length = len(line)
        line_labels = []

        # Iterate over each label and check if it falls within the current line
        while all_labels and all_labels[0][0] < current_pos + line_length:
            start, stop, label, user = all_labels.pop(0)

            # Calculate relative start and stop positions within current line
            rel_start = max(0, start - current_pos)
            rel_stop = min(line_length, stop - current_pos)

            # Append the labeled portion if within current line
            if rel_start < line_length and rel_stop > 0:
                line_labels.append((line[rel_start:rel_stop], label, user))

        # Create the annotated line as ['text', labels]
        annotated_lines.append([line, line_labels])

        # Update the current position to the start of the next line
        current_pos += line_length + 1  # +1 for the newline character

    return annotated_lines


def process_annotated_lines(annotated_lines, d, drop_bottom_level=None):

    for i in annotated_lines:
        users = {'Adela': [],
                 'Dessi': [],
                 'Jessie': [],
                 'Nadeen': [],
                 'Samiksha': [],
                 'Sara': []}

        if len(i) < 2:
            continue

        text, annotations = i
        processed_annotations = process_annotations(
            annotations, drop_bottom_level)
        d['text'].append(text)

        for label, user in processed_annotations:
            if user not in users:
                continue
            users[user].append(label)

        merged = {user: '/'.join(labels) for user, labels in users.items()}

        for k, v in merged.items():
            if len(v) == 0:
                v = '8-NO-PERSUASION'
            d[k].append(v)
    return d


def process_annotations(annotations, drop_bottom_level=None):

    labels = []
    for i in annotations:
        text, label, user = i
        user = user.split('.')[0]

        if drop_bottom_level:
            label = '-'.join(label.split('-')[:2])

        if (label, user) not in labels:
            labels.append((label, user))

    labels.sort(key=lambda x: x[1])
    return labels


def gen_multilabel_expanded_df(df, drop_bottom_level=None):
    d = {
        'text': [],
        'Adela': [],
        'Dessi': [],
        'Jessie': [],
        'Nadeen': [],
        'Samiksha': [],
        'Sara': []
    }

    for page_num in range(len(df.page.unique())):
        page_num += 1
        text = df[df['page'] == page_num].iloc[0].text

        labels = []
        for i in df[df['page'] == page_num].label.tolist():
            labels += i

        annotated_lines = annotate_string(text, labels)
        d = process_annotated_lines(annotated_lines[1:], d, drop_bottom_level)
    return pd.DataFrame(d)


def format_df(df):
    for i, row in df.iterrows():
        if len(row.text) < 5:
            df.drop(i, inplace=True)
    return df


def main(args):
    df = gen_root_df(args.output_folder_path)
    processed_df = gen_multilabel_expanded_df(df, args.drop_bottom_level)
    if args.output_df_path:
        processed_df = format_df(processed_df)
        processed_df.to_csv(args.output_df_path, index=False)
    return processed_df


if __name__ == '__main__':
    args = genargs()
    main()
