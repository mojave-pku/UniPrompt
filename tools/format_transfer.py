import json
import os
import csv

def raw_data_format_transfer(fin, fout):
    results = []
    with open(fin) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            review = json.loads(line)
            results.append(
                [i, review['review_body'], int(review['stars'])]
            )
        with open(fout, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(results)

if __name__ == '__main__':
    splits = ['train', 'dev', 'test']
    langs = ['de', 'en', 'es', 'fr', 'ja', 'zh']
    for lang in langs:
        if not os.path.exists(f'data/original/amazon-review-{lang}'):
            os.makedirs(f'data/original/amazon-review-{lang}')
        for split in splits:
            input_file = f'../amazon-review/json/{split}/dataset_{lang}_{split}.json'
            output_file = f'data/original/amazon-review-{lang}/{split}.csv'
            raw_data_format_transfer(input_file, output_file)