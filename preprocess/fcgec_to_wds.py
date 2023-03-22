import argparse
import json
import os
from itertools import product

import webdataset as wds


def apply_op(source, op):
    target = [c for c in source]
    if 'Switch' in op:
        if len(op['Switch']) == len(source):
            target = [source[i] for i in op['Switch']]
        else:
            # assert len(op['Switch']) == 2 * len(source)
            # target = [source[i] for i in op['Switch']]
            # print(len(op['Switch']), len(source))
            pass

    if 'Delete' in op:
        target = ['' if i in op['Delete'] else c for i, c in enumerate(target)]

    if 'Insert' in op:
        for m in op['Insert']:
            pos = m['pos']
            if isinstance(m['label'], list):
                target[pos] = [target[pos] + l for l in m['label']]
            else:
                target[pos] = target[pos] + m['label']

    if 'Modify' in op:
        # print(op['Modify'])
        for m in op['Modify']:
            pos = m['pos']
            for tag in m['tag'].split('+'):
                if tag.startswith('MOD'):
                    t, n = tag.split('_')
                    n = int(n)
                    target[pos] = m['label']
                    for i in range(1, n):
                        target[pos + i] = ''

    target = [c if isinstance(c, list) else [c] for c in target]

    return list(product(*target))


def convert_fcgec(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with wds.ShardWriter(f"{output_dir}/%08d.tar", maxcount=10000) as sink:
        with open(input_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            for k, element in data.items():
                error_type = element['error_type']
                source = element['sentence'].strip().strip('\ue004')
                operation = json.loads(element['operation'])
                if operation:
                    for i, op in enumerate(operation):
                        if 'Switch' in op and len(op['Switch']) != len(source):
                            print(k)

                        for j, target in enumerate(apply_op(source, op)):
                            sample = {
                                "__key__": f"{k}-{i}-{j}",
                                "json": {
                                    'source': source,
                                    'target': ''.join(target),
                                    'error_type': error_type
                                }
                            }
                            sink.write(sample)
                else:
                    sample = {
                        "__key__": f"{k}-{i}-{j}",
                        "json": {
                            'source': source,
                            'target': source,
                            'error_type': error_type
                        }
                    }
                    sink.write(sample)


def main(opts):
    convert_fcgec(os.path.join(opts.input_dir, "data/FCGEC_train.json"),
                  os.path.join(opts.output_dir, 'train'))
    convert_fcgec(os.path.join(opts.input_dir, "data/FCGEC_valid.json"),
                  os.path.join(opts.output_dir, 'valid'))
    # convert_func(os.path.join(opts.input_dir, "data/mucgec_A/MuCGEC_dev.txt"),
    #              os.path.join(opts.output_dir, 'dev'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='annotation JSON')
    parser.add_argument('--output_dir', required=True, help='annotation JSON')
    parser.add_argument('--cores', type=int, default=os.cpu_count(), help='JSON config files')
    args = parser.parse_args()
    main(args)
