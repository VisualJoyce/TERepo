"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import argparse
import os

import webdataset as wds


def convert_func(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with wds.ShardWriter(f"{output_dir}/%08d.tar", maxcount=1000) as sink:
        with open(input_file) as sfd:
            for i, l in enumerate(sfd):
                items = l.strip().split('\t')
                sens_id, source = items[:2]
                for j, target in enumerate(items[3:] or [source]):
                    sample = {
                        "__key__": f"{sens_id}_{j}",
                        "json": {
                            'source': source,
                            'target': target
                        }
                    }
                    sink.write(sample)


def convert_mucgec(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with wds.ShardWriter(f"{output_dir}/%08d.tar", maxcount=1000) as sink:
        with open(input_file) as sfd:
            for i, l in enumerate(sfd):
                source, target = l.strip().split('\t')
                sample = {
                    "__key__": f"{i}",
                    "json": {
                        'source': source,
                        'target': target
                    }
                }
                sink.write(sample)


def main(opts):
    convert_mucgec(os.path.join(opts.input_dir, "data/MuCGEC_exp_data/train/train.para"),
                   os.path.join(opts.output_dir, 'train'))
    convert_mucgec(os.path.join(opts.input_dir, "data/MuCGEC_exp_data/valid/valid.para"),
                   os.path.join(opts.output_dir, 'valid'))
    convert_func(os.path.join(opts.input_dir, "data/MuCGEC/MuCGEC_dev.txt"),
                 os.path.join(opts.output_dir, 'dev'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='annotation JSON')
    parser.add_argument('--output_dir', required=True, help='annotation JSON')
    parser.add_argument('--cores', type=int, default=os.cpu_count(), help='JSON config files')
    args = parser.parse_args()
    main(args)
