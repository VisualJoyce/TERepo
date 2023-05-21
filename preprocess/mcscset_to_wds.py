"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import argparse
import os

import webdataset as wds


def convert_func(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with wds.ShardWriter(f"{output_dir}/%08d.tar", maxcount=1000) as sink, open(input_file, 'r',
                                                                                encoding='utf-8') as fp, open(
        input_file.replace('.txt', '_gold.txt'), 'w', encoding='utf-8') as wp:
        with open(input_file) as sfd:
            for i, l in enumerate(sfd):
                items = l.strip().split('\t')
                source, target = items
                sample = {
                    "__key__": f"{i}",
                    "json": {
                        'source': source,
                        'target': target
                    }
                }
                sink.write(sample)
                wp.write(f'{i}\t{source}\t{target}\n')


def main(opts):
    convert_func(os.path.join(opts.input_dir, "train.txt"), os.path.join(opts.output_dir, 'train'))
    convert_func(os.path.join(opts.input_dir, "valid.txt"), os.path.join(opts.output_dir, 'valid'))
    convert_func(os.path.join(opts.input_dir, "test.txt"), os.path.join(opts.output_dir, 'test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='annotation JSON')
    parser.add_argument('--output_dir', required=True, help='annotation JSON')
    parser.add_argument('--cores', type=int, default=os.cpu_count(), help='JSON config files')
    args = parser.parse_args()
    main(args)
