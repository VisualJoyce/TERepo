"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import argparse
import os

import webdataset as wds


def convert_wikisplit(input_file, output_dir):
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
    convert_wikisplit(os.path.join(opts.input_dir, "train.tsv"),
                      os.path.join(opts.output_dir, 'train'))
    convert_wikisplit(os.path.join(opts.input_dir, "validation.tsv"),
                      os.path.join(opts.output_dir, 'valid'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='annotation JSON')
    parser.add_argument('--output_dir', required=True, help='annotation JSON')
    args = parser.parse_args()
    main(args)
