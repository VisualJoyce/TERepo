import argparse
import os

import webdataset as wds


def convert(input_file, output_dir):
    """
    On each line, we have information for one source text and corresponding compressed versions.

    The format is SourceInfo[ ||| CompressionInfo]+

    SourceInfo is information on the soure text and has the following fields (tab-separated):

    SourceID \t Domain \t SourceText

    The SourceID has one or more integers connected with _, for exmaple "15" or "101_102". There is one integer per sentence in the source text.

    CompressionInfo is information about a compression for the SourceText.

    CompressionInfo has the following fields [tab-separated]:

    CompressedText \t JudgeId \t numRatings[\t Rating]^numRatings

    This is the compressed text (un-tokenized), the JudgeId (the anonymized ids of one or more crowd-workers that proposed this compression ), and indication of how many ratings we have and the sequence of ratings.

    Each Rating has the format:

    CombinedRatingValue \t Meaning_quality_string \t Grammar_quality_string

    The CombinedRatingValue is sufficient to indicate the meaning and grammaticality qualitiy values assigned by a single rater, and the subseqeunt string values are not strictly necessary.

    Args:
        input_file:
        output_dir:

    Returns:

    """
    os.makedirs(output_dir, exist_ok=True)
    with wds.ShardWriter(f"{output_dir}/%08d.tar", maxcount=10000) as sink, open(input_file, 'r',
                                                                                 encoding='utf-8') as fp:
        for k, line in enumerate(fp):
            source_str, *targets = line.split("|||")
            source_id, domain, source = source_str.split("\t")
            for i, t in enumerate(targets):
                target, *_ = t.split("\t")
                sample = {
                    "__key__": f"{source_id}-{i}",
                    "json": {
                        'source': source,
                        'target': target,
                    }
                }
                sink.write(sample)


def main(opts):
    convert(os.path.join(opts.input_dir, "train.tsv"),
            os.path.join(opts.output_dir, 'train'))
    convert(os.path.join(opts.input_dir, "valid.tsv"),
            os.path.join(opts.output_dir, 'valid'))
    convert(os.path.join(opts.input_dir, "test.tsv"),
            os.path.join(opts.output_dir, 'test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='annotation JSON')
    parser.add_argument('--output_dir', required=True, help='annotation JSON')
    parser.add_argument('--cores', type=int, default=os.cpu_count(), help='JSON config files')
    args = parser.parse_args()
    main(args)
