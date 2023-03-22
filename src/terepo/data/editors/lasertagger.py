from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
from typing import List

from transformers.utils import logging

from terepo.data.editors import Operations
from terepo.data.editors.base import load_vocab

logger = logging.get_logger(__name__)


class TagType(Enum):
    """Base tag which indicates the type of an edit operation."""
    # Keep the tagged token.
    KEEP = 1
    # Delete the tagged token.
    DELETE = 2
    # Keep the tagged token but swap the order of sentences. This tag is only
    # applied if there are two source texts and the tag is applied to the last
    # token of the first source. In other contexts, it's treated as KEEP.
    SWAP = 3


class Tag(object):
    """Tag that corresponds to a token edit operation.

    Attributes:
      tag_type: TagType of the tag.
      added_phrase: A phrase that's inserted before the tagged token (can be
        empty).
    """

    def __init__(self, tag):
        """Constructs a Tag object by parsing tag to tag_type and added_phrase.

        Args:
          tag: String representation for the tag which should have the following
            format "<TagType>|<added_phrase>" or simply "<TagType>" if no phrase
            is added before the tagged token. Examples of valid tags include "KEEP",
            "DELETE|and", and "SWAP|.".

        Raises:
          ValueError: If <TagType> is invalid.
        """
        if '|' in tag:
            pos_pipe = tag.index('|')
            tag_type, added_phrase = tag[:pos_pipe], tag[pos_pipe + 1:]
        else:
            tag_type, added_phrase = tag, ''
        try:
            self.tag_type = TagType[tag_type]
        except KeyError:
            raise ValueError(
                'TagType should be KEEP, DELETE or SWAP, not {}'.format(tag_type))
        self.added_phrase = added_phrase

    def __str__(self):
        if not self.added_phrase:
            return self.tag_type.name
        else:
            return '{}|{}'.format(self.tag_type.name, self.added_phrase)


def get_phrase_vocabulary_from_label_map(label_map):
    """Extract the set of all phrases from label map.

    Args:
      label_map: Mapping from tags to tag IDs.

    Returns:
      Set of all phrases appearing in the label map.
    """
    phrase_vocabulary = set()
    for label in label_map.keys():
        tag = Tag(label)
        if tag.added_phrase:
            phrase_vocabulary.add(tag.added_phrase)
    return phrase_vocabulary


class LaserTaggerEditor:

    def __init__(self,
                 labels_vocab_file,
                 verb_form_vocab_file=None,
                 labels_keep="$KEEP",
                 labels_delete="$DELETE",
                 labels_swap="$SWAP",
                 use_swap_tag=False
                 ):
        self.labels_vocab = load_vocab(labels_vocab_file)
        self.labels_keep = labels_keep
        self.labels_delete = labels_delete

        self._phrase_vocabulary = get_phrase_vocabulary_from_label_map(self.labels_vocab)
        # Maximum number of tokens in an added phrase (inferred from the
        # vocabulary).
        self._max_added_phrase_length = 0
        # Set of tokens that are part of a phrase in self.phrase_vocabulary.
        self._token_vocabulary = set()
        for phrase in self._phrase_vocabulary:
            tokens = phrase.split()
            self._token_vocabulary |= set(tokens)
            if len(tokens) > self._max_added_phrase_length:
                self._max_added_phrase_length = len(tokens)

        self.use_swap_tag = use_swap_tag

    def _compute_tags_fixed_order(self, source_tokens, target_tokens):
        """Computes tags when the order of sources is fixed.

        Args:
          source_tokens: List of source tokens.
          target_tokens: List of tokens to be obtained via edit operations.

        Returns:
          List of tagging.Tag objects. If the source couldn't be converted into the
          target via tagging, returns an empty list.
        """
        tags = [Tag('DELETE') for _ in source_tokens]
        # Indices of the tokens currently being processed.
        source_token_idx = 0
        target_token_idx = 0
        while target_token_idx < len(target_tokens):
            tags[source_token_idx], target_token_idx = self._compute_single_tag(
                source_tokens[source_token_idx], target_token_idx, target_tokens)
            # If we're adding a phrase and the previous source token(s) were deleted,
            # we could add the phrase before a previously deleted token and still get
            # the same realized output. For example:
            #    [DELETE, DELETE, KEEP|"what is"]
            # and
            #    [DELETE|"what is", DELETE, KEEP]
            # Would yield the same realized output. Experimentally, we noticed that
            # the model works better / the learning task becomes easier when phrases
            # are always added before the first deleted token. Also note that in the
            # current implementation, this way of moving the added phrase backward is
            # the only way a DELETE tag can have an added phrase, so sequences like
            # [DELETE|"What", DELETE|"is"] will never be created.
            if tags[source_token_idx].added_phrase:
                first_deletion_idx = self._find_first_deletion_idx(
                    source_token_idx, tags)
                if first_deletion_idx != source_token_idx:
                    tags[first_deletion_idx].added_phrase = (
                        tags[source_token_idx].added_phrase)
                    tags[source_token_idx].added_phrase = ''
            source_token_idx += 1
            if source_token_idx >= len(tags):
                break

        # If all target tokens have been consumed, we have found a conversion and
        # can return the tags. Note that if there are remaining source tokens, they
        # are already marked deleted when initializing the tag list.
        if target_token_idx >= len(target_tokens):
            return tags
        return []

    def _compute_single_tag(
            self, source_token, target_token_idx,
            target_tokens):
        """Computes a single tag.

        The tag may match multiple target tokens (via tag.added_phrase) so we return
        the next unmatched target token.

        Args:
          source_token: The token to be tagged.
          target_token_idx: Index of the current target tag.
          target_tokens: List of all target tokens.

        Returns:
          A tuple with (1) the computed tag and (2) the next target_token_idx.
        """
        source_token = source_token.lower()
        target_token = target_tokens[target_token_idx].lower()
        if source_token == target_token:
            # edit = Operations(start=target_token_idx,
            #                              end=target_token_idx + 1,
            #                              operations=self.labels_keep)
            return Tag('KEEP'), target_token_idx + 1
            # return edit, target_token_idx + 1

        added_phrase = ''
        for num_added_tokens in range(1, self._max_added_phrase_length + 1):
            if target_token not in self._token_vocabulary:
                break
            added_phrase += (' ' if added_phrase else '') + target_token
            next_target_token_idx = target_token_idx + num_added_tokens
            if next_target_token_idx >= len(target_tokens):
                break
            target_token = target_tokens[next_target_token_idx].lower()
            if (source_token == target_token and
                    added_phrase in self._phrase_vocabulary):
                return Tag('KEEP|' + added_phrase), next_target_token_idx + 1
        return Tag('DELETE'), target_token_idx

    def _find_first_deletion_idx(self, source_token_idx, tags):
        """Finds the start index of a span of deleted tokens.

        If `source_token_idx` is preceded by a span of deleted tokens, finds the
        start index of the span. Otherwise, returns `source_token_idx`.

        Args:
          source_token_idx: Index of the current source token.
          tags: List of tags.

        Returns:
          The index of the first deleted token preceding `source_token_idx` or
          `source_token_idx` if there are no deleted tokens right before it.
        """
        # Backtrack until the beginning of the tag sequence.
        for idx in range(source_token_idx, 0, -1):
            if tags[idx - 1].tag_type != TagType.DELETE:
                return idx
        return 0

    def convert_sequences_to_edits(self, source_tokens, target_tokens):
        tags = self._compute_tags_fixed_order(source_tokens, target_tokens)
        # If conversion fails, try to obtain the target after swapping the source
        # order.
        # if not tags and len(task.sources) == 2 and self.use_swap_tag:
        #     swapped_task = EditingTask(task.sources[::-1])
        #     tags = self._compute_tags_fixed_order(swapped_task.source_tokens,
        #                                           target_tokens)
        #     if tags:
        #         tags = (tags[swapped_task.first_tokens[1]:] +
        #                 tags[:swapped_task.first_tokens[1]])
        #         # We assume that the last token (typically a period) is never deleted,
        #         # so we can overwrite the tag_type with SWAP (which keeps the token,
        #         # moving it and the sentence it's part of to the end).
        #         tags[task.first_tokens[1] - 1].tag_type = TagType.SWAP
        return tags

    def convert_labels_list_into_edits(self, labels_list: List[List[str]]):
        all_edits = []
        for i, labels in enumerate(labels_list):
            if labels[0] == self.labels_keep:
                continue
            else:
                edit = Operations(start=i, end=i + 1, operations=labels)
                all_edits.append(edit)
        return all_edits

    def convert_edits_to_sentence(self, source_tokens, edits):
        """Realize output text based on the source tokens and predicted tags.

        Args:
          tags: Predicted tags (one for each token in `self.source_tokens`).

        Returns:
          The realizer output text.

        Raises:
          ValueError: If the number of tags doesn't match the number of source
            tokens.
        """
        if len(tags) != len(self.source_tokens):
            raise ValueError('The number of tags ({}) should match the number of '
                             'source tokens ({})'.format(
                len(tags), len(self.source_tokens)))
        outputs = []  # Realized sources that are joined into the output text.
        if (len(self.first_tokens) == 2 and
                tags[self.first_tokens[1] - 1].tag_type == TagType.SWAP):
            order = [1, 0]
        else:
            order = range(len(self.first_tokens))

        for source_idx in order:
            # Get the span of tokens for the source: [first_token, last_token).
            first_token = self.first_tokens[source_idx]
            if source_idx + 1 < len(self.first_tokens):
                last_token = self.first_tokens[source_idx + 1]  # Not inclusive.
            else:
                last_token = len(self.source_tokens)
            # Realize the source and fix casing.
            realized_source = self._realize_sequence(
                self.source_tokens[first_token:last_token],
                tags[first_token:last_token])
            if outputs:
                if outputs[0][-1:] in '.!?':
                    realized_source = self._first_char_to_upper(realized_source)
                else:
                    # Note that ideally we should also test here whether the first word is
                    # a proper noun or an abbreviation that should always be capitalized.
                    realized_source = self._first_char_to_lower(realized_source)
            outputs.append(realized_source)
        return ' '.join(outputs)
