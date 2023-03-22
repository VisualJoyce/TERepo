import collections
import re

pattern_zh = re.compile(
    r'(['
    # 韩文字母：1100-11FF
    # 杂项符号（非CJK专用）：2600-26FF
    # 装饰符号（非CJK专用）：2700-27BF
    # 盲文符号：2800-28FF
    r'\u2F00-\u2FDF'  # 康熙部首：2F00-2FDF
    r'\u2E80-\u2EFF'  # CJK部首补充：2E80-2EFF
    r'\u2FF0-\u2FFF'  # 汉字结构描述字符：2FF0-2FFF
    r'\u3000-\u303F'  # CJK标点符号：3000-303F
    # 日文平假名：3040-309F
    # 日文片假名：30A0-30FF
    r'\u3100-\u312F'  # 注音符号：3100-312F
    # 韩文兼容字母：3130-318F
    r'\u31A0-\u31BF'  # 注音符号（闽南语、客家语扩展）：31A0-31BF
    r'\u31C0-\u31EF'  # CJK笔划：31C0-31EF
    # 日文片假名拼音扩展：31F0-31FF
    r'\u3200-\u32FF'  # CJK字母及月份：3200-32FF
    r'\u3300-\u33FF'  # CJK特殊符号（日期合并）：3300-33FF
    r'\u3400-\u4DB5'  # CJK Unified Ideographs Extension A ： 3400-4DBF
    r'\u4DC0-\u4DFF'  # 易经六十四卦象：4DC0-4DFF
    r'\u4E00-\u9FCB'  # CJK Unified Ideographs ： 4E00-9FFF
    # 彝文音节：A000-A48F
    # 彝文部首：A490-A4CF
    # 韩文拼音：AC00-D7AF
    # E000 — F8FF  	Private Use Area
    # r'\uE400-\uE5E8'
    # r'\uE600-\uE6CF'
    # r'\uE815-\uE86F'
    r'\uF900-\uFAD9'  # CJK Compatibility Ideographs : F900-FAFF
    # Alphabetic Presentation Forms: FB00-FB4F
    r'\uFE10-\uFE1F'  # 中文竖排标点：FE10-FE1F
    r'\uFE30-\uFE4F'  # CJK兼容符号（竖排变体、下划线、顿号）：FE30-FE4F
    r'\uFF00-\uFFEF'  # 全角ASCII、全角中英文标点、半宽片假名、半宽平假名、半宽韩文字母
    # 太玄经符号：1D300-1D35F
    r'\U00020000-\U0002A6D6'  # CJK Unified Ideographs Extension B : 20000 — 2A6DF  	
    r'\U0002A700-\U0002B734'  # CJK Unified Ideographs Extension C (2A700-2B734)
    r'\U0002B740-\U0002B81D'  # CJK Unified Ideographs Extension D (2B840-2B81D)
    r'\U0002F800-\U0002FA1D'  # CJK Compatibility Ideographs Supplement: 2F800 — 2FA1F
    r'])+'
)

REPLACEMENTS = {
    "''": '"',
    '--': '—',
    '`': "'",
    "'ve": "' ve",
}

special_tokens = {"n't": "not", "'m": "am", "ca": "can", "Ca": "Can", "wo": "would", "Wo": "Would",
                  "'ll": "will", "'ve": "have"}


def containsNumber(text):
    reg_ex = re.compile(r".*[0-9].*")
    if reg_ex.match(text):
        # print("{} contains numbers".format(text))
        return True
    else:
        return False


def containsMultiCapital(text):
    reg_ex = re.compile(r".*[A-Z].*[A-Z].*")
    if reg_ex.match(text):
        # print("{} conatains multiple capitals".format(text))
        return True
    else:
        return False


def checkAlternateDots(text):
    if text[0] == ".":
        return False
    alt = text[1::2]
    if set(alt) == {'.'}:
        # print("{} contains alternate dots".format(text))
        return True
    else:
        return False


def end_with_dotcom(text):
    if len(text) >= 4 and text[-4:] == ".com":
        # print("{} contains .com in the end".format(text))
        return True
    else:
        return False


def starts_with_www(text):
    reg_ex = re.compile(r"^www\..*")
    if reg_ex.match(text):
        # print("{} starts with www.".format(text))
        return True
    else:
        return False


def contains_slash(text):
    if "/" in text:
        # print("{} contains /".format(text))
        return True
    else:
        return False


def contains_percent(text):
    if "%" in text:
        # print("{} contains %".format(text))
        return True
    else:
        return False


def contains_ampersand(text):
    if "&" in text:
        # print("{} contains &".format(text))
        return True
    else:
        return False


def contains_at_rate(text):
    if "@" in text:
        # print("{} contains @".format(text))
        return True
    else:
        return False


def contains_square_brackets(text):
    if "[" in text or "]" in text:
        # print("{} contains ] or [".format(text))
        return True
    else:
        return False


def last_dot_first_capital(text):
    if len(text) > 1 and text[-1] == "." and text[0].upper() == text[0]:
        # print("{} has dot as last letter and it's first letter is capital".format(text))
        return True
    else:
        return False


def check_smilies(text):
    if text in [":)", ":(", ";)", ":/", ":|"]:
        # print("{} is a smiley".format(text))
        return True
    else:
        return False


def parse(text):
    endpos = len(text)
    start = 0
    for item in pattern_zh.finditer(text):
        if item.start() > start:
            yield (start, item.start()), text[start:item.start()], False
        start = item.end()
        yield item.span(), item.group(0), True
    if endpos > start:
        yield (start, endpos), text[start:endpos], False


def parse_to_segments(context):
    segments = []
    candidates = collections.defaultdict(list)
    zh_idx_list = []
    min_span_len, max_span_len = 999, 0
    for span, text, is_zh in parse(context):
        span_len = span[1] - span[0]
        if is_zh:
            min_span_len = min(min_span_len, span_len)
            max_span_len = max(max_span_len, span_len)
            candidates[span_len].append(len(segments))
            zh_idx_list.append(len(segments))
        segments.append((span, text, is_zh))

    return segments, zh_idx_list, candidates


def convert_sequence_to_tokens(sequence):
    segments, zh_idx_list, candidates = parse_to_segments(sequence)
    tokens = []
    for span, text, is_zh in segments:
        if is_zh:
            tokens.extend(list(text))
        else:
            tokens.extend(text.split())
    return tokens


def convert_tokens_to_string(tokens):
    """Converts a sequence of tokens (string) in a single string."""
    token = tokens.pop(0)
    last_is_zh = pattern_zh.match(token)
    out_string = [token]
    while tokens:
        token = tokens.pop(0)
        cur_is_zh = pattern_zh.match(token)
        if not last_is_zh and not cur_is_zh:
            out_string.append(' ')
            out_string.append(token)
        else:
            out_string.append(token)
        last_is_zh = cur_is_zh

    sent = ''.join(out_string)
    for fr, to in {
        '`': "'",
        "``": '"',
        "''": '"',
    }.items():
        sent = sent.replace(fr, to)
    return sent


def remove_double_tokens(sent):
    tokens = sent.split(' ')
    deleted_idx = []
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1]:
            deleted_idx.append(i + 1)
    if deleted_idx:
        tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
    return ' '.join(tokens)


def normalize(sent):
    sent = remove_double_tokens(sent)
    for fr, to in REPLACEMENTS.items():
        sent = sent.replace(fr, to)
    return sent.lower()
