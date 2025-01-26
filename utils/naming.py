from typing import List


def split_camel_case(string: str) -> List[str]:
    tokens = []
    token = []
    for prev, char, next in zip(' ' + string, string, string[1:] + ' '):
        if _is_camel_case_boundary(prev, char, next):
            if token:
                tokens.append(''.join(token))
            token = [char]
        else:
            token.append(char)
    if token:
        tokens.append(''.join(token))
    return tokens


def _is_camel_case_boundary(prev, char, next):
    if prev.isdigit():
        return not char.isdigit()
    if char.isupper():
        return next.islower() or prev.isalpha() and not prev.isupper()
    return char.isdigit()
