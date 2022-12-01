import subprocess
import re

token_re = re.compile('\S+')

punctuation_deletions = [u"'"]
punctuation_elisions = [u'-', u',', u'.', u',', u';', u':', u'|', u'&']

punctuation_translations = dict(
    [(ord(char), None) for char in punctuation_deletions] +
    [(ord(char), u' ') for char in punctuation_elisions])

whitespace_unicode_translations = {ord('\t'): u' ', ord('\n'): u' ', ord('\r'): u''}


def UpperCamelCase(name):
    return re.sub('(^|-|_)(.)', lambda g: g.group(2).upper(), name)


def underscore(name):
    return re.sub('([A-Z]+)', r'_\1', name).strip('_').lower()


def singular(name):
    return re.sub('s$', '', name)


def utf8str(s):
    if isinstance(s, unicode):
        return s.encode('utf8')
    return s


def zip_boundaries(xs, space_len=1):
    '''Take a list of strings and iterate through them along with boundary indices.

    >>> tokens = 'Into the void .'.split()
    >>> list(zip_boundaries(tokens))
    [('Into', 0, 4), ('the', 5, 8), ('void', 9, 13), ('.', 14, 15)]
    '''
    start = 0
    for x in xs:
        x_len = len(x)
        yield x, start, start + x_len
        start += x_len + space_len


def gloss(alignments, prefixes=None, postfixes=None, width=None, toksep=' ', linesep='\n', groupsep='\n'):
    '''
    Creates an interlinear gloss.

    Take a list of [('a', 'DET'), ('beluga', 'N')] and return a string covering multiples lines, like:
        a   beluga
        DET N
    each item in `alignments` should have the same length, N
    `prefixes`, if provided, should be N-long
    `postfixes`, if provided, should be N-long
    '''
    if width is None:
        width = int(subprocess.check_output(['tput', 'cols']))
    toksep_len = len(toksep)

    # a "group" is a N-line string, each line of which is at most `width` characters
    # `groups` is a list of such groups
    groups = []

    def flush_buffer(line_buffer):
        if len(line_buffer) > 0:
            lines = [toksep.join(tokens) for tokens in line_buffer]
            if prefixes:
                lines = [prefix + line for prefix, line in zip(prefixes, lines)]
            if postfixes:
                lines = [line + postfix for postfix, line in zip(postfixes, lines)]
            groups.append(linesep.join(lines))
        return [[] for _ in alignments[0]]

    # the line_buffer is an N-long list of lists of tokens (strings)
    # [[e1, e2, e3], [f1, f2, f3], [g1, g2, g3]]
    line_buffer = flush_buffer([])
    # the line_buffer_width is just the cumulative width of the current line_buffer
    line_buffer_width = 0

    for aligned in alignments:
        aligned = map(str, aligned)
        length = max(map(len, aligned))
        line_buffer_width += toksep_len + length
        if line_buffer_width >= width:
            line_buffer = flush_buffer(line_buffer)
            line_buffer_width = length
        for i, token in enumerate(aligned):
            line_buffer[i].append(token.ljust(length))

    flush_buffer(line_buffer)

    return groupsep.join(groups)
