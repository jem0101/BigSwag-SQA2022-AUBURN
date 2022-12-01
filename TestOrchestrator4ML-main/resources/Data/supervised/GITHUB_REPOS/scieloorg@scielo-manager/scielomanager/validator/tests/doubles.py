import packtools

# packtools.stylechecker double


class XMLValidatorDouble(object):
    @classmethod
    def parse(cls, file, dtd=None, no_doctype=False):
        pass

    def validate_all(self, fail_fast=False):
        return True, None

    @property
    def meta(self):
        return {
            'article_title': 'HIV/AIDS knowledge among men who have sex with men: applying the item response theory',
            'issue_year': '2014',
            'journal_title': u'Revista de Sa\xfade P\xfablica',
            'journal_pissn': '0034-8910',
            'journal_eissn': '1518-8787',
            'issue_number': '2',
            'issue_volume': '48'
        }


class XMLValidatorAnnotationsDouble(XMLValidatorDouble):
    def annotate_errors(self, fail_fast=False):
        return "some annotations in xml string"

    def validate_all(self, fail_fast=False):
        error_list = []

        class DummyError(object):
            line = 1
            column = 6
            message = u'Premature end of data in tag xml line 1, line 1, column 6'
            level_name = 'ERROR'

        for x in xrange(0, 6):
            error_list.append(DummyError())

        return False, error_list


# ------------------
# utils.analyze_xml
# ------------------
def make_stub_analyze_xml(type):
    """Factory for utils.analyze_xml stub functions.

    `type` can be: 'valid' or 'invalid'.
    """
    sample_xml = "<article>foo</article>"
    sample_meta = {}

    if type == 'valid':
        result = {
            'annotations': sample_xml,
            'validation_errors': None,
            'meta': sample_meta,
        }
        err = None
    elif type == 'invalid':
        result = {
            'annotations': sample_xml,
            'validation_errors': [],
            'meta': sample_meta,
        }
        err = None
    elif type == 'throw_io_error':
        result = None
        err = IOError(u'Error reading file foo.xml')
    elif type == 'syntax_error':
        result = None
        err = Exception(u'Premature end of data in tag unclosed_tag line 5, line 5, column 17')
    else:
        raise ValueError('Unknown type value')

    def stub__analyze_xml(f, extra_schematron=None):
        return result, err

    return stub__analyze_xml
