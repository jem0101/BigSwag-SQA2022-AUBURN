# coding: utf-8
from django.conf import settings

from . import bundle

MEDIA_ROOT = settings.MEDIA_ROOT + '/export/'
MEDIA_URL = settings.MEDIA_URL + '/export/'

standards = {
    'iso690': ('icitat', 'iso', u'iso 690/87 - international standard organization'),
    'nbr6023': ('acitat', 'abnt', u'nbr 6023/89 - associação nacional de normas técnicas'),
    'other': ('ocitat', 'other', u'other standard'),
    'vancouv': ('vcitat', 'vanc', u'the vancouver group - uniform requirements for manuscripts submitted to biomedical journals'),
    'apa': ('pcitat', 'apa', u'American Psychological Association'),
}

L10ISSUEMGS = {'en': (u'No section title', u'No Descriptor', u'Health Sciences Descriptors'),
        'es': (u'Sín título de sección', u'Ningun Descriptor', u'Descriptores en Ciencia de la Salud'),
        'pt': (u'Sem título de seção', u'Nenhum Descritor', u'Descritores em Ciência da Saúde')}

issns = {
    'print': 'print_issn',
    'electronic': 'eletronic_issn',
    'eletronic': 'eletronic_issn',
}


class GenerationError(Exception):
    def __init__(self, *args, **kwargs):
        super(GenerationError, self).__init__(*args, **kwargs)


class Automata(object):
    """
    Represents the automata.mds file
    http://ref.scielo.org/2qx6fb
    """

    def __init__(self, journal):
        self._journal = journal

    @property
    def citat(self):
        tags = standards.get(self._journal.editorial_standard, None)
        if not tags:
            return ''

        return tags[0]

    @property
    def norma(self):
        if not self._journal.editorial_standard:
            return ''
        else:
            return self._journal.editorial_standard

    @property
    def norma_acron(self):
        tags = standards.get(self._journal.editorial_standard, None)
        if not tags:
            return ''

        return tags[1]

    @property
    def norma_name(self):
        tags = standards.get(self._journal.editorial_standard, None)
        if not tags:
            return ''

        return tags[2]

    @property
    def issn(self):
        pid_issn_field = self._journal.scielo_issn
        pid_issn = getattr(self._journal, issns[pid_issn_field], None)
        if not pid_issn:
            return ''

        return pid_issn

    @property
    def acron(self):
        return self._journal.acronym.lower()

    def __unicode__(self):
        return u'{0};{1};{2}.amd;tg{3}.amd'.format(self.issn,
            self.citat, self.acron, self.norma_acron)


class Issue(object):

    MONTHS = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    def __init__(self, issue):
        self._issue = issue

    @property
    def legend(self):
        return u'{0} v.{1} n.{2}'.format(self._issue.journal.short_title,
                                         unicode(self._issue.volume),
                                         unicode(self._issue.identification))

    @property
    def period(self):
        '''
        O período deve ser o especificado pelo PMC.
        EX.: Apr/Jun ou Apr/ ou /Jun
        '''
        return '%s/%s' % (self.MONTHS.get(self._issue.publication_start_month, ''),
             self.MONTHS.get(self._issue.publication_end_month, ''))

    @property
    def order(self):
        '''
        A propriedade order deve ser o publication_year concatenado com o order.
        '''
        return str(self._issue.publication_year) + str(self._issue.order)

    def __unicode__(self):
        rows = u'\r\n'.join([self.legend, self.period, self.order, u'', u''])
        return rows


class L10nIssue(Automata, Issue):

    def __init__(self, journal, issue, language):
        self._journal = journal
        self._issue = issue
        self._language = language

    @property
    def abbrev_title(self):
        return self._issue.journal.title_iso

    @property
    def short_title(self):
        return self._issue.journal.short_title

    @property
    def volume(self):
        v = self._issue.volume
        return unicode(v) if v else u''

    @property
    def number(self):
        v = self._issue.number
        return unicode(v) if v else u''

    @property
    def suppl_volume(self):
        if self._issue.type == 'supplement' and not self._issue.number:
            return unicode(self._issue.suppl_text)
        else:
            return u''

    @property
    def suppl_number(self):
        if self._issue.type == 'supplement' and self._issue.number:
            return unicode(self._issue.suppl_text)
        else:
            return u''

    @property
    def date_iso(self):
        try:
            month = u'%02d' % self._issue.publication_end_month
        except TypeError:
            month = u'00'
        year = unicode(self._issue.publication_year)

        if year:
            return year + month + u'00'
        else:
            return u''

    @property
    def status(self):
        # placebo
        return u'1'

    @property
    def issue_meta(self):
        return u';'.join([
            self.short_title,
            self.volume,
            self.suppl_volume,
            self.number,
            self.suppl_number,
            self.date_iso,
            self.issn,
            self.status,
        ])

    @property
    def sections(self):
        sections = ';'.join([unicode(section) for section in self._issue.section.available(True).all()])
        return sections + u';' + L10ISSUEMGS[self._language][0] if sections else L10ISSUEMGS[self._language][0]

    @property
    def sections_ids(self):
        ids = ';'.join([unicode(section.actual_code) for section in self._issue.section.available(True).all()])
        return ids + u';nd' if ids else u'nd'

    @property
    def ctrl_vocabulary(self):
        '''
        O vocabulário controlado deve ser traduzido
        '''
        if self._issue.journal.ctrl_vocabulary == 'decs':
            return L10ISSUEMGS[self._language][2]
        else:
            return L10ISSUEMGS[self._language][1]

    def __unicode__(self):
        rows = u'\r\n'.join([
            self.legend,
            self.issue_meta,
            self.sections,
            self.sections_ids,
            self.ctrl_vocabulary,
            self.norma_name,
            u'',
        ])
        return rows


class JournalStandard(L10nIssue):

    def __init__(self, journal, issue):
        self._journal = journal
        self._issue = issue

    @property
    def pub_type(self):
        issns = {
            'print': u'ppub',
            'electronic': u'epub',
            'eletronic': u'epub',
        }
        return issns[self._journal.scielo_issn]

    @property
    def study_area(self):
        return '/'.join((area.study_area for area in self._journal.study_areas.all()))

    @property
    def medline_title(self):
        return unicode(self._journal.medline_title)

    @property
    def medline_code(self):
        return unicode(self._journal.medline_code)

    @property
    def pissn(self):
        return unicode(self._journal.print_issn)

    @property
    def eissn(self):
        return unicode(self._journal.eletronic_issn)

    @property
    def publisher(self):
        return unicode(self._journal.publisher_name)

    @property
    def title(self):
        return unicode(self._journal.title)

    @property
    def journal_meta(self):
        return '#'.join([
            self.issn,
            self.short_title,
            self.norma,
            self.pub_type,
            self.issn,
            self.study_area,
            self.medline_title,
            self.medline_code,
            self.title,
            self.acron.lower(),
            self.pissn,
            self.eissn,
            self.publisher,
            ])

    def __unicode__(self):
        return self.journal_meta


def generate(journal, issue):
    export_automata = Automata(journal)
    export_issue = Issue(issue)
    export_l10n_issue_en = L10nIssue(journal, issue, 'en')
    export_l10n_issue_pt = L10nIssue(journal, issue, 'pt')
    export_l10n_issue_es = L10nIssue(journal, issue, 'es')
    export_journal_standard = JournalStandard(journal, issue)

    try:
        packmeta = [
            ('automata.mds', unicode(export_automata)),
            ('issue.mds', unicode(export_issue)),
            ('en_issue.mds', unicode(export_l10n_issue_en)),
            ('es_issue.mds', unicode(export_l10n_issue_es)),
            ('pt_issue.mds', unicode(export_l10n_issue_pt)),
            ('journal-standard.txt', unicode(export_journal_standard)),
        ]
    except AttributeError as exc:
        raise GenerationError('it was impossible to generate the package for %s. %s' % (journal.pk, exc))
    else:
        pkg = bundle.Bundle(*packmeta)

    pkg_filename = bundle.generate_filename('markupfiles', filetype='zip')

    pkg.deploy(MEDIA_ROOT + pkg_filename)
    return MEDIA_URL + pkg_filename


class Ahead(object):

    def __init__(self, journal, year):
        self._journal = journal
        self._year = year

    @property
    def legend(self):
        return u'{0} n.ahead'.format(self._journal.title_iso)

    @property
    def period(self):
        '''
        O período deve ser o especificado pelo PMC.
        EX.: Apr/Jun ou Apr/ ou /Jun
        '''
        return '%s/%s' % ('', '')

    @property
    def year(self):
        return self._year

    @property
    def order(self):
        '''
        A propriedade order deve ser o publication_year concatenado com o order.
        '''
        return str(self.year) + '50'

    def __unicode__(self):
        rows = '\r\n'.join([self.legend, self.period, self.order, '', ''])
        return rows


class L10nAhead(Ahead):

    def __init__(self, journal, year, language):
        self._journal = journal
        self._year = year
        self._language = language

    @property
    def short_title(self):
        return self._journal.short_title

    @property
    def title_ahead(self):
        return self.short_title + ' n.ahead ' + self._year

    @property
    def date_iso(self):
        year = unicode(self._year)

        if year:
            return year + u'0000'
        else:
            return u''

    @property
    def status(self):
        return u'1'

    @property
    def issn(self):
        pid_issn_field = self._journal.scielo_issn
        pid_issn = getattr(self._journal, issns[pid_issn_field], None)
        if not pid_issn:
            return ''

        return pid_issn

    @property
    def sections(self):
        return L10ISSUEMGS[self._language][0]

    @property
    def sections_ids(self):
        return 'nd'

    @property
    def ctrl_vocabulary(self):
        '''
        O vocabulário controlado deve ser traduzido
        '''
        if self._journal.ctrl_vocabulary == 'decs':
            return L10ISSUEMGS[self._language][2]
        else:
            return L10ISSUEMGS[self._language][1]

    @property
    def norma_name(self):
        tags = standards.get(self._journal.editorial_standard, None)
        if not tags:
            return ''

        return tags[2]

    @property
    def ahead_meta(self):
        return u';'.join([
            self.short_title,
            '',
            '',
            'ahead',
            '',
            self.date_iso,
            self.issn,
            self.status,
        ])

    def __unicode__(self):
        rows = u'\r\n'.join([
            self.title_ahead,
            self.ahead_meta,
            self.sections,
            self.sections_ids,
            self.ctrl_vocabulary,
            self.norma_name,
            u'',
        ])
        return rows


class JournalStandardAhead(L10nAhead):

    def __init__(self, journal):
        self._journal = journal

    @property
    def pub_type(self):
        issns = {
            'print': u'ppub',
            'electronic': u'epub',
            'eletronic': u'epub',
        }
        return issns[self._journal.scielo_issn]

    @property
    def study_area(self):
        return '/'.join((area.study_area for area in self._journal.study_areas.all()))

    @property
    def medline_title(self):
        return unicode(self._journal.medline_title)

    @property
    def medline_code(self):
        return unicode(self._journal.medline_code)

    @property
    def pissn(self):
        return unicode(self._journal.print_issn)

    @property
    def eissn(self):
        return unicode(self._journal.eletronic_issn)

    @property
    def publisher(self):
        return unicode(self._journal.publisher_name)

    @property
    def title(self):
        return unicode(self._journal.title)

    @property
    def acron(self):
        return self._journal.acronym.lower()

    @property
    def norma(self):
        if not self._journal.editorial_standard:
            return ''
        else:
            return self._journal.editorial_standard

    @property
    def journal_meta(self):
        return '#'.join([
            self.issn,
            self.short_title,
            self.norma,
            self.pub_type,
            self.issn,
            self.study_area,
            self.medline_title,
            self.medline_code,
            self.title,
            self.acron,
            self.pissn,
            self.eissn,
            self.publisher,
            ])

    def __unicode__(self):
        return self.journal_meta


def generate_ahead(journal_id, year):
    from journalmanager import models
    journal = models.Journal.objects.get(id=journal_id)

    export_automata = Automata(journal)
    export_ahead = Ahead(journal, year)
    export_l10n_issue_en = L10nAhead(journal, year, 'en')
    export_l10n_issue_pt = L10nAhead(journal, year, 'pt')
    export_l10n_issue_es = L10nAhead(journal, year, 'es')
    export_journal_standard = JournalStandardAhead(journal)

    try:
        packmeta = [
            ('automata.mds', unicode(export_automata)),
            ('issue.mds', unicode(export_ahead)),
            ('en_issue.mds', unicode(export_l10n_issue_en)),
            ('es_issue.mds', unicode(export_l10n_issue_es)),
            ('pt_issue.mds', unicode(export_l10n_issue_pt)),
            ('journal-standard.txt', unicode(export_journal_standard)),
        ]
    except AttributeError as exc:
        raise GenerationError('it was impossible to generate the package for %s. %s' % (journal.pk, exc))
    else:
        pkg = bundle.Bundle(*packmeta)

    pkg_filename = bundle.generate_filename('markupfiles', filetype='zip')

    pkg.deploy(MEDIA_ROOT + pkg_filename)
    return MEDIA_URL + pkg_filename
