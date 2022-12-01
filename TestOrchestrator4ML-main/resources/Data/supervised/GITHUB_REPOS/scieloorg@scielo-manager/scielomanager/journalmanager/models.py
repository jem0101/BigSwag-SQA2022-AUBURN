# -*- encoding: utf-8 -*-
import urllib
import hashlib
import logging
import choices
from pytz import all_timezones
from scielomanager import tools
import datetime
from uuid import uuid4
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from django.db import (
    models,
    IntegrityError,
    DatabaseError,
    )
from django.db.models import Q
from django.core.exceptions import ObjectDoesNotExist, ImproperlyConfigured
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes import generic
from django.contrib.auth.models import User
from django.utils.translation import ugettext_lazy as _
from django.utils.translation import ugettext as __
from django.conf import settings
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.template.defaultfilters import slugify
from scielo_extensions import modelfields
from tastypie.models import create_api_key
import celery
from PIL import Image

from scielomanager.utils import base28
from scielomanager.custom_fields import (
        ContentTypeRestrictedFileField,
        XMLSPSField,
)
from . import modelmanagers


logger = logging.getLogger(__name__)

LINKABLE_ARTICLE_TYPES = ['correction', ]
EVENT_TYPES = [(ev_type, ev_type) for ev_type in ['added', 'deleted', 'updated']]
ISSUE_DEFAULT_LICENSE_HELP_TEXT = _(u"If not defined, will be applied the related journal's use license. \
The SciELO default use license is BY-NC. Please visit: http://ref.scielo.org/jf5ndd (5.2.11. Política de direitos autorais) for more details.")


def get_user_collections(user_id):
    """
    Return all the collections of a given user, The returned collections are the collections where the
    user could have access by the collections bar.
    """
    user_collections = User.objects.get(pk=user_id).usercollections_set.all().order_by(
        'collection__name')

    return user_collections


def get_journals_default_use_license():
    """
    Returns the default use license for all new Journals.

    This callable is passed as the default value on Journal.use_license field.
    The default use license is the one defined on SciELO criteria, and at
    the time is BY-NC. See http://ref.scielo.org/jf5ndd for more information.
    """
    try:
        return UseLicense.objects.get(is_default=True)
    except UseLicense.DoesNotExist:
        raise ImproperlyConfigured("There is no UseLicense set as default")


class AppCustomManager(models.Manager):
    """
    Domain specific model managers.
    """

    def available(self, is_available=True):
        """
        Filter the queryset based on its availability.
        """
        data_queryset = self.get_query_set()

        if not isinstance(is_available, bool):
            try:
                if int(is_available) == 0:
                    is_available = False
                else:
                    is_available = True
            except (ValueError, TypeError):
                is_available = True

        data_queryset = data_queryset.filter(is_trashed=not is_available)

        return data_queryset


class JournalCustomManager(AppCustomManager):

    def all_by_user(self, user, is_available=True, pub_status=None):
        """
        Retrieves all the user's journals, contextualized by
        their default collection.
        """
        objects_all = Journal.userobjects.all()
        if is_available:
            objects_all = objects_all.available().distinct()
        else:
            objects_all = objects_all.unavailable().distinct()

        if pub_status:
            if pub_status in [stat[0] for stat in choices.JOURNAL_PUBLICATION_STATUS]:
                objects_all = objects_all.filter(pub_status=pub_status)

        return objects_all

    def recents_by_user(self, user):
        """
        Retrieves the recently modified objects related to the given user.
        """
        default_collection = Collection.userobjects.active()

        recents = self.filter(
            collections=default_collection).distinct().order_by('-updated')[:5]

        return recents

    def all_by_collection(self, collection, is_available=True):
        objects_all = self.available(is_available).filter(
            collections=collection)
        return objects_all

    def by_issn(self, issn):
        """
        Get the journal assigned to `issn`, being electronic or print.

        In some cases more than one instance of the same journal will be
        returned due to the fact that journals present in more than one
        collection is handled separately.
        """
        if issn == '':
            return Journal.objects.none()

        journals = Journal.objects.filter(
            models.Q(print_issn=issn) | models.Q(eletronic_issn=issn)
        )

        return journals


class SectionCustomManager(AppCustomManager):

    def all_by_user(self, user, is_available=True):
        default_collection = Collection.objects.get_default_by_user(user)

        objects_all = self.available(is_available).filter(
            journal__collections=default_collection).distinct()

        return objects_all


class IssueCustomManager(AppCustomManager):

    def all_by_collection(self, collection, is_available=True):
        objects_all = self.available(is_available).filter(
            journal__collections=collection)

        return objects_all


class InstitutionCustomManager(AppCustomManager):
    """
    Add capabilities to Institution subclasses to retrieve querysets
    based on user's collections.
    """
    def all_by_user(self, user, is_available=True):
        default_collection = Collection.objects.get_default_by_user(user)

        objects_all = self.available(is_available).filter(
            collections__in=[default_collection]).distinct()

        return objects_all


class CollectionCustomManager(AppCustomManager):

    def all_by_user(self, user):
        """
        Returns all the Collections related to the given
        user.
        """
        collections = self.filter(usercollections__user=user).order_by(
            'name')

        return collections

    def get_default_by_user(self, user):
        """
        Returns the Collection marked as default by the given user.
        If none satisfies this condition, the first
        instance is then returned.

        Like any manager method that does not return Querysets,
        `get_default_by_user` raises DoesNotExist if there is no
        result for the given parameter.
        """
        collections = self.filter(
            usercollections__user=user,
            usercollections__is_default=True).order_by('name')

        if not collections.count():
            try:
                collection = self.all_by_user(user)[0]
            except IndexError:
                raise Collection.DoesNotExist()
            else:
                collection.make_default_to_user(user)
                return collection

        return collections[0]

    def get_managed_by_user(self, user):
        """
        Returns all collections managed by a given user.
        """
        collections = self.filter(
            usercollections__user=user,
            usercollections__is_manager=True).order_by('name')

        return collections


class RegularPressReleaseCustomManager(models.Manager):

    def by_journal_pid(self, journal_pid):
        """
        Returns all PressReleases related to a Journal, given its
        PID.
        """
        journals = Journal.objects.filter(
            models.Q(print_issn=journal_pid) | models.Q(eletronic_issn=journal_pid))

        preleases = self.filter(issue__journal__in=journals.values('id')).select_related('translations')
        return preleases

    def all_by_journal(self, journal):
        """
        Returns all PressReleases related to a Journal
        """
        preleases = self.filter(issue__journal=journal)
        return preleases

    def by_issue_pid(self, issue_pid):
        """
        Returns all PressReleases related to an Issue, given its
        PID.
        """
        issn_slice = slice(0, 9)
        year_slice = slice(9, 13)
        order_slice = slice(13, None)

        issn = issue_pid[issn_slice]
        year = issue_pid[year_slice]
        order = int(issue_pid[order_slice])

        preleases_qset = self.by_journal_pid(issn)
        return preleases_qset.filter(issue__publication_year=year).filter(issue__order=order)


class AheadPressReleaseCustomManager(models.Manager):

    def by_journal_pid(self, journal_pid):
        """
        Returns all PressReleases related to a Journal, given its
        PID.
        """
        preleases = self.filter(models.Q(journal__print_issn=journal_pid) | models.Q(journal__eletronic_issn=journal_pid))
        return preleases


class Language(models.Model):
    """
    Represents ISO 639-1 Language Code and its language name in English. Django
    automaticaly translates language names, if you write them right.

    http://en.wikipedia.org/wiki/ISO_639-1_language_matrix
    """
    iso_code = models.CharField(_('ISO 639-1 Language Code'), max_length=2)
    name = models.CharField(_('Language Name (in English)'), max_length=64)

    def __unicode__(self):
        return __(self.name)

    class Meta:
        ordering = ['name']


PROFILE_TIMEZONES_CHOICES = zip(all_timezones, all_timezones)


class UserProfile(models.Model):
    user = models.OneToOneField(User)
    email_notifications = models.BooleanField("Want to receive email notifications?", default=True)
    tz = models.CharField("Time Zone",  max_length=150, choices=PROFILE_TIMEZONES_CHOICES, default=settings.TIME_ZONE)

    @property
    def is_editor(self):
        return self.user.groups.filter(name__iexact='Editors').exists()

    @property
    def is_librarian(self):
        return self.user.groups.filter(name__iexact='Librarian').exists()

    @property
    def is_trainee(self):
        return self.user.groups.filter(name__iexact='Trainee').exists()

    @property
    def gravatar_id(self):
        return hashlib.md5(self.user.email.lower().strip()).hexdigest()

    @property
    def avatar_url(self):
        params = urllib.urlencode({'s': 18, 'd': 'mm'})
        return '{0}/avatar/{1}?{2}'.format(getattr(settings, 'GRAVATAR_BASE_URL', 'https://secure.gravatar.com'), self.gravatar_id, params)

    @property
    def get_default_collection(self):
        """
        Return the default collection for this user
        """
        uc = UserCollections.objects.get(user=self.user, is_default=True)
        return uc.collection


class Collection(models.Model):
    objects = models.Manager()  # The default manager.
    userobjects = modelmanagers.CollectionManager()  # Custom manager

    collection = models.ManyToManyField(User, related_name='user_collection', through='UserCollections', null=True, blank=True, )
    name = models.CharField(_('Collection Name'), max_length=128, db_index=True, )
    name_slug = models.SlugField(unique=True, db_index=True, blank=True, null=True)
    url = models.URLField(_('Instance URL'), )
    logo = models.ImageField(_('Logo'), upload_to='img/collections_logos', null=True, blank=True, )
    acronym = models.CharField(_('Sigla'), max_length=16, db_index=True, blank=True, )
    country = models.CharField(_('Country'), max_length=32,)
    state = models.CharField(_('State'), max_length=32, null=False, blank=True,)
    city = models.CharField(_('City'), max_length=32, null=False, blank=True,)
    address = models.TextField(_('Address'),)
    address_number = models.CharField(_('Number'), max_length=8,)
    address_complement = models.CharField(_('Complement'), max_length=128, null=False, blank=True,)
    zip_code = models.CharField(_('Zip Code'), max_length=16, null=True, blank=True, )
    phone = models.CharField(_('Phone Number'), max_length=16, null=False, blank=True, )
    fax = models.CharField(_('Fax Number'), max_length=16, null=False, blank=True, )
    email = models.EmailField(_('Email'), )

    def __unicode__(self):
        return unicode(self.name)

    class Meta:
        ordering = ['name']
        permissions = (("list_collection", "Can list Collections"),)

    def save(self, *args, **kwargs):
        self.name_slug = slugify(self.name)
        super(Collection, self).save(*args, **kwargs)

    def add_user(self, user, is_manager=False):
        """
        Add the user to the current collection.
        If user have not a default collection, then ``self``
        will be the default one
        """
        UserCollections.objects.create(collection=self,
                                       user=user,
                                       is_default=False,
                                       is_manager=is_manager)

        # if user do not have a default collections, make this the default one
        user_has_default_collection = UserCollections.objects.filter(is_default=True, user=user).exists()
        if not user_has_default_collection:
            self.make_default_to_user(user)

    def remove_user(self, user):
        """
        Removes the user from the current collection.
        If the user isn't already related to the given collection,
        it will do nothing, silently.
        """
        try:
            uc = UserCollections.objects.get(collection=self, user=user)
        except UserCollections.DoesNotExist:
            return None
        else:
            uc.delete()

    def make_default_to_user(self, user):
        """
        Makes the current collection, the user's default.
        """
        UserCollections.objects.filter(user=user).update(is_default=False)
        uc, created = UserCollections.objects.get_or_create(
            collection=self, user=user)
        uc.is_default = True
        uc.save()

    def is_default_to_user(self, user):
        """
        Returns a boolean value depending if the current collection
        is set as default to the given user.
        """
        try:
            uc = UserCollections.objects.get(collection=self, user=user)
            return uc.is_default
        except UserCollections.DoesNotExist:
            return False

    def is_managed_by_user(self, user):
        """
        Returns a boolean value depending if the current collection
        is managed by the given user.
        """
        try:
            uc = UserCollections.objects.get(collection=self, user=user)
            return uc.is_manager
        except UserCollections.DoesNotExist:
            return False


class UserCollections(models.Model):
    objects = models.Manager()  # The default manager.
    userobjects = modelmanagers.UserCollectionsManager()  # Custom manager

    user = models.ForeignKey(User)
    collection = models.ForeignKey(Collection)
    is_default = models.BooleanField(_('Is default'), default=False, null=False, blank=False)
    is_manager = models.BooleanField(_('Is manager of the collection?'), default=False, null=False, blank=False)

    class Meta:
        unique_together = ("user", "collection", )


class Institution(models.Model):
    objects = models.Manager()  # The default manager.
    userobjects = modelmanagers.InstitutionManager()  # Custom manager

    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    name = models.CharField(_('Institution Name'), max_length=256, db_index=True)
    complement = models.TextField(_('Institution Complements'), blank=True, default="")
    acronym = models.CharField(_('Sigla'), max_length=16, db_index=True, blank=True)
    country = models.CharField(_('Country'), max_length=32)
    state = models.CharField(_('State'), max_length=32, null=False, blank=True)
    city = models.CharField(_('City'), max_length=32, null=False, blank=True)
    address = models.TextField(_('Address'))
    address_number = models.CharField(_('Number'), max_length=8)
    address_complement = models.CharField(_('Address Complement'), max_length=128, null=False, blank=True)
    zip_code = models.CharField(_('Zip Code'), max_length=16, null=True, blank=True)
    phone = models.CharField(_('Phone Number'), max_length=16, null=False, blank=True)
    fax = models.CharField(_('Fax Number'), max_length=16, null=False, blank=True)
    cel = models.CharField(_('Cel Number'), max_length=16, null=False, blank=True)
    email = models.EmailField(_('E-mail'))
    is_trashed = models.BooleanField(_('Is trashed?'), default=False, db_index=True)

    def __unicode__(self):
        return u'%s' % (self.name)

    class Meta:
        ordering = ['name']


class Sponsor(Institution):
    objects = models.Manager()  # The default manager.
    userobjects = modelmanagers.SponsorManager()  # Custom manager

    collections = models.ManyToManyField(Collection)

    class Meta:
        permissions = (("list_sponsor", "Can list Sponsors"),)


class SubjectCategory(models.Model):
    objects = JournalCustomManager()  # Custom manager

    term = models.CharField(_('Term'), max_length=256, db_index=True)

    def __unicode__(self):
        return self.term


class StudyArea(models.Model):
    study_area = models.CharField(_('Study Area'), max_length=256,
                                  choices=sorted(choices.SUBJECTS, key=lambda SUBJECTS: SUBJECTS[1]))

    def __unicode__(self):
        return self.study_area


class Journal(models.Model):
    """
    Represents a Journal that is managed by one SciELO Collection.

    `editor_address` references the institution who operates the
    process.
    `publisher_address` references the institution who is responsible
    for the Journal.
    """

    # Custom manager
    objects = JournalCustomManager()
    userobjects = modelmanagers.JournalManager()

    # Relation fields
    editor = models.ForeignKey(User, verbose_name=_('Editor'),
            related_name='editor_journal', null=True, blank=True)
    creator = models.ForeignKey(User, related_name='enjoy_creator',
            editable=False)
    sponsor = models.ManyToManyField('Sponsor', verbose_name=_('Sponsor'),
            related_name='journal_sponsor', null=True, blank=True)
    previous_title = models.ForeignKey('Journal', verbose_name=_('Previous title'),
            related_name='prev_title', null=True, blank=True)
    # licença de uso padrão definida pelo editor da revista
    use_license = models.ForeignKey('UseLicense', verbose_name=_('Use license'),
            default=get_journals_default_use_license)
    collections = models.ManyToManyField('Collection', through='Membership')
    # os idiomas que a revista publica conteúdo
    languages = models.ManyToManyField('Language')
    ccn_code = models.CharField(_("CCN Code"), max_length=64, default='',
            blank=True, help_text=_("The code of the journal at the CCN database."))
    # os idiomas que os artigos da revista apresentam as palavras-chave
    abstract_keyword_languages = models.ManyToManyField('Language',
            related_name="abstract_keyword_languages")
    subject_categories = models.ManyToManyField(SubjectCategory,
            verbose_name=_("Subject Categories"), related_name="journals",
            null=True)
    study_areas = models.ManyToManyField(StudyArea, verbose_name=_("Study Area"),
            related_name="journals_migration_tmp", null=True)

    # Fields
    current_ahead_documents = models.IntegerField(
            _('Total of ahead of print documents for the current year'),
            max_length=3, default=0, blank=True)
    previous_ahead_documents = models.IntegerField(
            _('Total of ahead of print documents for the previous year'),
            max_length=3, default=0, blank=True)
    twitter_user = models.CharField(_('Twitter User'), max_length=128,
            default='', blank=True)
    title = models.CharField(_('Journal Title'), max_length=256, db_index=True)
    title_iso = models.CharField(_('ISO abbreviated title'), max_length=256,
            db_index=True)
    short_title = models.CharField(_('Short Title'), max_length=256,
            db_index=True, default='')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    acronym = models.CharField(_('Acronym'), max_length=16, blank=False)
    scielo_issn = models.CharField(_('The ISSN used to build the Journal PID.'),
            max_length=16, choices=sorted(
                choices.SCIELO_ISSN, key=lambda SCIELO_ISSN: SCIELO_ISSN[1]))
    print_issn = models.CharField(_('Print ISSN'), max_length=9, db_index=True)
    eletronic_issn = models.CharField(_('Electronic ISSN'), max_length=9,
            db_index=True)
    subject_descriptors = models.CharField(_('Subject / Descriptors'),
            max_length=1024)
    init_year = models.CharField(_('Initial Year'), max_length=4)
    init_vol = models.CharField(_('Initial Volume'), max_length=16, default='',
            blank=True)
    init_num = models.CharField(_('Initial Number'), max_length=16, default='',
            blank=True)
    final_year = models.CharField(_('Final Year'), max_length=4, default='',
            blank=True)
    final_vol = models.CharField(_('Final Volume'), max_length=16, default='',
            blank=True)
    final_num = models.CharField(_('Final Number'), max_length=16, default='',
            blank=True)
    medline_title = models.CharField(_('Medline Title'), max_length=256,
            default='', blank=True)
    medline_code = models.CharField(_('Medline Code'), max_length=64,
            default='', blank=True)
    frequency = models.CharField(_('Frequency'), max_length=16,
            choices=sorted(
                choices.FREQUENCY, key=lambda FREQUENCY: FREQUENCY[1]))
    editorial_standard = models.CharField(_('Editorial Standard'), max_length=64,
            choices=sorted(choices.STANDARD, key=lambda STANDARD: STANDARD[1]))
    ctrl_vocabulary = models.CharField(_('Controlled Vocabulary'), max_length=64,
            choices=choices.CTRL_VOCABULARY)
    pub_level = models.CharField(_('Publication Level'), max_length=64,
            choices=sorted(
                choices.PUBLICATION_LEVEL, key=lambda PUBLICATION_LEVEL: PUBLICATION_LEVEL[1]))
    secs_code = models.CharField(_('SECS Code'), max_length=64, default='',
            blank=True)
    # detentor dos direitos de uso
    copyrighter = models.CharField(_('Copyrighter'), max_length=254)
    url_online_submission = models.CharField(_('URL of online submission'),
            max_length=128, default='', blank=True)
    url_journal = models.CharField(_('URL of the journal'), max_length=128,
            default='', blank=True)
    notes = models.TextField(_('Notes'), max_length=254, default='', blank=True)
    index_coverage = models.TextField(_('Index Coverage'), default='',
            blank=True)
    cover = ContentTypeRestrictedFileField(_('Journal Cover'),
            upload_to='img/journal_cover/', null=True, blank=True,
            content_types=settings.IMAGE_CONTENT_TYPE,
            max_upload_size=settings.JOURNAL_COVER_MAX_SIZE)
    logo = ContentTypeRestrictedFileField(_('Journal Logo'),
            upload_to='img/journals_logos', null=True, blank=True,
            content_types=settings.IMAGE_CONTENT_TYPE,
            max_upload_size=settings.JOURNAL_LOGO_MAX_SIZE)
    is_trashed = models.BooleanField(_('Is trashed?'), default=False,
            db_index=True)
    other_previous_title = models.CharField(_('Other Previous Title'),
            max_length=255, default='', blank=True)
    editor_name = models.CharField(_('Editor Names'), max_length=512)
    editor_address = models.CharField(_('Editor Address'), max_length=512)
    editor_address_city = models.CharField(_('Editor City'), max_length=256)
    editor_address_state = models.CharField(_('Editor State/Province/Region'),
            max_length=128)
    editor_address_zip = models.CharField(_('Editor Zip/Postal Code'),
            max_length=64)
    editor_address_country = modelfields.CountryField(_('Editor Country'))
    editor_phone1 = models.CharField(_('Editor Phone 1'), max_length=32)
    editor_phone2 = models.CharField(_('Editor Phone 2'), null=True, blank=True,
            max_length=32)
    editor_email = models.EmailField(_('Editor E-mail'))
    publisher_name = models.CharField(_('Publisher Name'), max_length=512)
    publisher_country = modelfields.CountryField(_('Publisher Country'))
    publisher_state = models.CharField(_('Publisher State/Province/Region'),
            max_length=64)
    publication_city = models.CharField(_('Publication City'), max_length=64)
    is_indexed_scie = models.BooleanField(_('SCIE'), default=False)
    is_indexed_ssci = models.BooleanField(_('SSCI'), default=False)
    is_indexed_aehci = models.BooleanField(_('A&HCI'), default=False)

    def __repr__(self):
        return u'<%s pk="%s" acronym="%s">' % (self.__class__.__name__, self.pk,
                self.acronym)

    def __unicode__(self):
        return self.title

    class Meta:
        ordering = ('title', 'id')
        permissions = (("list_journal", "Can list Journals"),
                       ("list_editor_journal", "Can list editor Journal"),
                       ("change_editor", "Can change editor of the journal"))

    def get_last_issue(self):
        """
        Return the latest issue based on descending ordering of parameters:
        ``publication_year``, ``volume`` and ``number``

        Return a issue instance otherwise ``None``
        """
        try:
            return self.issue_set.order_by('-publication_year', '-volume', '-number')[0]
        except IndexError:
            return None

    def issues_as_grid(self, is_available=True):
        objects_all = self.issue_set.available(is_available).order_by(
            '-publication_year', '-volume')

        grid = OrderedDict()

        for issue in objects_all:
            year_node = grid.setdefault(issue.publication_year, OrderedDict())
            volume_node = year_node.setdefault(issue.volume, [])
            volume_node.append(issue)

        for year, volume in grid.items():
            for vol, issues in volume.items():
                issues.sort(key=lambda x: x.order)

        return grid

    @property
    def succeeding_title(self):
        try:
            return self.prev_title.get()
        except ObjectDoesNotExist:
            return None

    def has_issues(self, issues):
        """
        Returns ``True`` if all the given issues are bound to the journal.

        ``issues`` is a list of Issue pk.
        """
        issues_to_test = set(int(issue) for issue in issues)
        bound_issues = set(issue.pk for issue in self.issue_set.all())

        return issues_to_test.issubset(bound_issues)

    @property
    def scielo_pid(self):
        """
        Returns the ISSN used as PID on SciELO public catalogs.
        """

        attr = u'print_issn' if self.scielo_issn == u'print' else u'eletronic_issn'
        return getattr(self, attr)

    def join(self, collection, responsible):
        """Make this journal part of the collection.
        """

        Membership.objects.create(journal=self,
                                  collection=collection,
                                  created_by=responsible,
                                  status='inprogress')

    def membership_info(self, collection, attribute=None):
        """Retrieve info about the relation of this journal with a
        given collection.
        """
        rel = self.membership_set.get(collection=collection)
        if attribute:
            return getattr(rel, attribute)
        else:
            return rel

    def is_member(self, collection):
        """
        Returns a boolean indicating whether or not a member of a specific collection
        """
        return self.membership_set.filter(collection=collection).exists()

    def change_status(self, collection, new_status, reason, responsible):
        rel = self.membership_info(collection)
        rel.status = new_status
        rel.reason = reason
        rel.created_by = responsible
        rel.save()


class Membership(models.Model):
    """
    Represents the many-to-many relation
    between Journal and Collection.
    """
    journal = models.ForeignKey('Journal')
    collection = models.ForeignKey('Collection')
    status = models.CharField(max_length=16, default="inprogress",
        choices=choices.JOURNAL_PUBLICATION_STATUS)
    since = models.DateTimeField(auto_now=True)
    reason = models.TextField(_('Why are you changing the publication status?'),
        blank=True, default="")
    created_by = models.ForeignKey(User, editable=False)

    def save(self, *args, **kwargs):
        """
        Always save a copy at JournalTimeline
        """
        super(Membership, self).save(*args, **kwargs)
        JournalTimeline.objects.create(journal=self.journal,
                                       collection=self.collection,
                                       status=self.status,
                                       reason=self.reason,
                                       created_by=self.created_by,
                                       since=self.since)

    class Meta():
        unique_together = ("journal", "collection")


class JournalTimeline(models.Model):
    """
    Represents the status history of a journal.
    """
    journal = models.ForeignKey('Journal', related_name='statuses')
    collection = models.ForeignKey('Collection')
    status = models.CharField(max_length=16,
        choices=choices.JOURNAL_PUBLICATION_STATUS)
    since = models.DateTimeField()
    reason = models.TextField(_('Reason'), default="")
    created_by = models.ForeignKey(User)


class JournalTitle(models.Model):
    journal = models.ForeignKey(Journal, related_name='other_titles')
    title = models.CharField(_('Title'), null=False, max_length=128)
    category = models.CharField(_('Title Category'), null=False, max_length=128, choices=sorted(choices.TITLE_CATEGORY, key=lambda TITLE_CATEGORY: TITLE_CATEGORY[1]))


class JournalMission(models.Model):
    journal = models.ForeignKey(Journal, related_name='missions')
    description = models.TextField(_('Mission'))
    language = models.ForeignKey('Language', blank=False, null=True)


class UseLicense(models.Model):
    license_code = models.CharField(_('License Code'), unique=True, null=False, blank=False, max_length=64)
    reference_url = models.URLField(_('License Reference URL'), null=True, blank=True)
    disclaimer = models.TextField(_('Disclaimer'), null=True, blank=True, max_length=512)
    is_default = models.BooleanField(_('Is Default?'), default=False)

    def __unicode__(self):
        return self.license_code

    class Meta:
        ordering = ['license_code']

    def save(self, *args, **kwargs):
        """
        Only one UseLicense must be the default (is_default==True).
        If already have one, these will be unset as default (is_default==False)
        If None is already setted, this instance been saved, will be the default.
        If the only one is unsetted as default, then will be foreced to be the default anyway,
        to allways get one license setted as default
        """
        qs = UseLicense.objects.filter(is_default=True)
        if (qs.count() == 0 ) or (self in qs):
            # no other was default, or ``self`` is the current default one,
            # so ``self`` will be set as default
            self.is_default = True

        if self.is_default:
            if self.pk:
                qs = qs.exclude(pk=self.pk)
            if qs.count() != 0:
                qs.update(is_default=False)
        super(UseLicense, self).save(*args, **kwargs)


class TranslatedData(models.Model):
    translation = models.CharField(_('Translation'), null=True, blank=True, max_length=512)
    language = models.CharField(_('Language'), choices=sorted(choices.LANGUAGE, key=lambda LANGUAGE: LANGUAGE[1]), null=False, blank=False, max_length=32)
    model = models.CharField(_('Model'), null=False, blank=False, max_length=32)
    field = models.CharField(_('Field'), null=False, blank=False, max_length=32)

    def __unicode__(self):
        return self.translation if self.translation is not None else 'Missing trans: {0}.{1}'.format(self.model, self.field)


class SectionTitle(models.Model):
    section = models.ForeignKey('Section', related_name='titles')
    title = models.CharField(_('Title'), max_length=256, null=False)
    language = models.ForeignKey('Language')

    class Meta:
        ordering = ['title']


class Section(models.Model):
    """
    Represents a multilingual section of one/many Issues of
    a given Journal.

    ``legacy_code`` contains the section code used by the old
    title manager. We've decided to store this value just by
    historical reasons, and we don't know if it will last forever.
    """
    # Custom manager
    objects = SectionCustomManager()
    userobjects = modelmanagers.SectionManager()

    journal = models.ForeignKey(Journal)

    code = models.CharField(_('Legacy code'), unique=True, max_length=21, blank=True)
    legacy_code = models.CharField(null=True, blank=True, max_length=16)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    is_trashed = models.BooleanField(_('Is trashed?'), default=False, db_index=True)

    def __unicode__(self):
        return ' | '.join([sec_title.title for sec_title in self.titles.all().order_by('language')])

    @property
    def actual_code(self):
        if not self.pk or not self.code:
            raise AttributeError('section must be saved in order to have a code')

        return self.code

    def is_used(self):
        try:
            return True if self.issue_set.all().count() else False
        except ValueError:  # raised when the object is not yet saved
            return False

    def add_title(self, title, language):
        """
        Adds a section title in the given language.

        A Language instance must be passed as the language argument.
        """
        SectionTitle.objects.create(section=self, title=title, language=language)

    def _suggest_code(self, rand_generator=base28.genbase):
        """
        Suggests a code for the section instance.
        The code is formed by the journal acronym + 4 pseudo-random
        base 28 chars.

        ``rand_generator`` is the callable responsible for the pseudo-random
        chars sequence. It may accept the number of chars as argument.
        """
        num_chars = getattr(settings, 'SECTION_CODE_TOTAL_RANDOM_CHARS', 4)
        fmt = u'{0}-{1}'.format(self.journal.acronym, rand_generator(num_chars))
        return fmt

    def _create_code(self, *args, **kwargs):
        if not self.code:
            tries = kwargs.pop('max_tries', 5)
            while tries > 0:
                self.code = self._suggest_code()
                try:
                    super(Section, self).save(*args, **kwargs)
                except IntegrityError:
                    tries -= 1
                    logger.warning('conflict while trying to generate a section code. %i tries remaining.' % tries)
                    continue
                else:
                    logger.info('code created successfully for %s' % unicode(self))
                    break
            else:
                msg = 'max_tries reached while trying to generate a code for the section %s.' % unicode(self)
                logger.error(msg)
                raise DatabaseError(msg)

    class Meta:
        ordering = ('id',)
        permissions = (("list_section", "Can list Sections"),)

    def save(self, *args, **kwargs):
        """
        If ``code`` already exists, the section is saved. Else,
        the ``code`` will be generated before the save process is
        performed.
        """
        if self.code:
            super(Section, self).save(*args, **kwargs)
        else:
            # the call to super().save is delegated to _create_code
            # because there are needs to control saving max tries.
            self._create_code(*args, **kwargs)


class Issue(models.Model):

    # Custom manager
    objects = IssueCustomManager()
    userobjects = modelmanagers.IssueManager()

    section = models.ManyToManyField(Section, verbose_name=_("Section"), blank=True)
    journal = models.ForeignKey(Journal)
    volume = models.CharField(_('Volume'), blank=True, max_length=16)
    number = models.CharField(_('Number'), blank=True, max_length=16)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    publication_start_month = models.IntegerField(_('Start Month'), blank=True, null=True, choices=choices.MONTHS)
    publication_end_month = models.IntegerField(_('End Month'), blank=True, null=True, choices=choices.MONTHS)
    publication_year = models.IntegerField(_('Year'))
    is_marked_up = models.BooleanField(_('Is Marked Up?'), default=False, null=False, blank=True)
    use_license = models.ForeignKey(UseLicense, verbose_name=_("Use License"), null=True, help_text=ISSUE_DEFAULT_LICENSE_HELP_TEXT)
    total_documents = models.IntegerField(_('Total of Documents'), default=0)
    ctrl_vocabulary = models.CharField(_('Controlled Vocabulary'), max_length=64,
        choices=sorted(choices.CTRL_VOCABULARY, key=lambda CTRL_VOCABULARY: CTRL_VOCABULARY[1]), null=False, blank=True)
    editorial_standard = models.CharField(_('Editorial Standard'), max_length=64,
        choices=sorted(choices.STANDARD, key=lambda STANDARD: STANDARD[1]))
    cover = models.ImageField(_('Issue Cover'), upload_to='img/issue_cover/', null=True, blank=True)
    is_trashed = models.BooleanField(_('Is trashed?'), default=False, db_index=True)
    label = models.CharField(db_index=True, blank=True, null=True, max_length=64)

    order = models.IntegerField(_('Issue Order'), blank=True)

    type = models.CharField(_('Type'),  max_length=15, choices=choices.ISSUE_TYPES, default='regular', editable=False)
    suppl_text = models.CharField(_('Suppl Text'),  max_length=15, null=True, blank=True)
    spe_text = models.CharField(_('Special Text'),  max_length=15, null=True, blank=True)

    class Meta:
        ordering = ('created', 'id')
        permissions = (("list_issue", "Can list Issues"), )

    @property
    def scielo_pid(self):
        """
        Returns the PID used on SciELO public catalogs, in the form:
        ``journal_issn + year + order``
        """
        jissn = self.journal.scielo_pid
        return ''.join(
            [
                jissn,
                unicode(self.publication_year),
                u'%04d' % self.order,
            ]
        )

    @property
    def identification(self):
        values = [self.number]
        if self.type == 'supplement':
            values.append('suppl.%s' % self.suppl_text)

        if self.type == 'special':
            _spe_text = 'spe' + self.spe_text if self.spe_text else 'spe'
            values.append(_spe_text)

        return ' '.join([val for val in values if val]).strip().replace('ahead', 'ahead of print')

    def __unicode__(self):

        return "{0} ({1})".format(self.volume, self.identification).replace('()', '')

    @property
    def publication_date(self):
        start = self.get_publication_start_month_display()
        end = self.get_publication_end_month_display()
        if start and end:
            return '{0}/{1} {2}'.format(start[:3], end[:3], self.publication_year)
        elif start:
            return '{0} {1}'.format(start[:3], self.publication_year)
        elif end:
            return '{0} {1}'.format(end[:3], self.publication_year)
        else:
            return self.publication_year

    @property
    def verbose_identification(self):
        if self.type == 'supplement':
            prefixed_number = 'suppl.%s' % self.suppl_text
        elif self.type == 'special':
            prefixed_number = 'spe.%s' % self.spe_text
        else:  # regular
            prefixed_number = 'n.%s' % self.number
        return 'vol.%s %s' % (self.volume, prefixed_number)

    @property
    def suppl_type(self):
        if self.type == 'supplement':

            if self.number != '' and self.volume == '':
                return 'number'
            elif self.number == '' and self.volume != '':
                return 'volume'

        else:
            raise AttributeError('Issues of type %s do not have an attribute named: suppl_type' % self.get_type_display())

    @property
    def spe_type(self):
        if self.type == 'special':

            if self.number != '' and self.volume == '':
                return 'number'
            elif self.number == '' and self.volume != '':
                return 'volume'

        else:
            raise AttributeError('Issues of type %s do not have an attribute named: spe_type' % self.get_type_display())

    @property
    def bibliographic_legend(self):
        abrev_title = self.journal.title_iso
        issue = self.verbose_identification
        city = self.journal.publication_city
        dates = self.publication_date
        return '%s %s %s %s' % (abrev_title, issue, city, dates)

    def _suggest_order(self, force=False):
        """
        Based on ``publication_year``, ``volume`` and a pre defined
        ``order``, this method suggests the subsequent ``order`` value.

        If the Issues already has a ``order``, it suggests it. Else,
        a query is made for the given ``publication_year`` and ``volume``
        and the ``order`` attribute of the last instance is used.

        When force ``True`` this method ignore order attribute from the instance
        and return the suggest order.
        """
        if self.order and force is False:
            return self.order

        filters = {
            'publication_year': self.publication_year,
            'journal': self.journal,
        }

        try:
            last = Issue.objects.filter(**filters).order_by('order').reverse()[0]
            next_order = last.order + 1
        except IndexError:
            next_order = 1

        return next_order

    def _get_default_use_license(self):
        return self.journal.use_license

    def save(self, *args, **kwargs):
        self.label = unicode(self)

        if self.use_license is None and self.journal:
            self.use_license = self._get_default_use_license()

        # auto_order=False é passado na importação de dados para evitar o order automático
        if kwargs.pop('auto_order', True):
            if not self.pk:
                self.order = self._suggest_order()
            else:
                # the ordering control is based on publication year attr.
                # if an issue is moved between pub years, the order must be reset.
                if tools.has_changed(self, 'publication_year'):
                    self.order = self._suggest_order(force=True)

        super(Issue, self).save(*args, **kwargs)


class IssueTitle(models.Model):
    issue = models.ForeignKey(Issue)
    language = models.ForeignKey('Language')
    title = models.CharField(_('Title'), max_length=256)


class PendedForm(models.Model):
    view_name = models.CharField(max_length=128)
    form_hash = models.CharField(max_length=32)
    user = models.ForeignKey(User, related_name='pending_forms')
    created_at = models.DateTimeField(auto_now=True)


class PendedValue(models.Model):
    form = models.ForeignKey(PendedForm, related_name='data')
    name = models.CharField(max_length=255)
    value = models.TextField()


class PressRelease(models.Model):
    """
    Represents a press-release bound to a Journal.
    If ``issue`` is None, the pressrelease is refers to an ahead article.
    It can be available in one or any languages (restricted by the Journal
    publishing policy).
    """
    doi = models.CharField(_("Press release DOI number"),
                           max_length=128, null=True, blank=True)

    def add_article(self, article):
        """
        ``article`` is a string of the article pid.
        """
        PressReleaseArticle.objects.create(press_release=self,
                                           article_pid=article)

    def remove_article(self, article):
        try:
            pra = PressReleaseArticle.objects.get(press_release=self,
                                                  article_pid=article)
        except PressReleaseArticle.DoesNotExist:
            return None
        else:
            pra.delete()

    def add_translation(self, title, content, language):
        """
        Adds a new press-release translation.

        ``language`` is an instance of Language.
        """
        PressReleaseTranslation.objects.create(press_release=self,
                                               language=language,
                                               title=title,
                                               content=content)

    def remove_translation(self, language):
        """
        Removes the translation for the given press-release.
        If the translation doesn't exist, nothing happens silently.
        """
        qry_params = {'press_release': self}
        if isinstance(language, basestring):
            qry_params['language__iso_code'] = language
        else:
            qry_params['language'] = language

        try:
            pr = PressReleaseTranslation.objects.get(**qry_params)
        except PressReleaseTranslation.DoesNotExist:
            return None
        else:
            pr.delete()

    def get_trans(self, language):
        """
        Syntatic suggar for retrieving translations in a given language
        """
        prt = self.translations.get(language__iso_code=language)
        return prt

    def __unicode__(self):
        """
        Try to get the first title of the Press Release.
        The form ensures at least one title.
        """
        try:
            title = PressReleaseTranslation.objects.filter(press_release=self).order_by('language')[0].title
        except IndexError:
            return __('No Title')

        return title

    class Meta:
        abstract = False
        permissions = (("list_pressrelease", "Can list PressReleases"),)


class PressReleaseTranslation(models.Model):
    """
    Represents a press-release in a given language.
    """
    press_release = models.ForeignKey(PressRelease, related_name='translations')
    language = models.ForeignKey('Language')
    title = models.CharField(_('Title'), max_length=128)
    content = models.TextField(_('Content'))


class PressReleaseArticle(models.Model):
    """
    Represents press-releases bound to Articles.
    """
    press_release = models.ForeignKey(PressRelease, related_name='articles')
    article_pid = models.CharField(_('PID'), max_length=32, db_index=True)


class RegularPressRelease(PressRelease):
    objects = RegularPressReleaseCustomManager()
    userobjects = modelmanagers.RegularPressReleaseManager()

    issue = models.ForeignKey(Issue, related_name='press_releases')


class AheadPressRelease(PressRelease):
    objects = AheadPressReleaseCustomManager()
    userobjects = modelmanagers.AheadPressReleaseManager()

    journal = models.ForeignKey(Journal, related_name='press_releases')


class ArticlesLinkage(models.Model):
    """ Relação entre entidades do tipo `Article`.

    Representa a relação que no XML é realizada por meio do elemento
    `related-article`.

      - `referrer` é a instância de `Article` que remete à outra.
      - `link_to` é a instância de `Article` referida pela outra.
      - `link_type` é o tipo da relação, que pode ser: corrected-article ou
        commentary-article.
    """
    referrer = models.ForeignKey('Article', related_name='links_to')
    link_to = models.ForeignKey('Article', related_name='referrers')
    link_type = models.CharField(max_length=32)

    def __repr__(self):
        return u'<%s referrer="%s" link_to="%s" link_type="%s">' % (
                self.__class__.__name__, repr(self.referrer), repr(self.link_to),
                self.link_type)


class ArticleControlAttributes(models.Model):
    """ Atributos de controle de instância tipo `Article`.
    """
    es_updated_at = models.DateTimeField(null=True, blank=True)
    es_is_dirty = models.BooleanField(default=True)
    articles_linkage_is_pending = models.BooleanField(default=False)

    article = models.OneToOneField('Article', on_delete=models.CASCADE,
            related_name='control_attributes')

    def __repr__(self):
        return '<%s article="%s" es_updated_at="%s" es_is_dirty="%s" articles_linkage_is_pending="%s">' % (
                self.__class__.__name__, repr(self.article), self.es_updated_at,
                self.es_is_dirty, self.articles_linkage_is_pending)


class Article(models.Model):
    """
    Artigo associado ou não a um periódico ou fascículo.

    Atributos de controle -- integração com Elasticsearch e referenciamento
    recursivo -- podem ser acessador por meio do atributo `control_attributes`,
    duh.

    Obs:
        Na versão 1.8 do Django, tem incorporado um novo tipo de campo no core: UUIDField, que seria
        a solucão ideal para o campo ``aid``, incluido otimização no banco de dados.
        Mais informação em:
        - https://code.djangoproject.com/ticket/19463
        - https://github.com/django/django/commit/ed7821231b7dbf34a6c8ca65be3b9bcbda4a0703
    """
    objects = models.Manager()  # The default manager.
    userobjects = modelmanagers.ArticleManager()

    created_at = models.DateTimeField(auto_now_add=True, default=datetime.datetime.now)
    updated_at = models.DateTimeField(auto_now=True, default=datetime.datetime.now)

    aid = models.CharField(max_length=32, unique=True, editable=False)
    doi = models.CharField(max_length=2048, default=u'', db_index=True)
    domain_key = models.SlugField(max_length=2048, unique=True, db_index=False,
            editable=False)
    is_visible = models.BooleanField(default=True)
    is_aop = models.BooleanField(default=False)
    xml = XMLSPSField()
    xml_version = models.CharField(max_length=9)
    article_type = models.CharField(max_length=32, db_index=True)

    # artigo pode estar temporariamente desassociado de seu periódico e fascículo
    journal = models.ForeignKey(Journal, related_name='articles', blank=True, null=True)
    issue = models.ForeignKey(Issue, related_name='articles', blank=True, null=True)
    related_articles = models.ManyToManyField('self', through='ArticlesLinkage',
            symmetrical=False, blank=True, null=True)
    journal_title = models.CharField(_('Journal title'), max_length=512, db_index=True)
    issn_ppub = models.CharField(max_length=9, db_index=True)
    issn_epub = models.CharField(max_length=9, db_index=True)

    class Meta:
        permissions = (("list_article", "Can list Article"),)

    class XPaths:
        """ A classe XPaths serve apenas como um namespace para constantes
        que representam expressões xpath para elementos do XML comumente
        acessados. Importante: As expressões são para os elementos e não para
        seus textos.
        """
        SPS_VERSION = '/article/@specific-use'
        ABBREV_JOURNAL_TITLE = '/article/front/journal-meta/journal-title-group/abbrev-journal-title[@abbrev-type="publisher"]'
        JOURNAL_TITLE = '/article/front/journal-meta/journal-title-group/journal-title'
        ISSN_PPUB = '/article/front/journal-meta/issn[@pub-type="ppub"]'
        ISSN_EPUB = '/article/front/journal-meta/issn[@pub-type="epub"]'
        ARTICLE_TITLE = '/article/front/article-meta/title-group/article-title'
        YEAR = '/article/front/article-meta/pub-date/year'
        VOLUME = '/article/front/article-meta/volume'
        ISSUE = '/article/front/article-meta/issue'
        FPAGE = '/article/front/article-meta/fpage'
        FPAGE_SEQ = '/article/front/article-meta/fpage/@seq'
        LPAGE = '/article/front/article-meta/lpage'
        ELOCATION_ID = '/article/front/article-meta/elocation-id'
        HEAD_SUBJECT = '/article/front/article-meta/article-categories/subj-group[@subj-group-type="heading"]/subject'
        DOI = '/article/front/article-meta/article-id[@pub-id-type="doi"]'
        PID = '/article/front/article-meta/article-id[@pub-id-type="publisher-id"]'
        ARTICLE_TYPE = '/article/@article-type'
        AOP_ID = '/article/front/article-meta/article-id[@pub-id-type="other"]'
        RELATED_CORRECTED_ARTICLES = '/article/front/article-meta/related-article[@related-article-type="corrected-article"]'
        RELATED_COMMENTARY_ARTICLES = '/article/response/front-stub/related-article[@related-article-type="commentary-article"]'

    def save(self, *args, **kwargs):
        """
        Ao salvar a instância pela primeira vez, o valor do atributo `aid` é
        gerado automaticamente caso nenhum valor tenha sido atribuido.
        """
        if self.pk is None and not self.aid:
            self.aid = str(uuid4().hex)

        super(Article, self).save(*args, **kwargs)

    @classmethod
    def parse(cls, content_as_bytes):
        newarticle = cls(xml=content_as_bytes)

        xpaths = cls.XPaths
        get_value = newarticle.get_value

        newarticle.is_aop = newarticle._get_is_aop()
        newarticle.domain_key = newarticle._get_domain_key()
        newarticle.journal_title = get_value(xpaths.JOURNAL_TITLE)
        newarticle.issn_ppub = get_value(xpaths.ISSN_PPUB) or ''
        newarticle.issn_epub = get_value(xpaths.ISSN_EPUB) or ''
        newarticle.xml_version = get_value(xpaths.SPS_VERSION) or 'pre-sps'
        newarticle.article_type = get_value(xpaths.ARTICLE_TYPE)
        newarticle.doi = get_value(xpaths.DOI) or ''

        if not any([newarticle.issn_ppub, newarticle.issn_epub]):
            raise ValueError('Either issn_ppub or issn_epub must be set')

        if not newarticle.journal_title:
            raise ValueError('Could not get journal-title.')

        if not newarticle.article_type:
            raise ValueError('Could not get article-type.')

        return newarticle

    def get_value(self, expression):
        """ Busca `expression` em `self.xml` e retorna o resultado da primeira ocorrência.

        Espaços em branco no início ou fim são removidos. Retorna `None` caso
        `expression` não encontre elementos, ou o elemento esteja vazio.

        :param expression: expressão xpath para elemento ou atributo.
        """
        try:
            first_occ = self.xml.xpath(expression)[0]
        except IndexError:
            return None

        try:
            value = first_occ.text
        except AttributeError:
            # valor de atributo
            value = first_occ

        try:
            return value.strip()
        except AttributeError:
            return value

    def _get_is_aop(self):
        """ Infere se trata-se de um artigo AOP.

        Diferentemente das regras publicadas no SciELO PS, o SciELO Manager
        considera AOP (ahead-of-print) os artigos que apresentam o número
        identificador de AOP no elemento
        ``//article-meta/article-id[pub-id-type="other"]``. O SciELO Manager
        adota uma abordagem não-opinionada acerca da estrutura de publicação
        do periódico, portanto não utiliza metadados do número na lógica de
        inferência.
        """
        aop_id = self.get_value(self.XPaths.AOP_ID)
        return bool(aop_id)

    def _get_domain_key(self):
        """ Produz uma chave de domínio (Domain key ou Natural key)

        A chave é utilizada na detecção de duplicidades. Os metadados
        que a compõem e não estão presentes são substituídos pela string
        `None`, para que a estrutura seja mantida.

        A chave é composta pela concatenação dos metadados por meio do
        separador `_`:

          * ``//journal-meta/journal-title-group/journal-title``
          * ``//article-meta/volume``
          * ``//article-meta/issue``
          * ``//article-meta/pub-date/year``
          * ``//article-meta/fpage``
          * ``//article-meta/fpage/@seq``
          * ``//article-meta/lpage``
          * ``//article-meta/elocation-id``
          * ``//article-meta/article-id[pub-id-type="other"]``

        Após a concatenação, é normalizada por meio da função `slugify`.

        Formato:

          <journal-title>_<volume>_<issue>_<year>_<fpage>_<seq>_<lpage>_<elocation-id>_<article-id>
        """
        id_fields = [
                self.XPaths.JOURNAL_TITLE,
                self.XPaths.VOLUME,
                self.XPaths.ISSUE,
                self.XPaths.YEAR,
                self.XPaths.FPAGE,
                self.XPaths.FPAGE_SEQ,
                self.XPaths.LPAGE,
                self.XPaths.ELOCATION_ID,
                self.XPaths.AOP_ID,
        ]

        values = (self.get_value(path) for path in id_fields)
        text_values = (value if value else 'none' for value in values)
        joined_values = '_'.join(text_values)
        return slugify(joined_values)

    def __repr__(self):
        # para instâncias não salvas
        if self.xml is None:
            domain_key = u''
        else:
            domain_key = self.domain_key or self._get_domain_key()

        return u'<%s aid="%s" domain_key="%s">' % (self.__class__.__name__,
                self.aid, domain_key)


def make_article_directory_path(content_type):
    """ Produz funções que definem o diretório de armazenamento dos arquivos
    relacionados a um artigo.

    O ativo será armazenado em:
    MEDIA_ROOT/articles/<aid_seg1>/<aid_seg2>/<aid_seg3>/<aid>/<content_type>/<filename>.
    """
    def article_directory_path(instance, filename):
        aid = instance.article.aid
        seg1 = aid[:2]
        seg2 = aid[2:4]
        seg3 = aid[4:6]

        return 'articles/{seg1}/{seg2}/{seg3}/{aid}/{type}/{filename}'.format(
                seg1=seg1, seg2=seg2, seg3=seg3, aid=aid,
                type=content_type, filename=filename)

    return article_directory_path


class ArticleAsset(models.Model):
    """Ativo digital vinculado a uma instância de Article.
    """
    article = models.ForeignKey('Article', on_delete=models.CASCADE,
            related_name='assets')
    file = models.FileField(
            upload_to=make_article_directory_path('assets'),
            max_length=1024)
    preferred_alt_file = models.FileField(
            upload_to=make_article_directory_path('alt_assets'),
            max_length=1024, default=u'')
    owner = models.CharField(max_length=1024, default=u'')
    use_license = models.TextField(default=u'')
    updated_at = models.DateTimeField(auto_now=True)

    def __repr__(self):
        return u'<%s id="%s" url="%s">' % (self.__class__.__name__,
                self.pk, self.file.url)

    def is_image(self):
        """Verifica se o ativo digital é uma imagem.
        """
        try:
            img = Image.open(self.file)
        except IOError:
            return False
        else:
            return True

    @property
    def best_file(self):
        if self.preferred_alt_file:
            return self.preferred_alt_file
        else:
            return self.file


class ArticleHTMLRendition(models.Model):
    """Documento HTML de uma tradução de uma instância de Article.

    Armazena apenas o *build* mais recente para cada idioma.
    """
    article = models.ForeignKey('Article', on_delete=models.CASCADE,
            related_name='htmls')
    file = models.FileField(upload_to=make_article_directory_path('htmls'),
            max_length=1024)
    lang = models.CharField(_('ISO 639-1 Language Code'), max_length=2)
    build_version = models.CharField(max_length=8)
    updated_at = models.DateTimeField(auto_now=True)

    def __repr__(self):
        return u'<%s id="%s" url="%s">' % (self.__class__.__name__,
                self.pk, self.file.url)

    class Meta:
        unique_together = (('article', 'lang'),)


# --------------------
# Callbacks de signals
# --------------------
@receiver(post_save, sender=Membership)
def journal_update_date(sender, instance, created, **kwargs):
    """ Update journal.update_date when journal status is changed.
    """

    instance.journal.updated = datetime.datetime.now()
    instance.journal.save()


@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    """ Create a matching profile whenever a user object is created.
    """
    if created:
        profile, new = UserProfile.objects.get_or_create(user=instance)


@receiver(post_save, sender=ArticleControlAttributes)
def submit_to_elasticsearch(sender, instance, **kwargs):
    """ Indexa o artigo no Elasticsearch sempre que o atributo `es_is_dirty`
    for igual a `True`.
    """
    if instance.es_is_dirty:
        celery.current_app.send_task(
                'journalmanager.tasks.submit_to_elasticsearch',
                args=[instance.article.pk])


@receiver(post_save, sender=Article)
def link_article_to_journal(sender, instance, created, **kwargs):
    """ Tenta encontrar o periódico e o fascículo do artigo recém criado.

    A busca pelo fascículo é disparada automaticamente após a associação
    bem-sucedida com o periódico.
    """
    if created:
        celery.current_app.send_task(
                'journalmanager.tasks.link_article_to_journal',
                args=[instance.pk])


@receiver(post_save, sender=Article)
def create_article_control_attributes(sender, instance, created, **kwargs):
    """ Cria a entidade com os atributos de controle da instância de `Article`.
    """
    ctrl_attrs, ctrl_attrs_created = ArticleControlAttributes.objects.get_or_create(
            article=instance)

    if ctrl_attrs_created:
        linkage_is_pending = instance.article_type in LINKABLE_ARTICLE_TYPES
        ctrl_attrs.articles_linkage_is_pending = linkage_is_pending
        ctrl_attrs.save()


@receiver(post_save, sender=Article)
def create_article_html_renditions(sender, instance, created, **kwargs):
    """ Cria os documentos HTML para cada idioma do artigo.
    """
    if created:
        celery.current_app.send_task(
                'journalmanager.tasks.create_article_html_renditions',
                args=[instance.pk])


# Callback da tasty-pie para a geração de token para os usuários
models.signals.post_save.connect(create_api_key, sender=User)

