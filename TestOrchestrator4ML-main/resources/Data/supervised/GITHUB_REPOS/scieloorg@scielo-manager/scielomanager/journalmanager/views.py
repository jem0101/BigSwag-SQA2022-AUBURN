# coding: utf-8
import json
import urlparse
from datetime import datetime
import operator

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import packtools
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.decorators import login_required
from django.contrib.auth.decorators import permission_required
from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth.models import User
from django.contrib.auth.models import Group
from django.contrib import messages
from django.core.urlresolvers import reverse
from django.core.urlresolvers import resolve
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.shortcuts import render_to_response
from django.template import loader
from django.template.context import RequestContext
from django.utils.translation import ugettext as _
from django.utils.functional import curry
from django.utils.html import escape
from django.forms.models import inlineformset_factory
from django.forms.formsets import formset_factory
from django.conf import settings
from django.db.models import Q
from django.templatetags.static import static

from . import models
from .forms import *
from scielomanager.utils.pendingform import PendingPostData
from scielomanager.utils import usercontext
from scielomanager.tools import (
    get_paginated,
    get_referer_view,
    asbool,
    get_users_by_group,
)
from audit_log import helpers
from editorialmanager.models import EditorialBoard
from editorialmanager import notifications
from accounts import forms as accounts_forms

MSG_FORM_SAVED = _(u'Saved.')
MSG_FORM_SAVED_PARTIALLY = _(u'Saved partially. You can continue to fill in this form later.')
MSG_FORM_MISSING = _(u'There are some errors or missing data.')
MSG_DELETE_PENDED = _(u'The pended form has been deleted.')

user_request_context = usercontext.get_finder()


class ArticleAttrGetter(object):
    """ Adapta a interface de instâncias de Article para a recuperação
    de valores do XML via sintaxe de acesso à atributos.
    """
    def __init__(self, article):
        self.article = article

    def __getattr__(self, name):
        xpath = getattr(self.article.XPaths, name, None)
        if xpath is not None:
            return self.article.get_value(xpath)
        else:
            return getattr(self.article, name)


def _user_has_access(user):

    collection = models.Collection.userobjects.active()

    if not collection.is_managed_by_user(user):
        return False

    return True


def get_first_letter(objects_all):
    """
    Returns a set of first letters from names in `objects_all`
    """
    letters_set = set(unicode(letter).strip()[0].upper() for letter in objects_all)

    return sorted(list(letters_set))


@permission_required('journalmanager.list_editor_journal', login_url=settings.AUTHZ_REDIRECT_URL)
def get_editor(request, journal_id):
    """
    Get the editor of the journal.
    """
    users_editor = None

    journal = get_object_or_404(models.Journal, id=journal_id)

    if not journal.editor:
        try:
            users_editor = get_users_by_group('Editors')
        except ObjectDoesNotExist:
            umessages.error(request, _(u"Does not exist the group 'Editors'"))

    return render_to_response('journalmanager/editor.html',
                             {'editor': journal.editor,
                              'journal': journal,
                              'users_editor': users_editor},
                              context_instance=RequestContext(request))


@permission_required('journalmanager.change_editor', login_url=settings.AUTHZ_REDIRECT_URL)
def add_editor(request, journal_id):
    """
    Set any user from Editors as editor of any journal.
    """
    journal = get_object_or_404(models.Journal, id=journal_id)

    if request.method == "POST":
        editor_pk = request.POST.get('editor', None)
        if editor_pk:
            editor = User.objects.get(pk=editor_pk)
            journal.editor = editor
            journal.save()
            messages.success(request, _(u"Successfully selected %s as editor of this Journal" % editor.get_full_name()))
        else:
            #Remove editor
            journal.editor = None
            journal.save()
            messages.success(request, _(u"No user selected as editor of this journal!"))
    else:
        try:
            users_editor = get_users_by_group('Editors')
        except ObjectDoesNotExist:
            messages.error(request, _(u"Does not exist the group 'Editors'"))

        return render_to_response('journalmanager/includes/form_add_editor.html',
                                 {'journal': journal,
                                  'users_editor': users_editor},
                                 context_instance=RequestContext(request))

    return HttpResponseRedirect(reverse('editor.index', args=[journal.id]))


def index(request):

    if not request.user.is_authenticated():
        return HttpResponseRedirect(reverse('journalmanager.user_login'))

    #Redirect user when it`s is editor
    if request.user.get_profile().is_editor:
        return HttpResponseRedirect(reverse('editorial.index'))

    pending_journals = models.PendedForm.objects.filter(
        user=request.user.id).filter(view_name='journal.add').order_by('-created_at')

    # recent activities
    recent_journals = models.Journal.objects.recents_by_user(request.user)

    context = {
        'pending_journals': pending_journals,
        'recent_activities': recent_journals,
        }

    return render_to_response('journalmanager/home_journal.html',
        context, context_instance=RequestContext(request))


@permission_required('journalmanager.list_journal', login_url=settings.AUTHZ_REDIRECT_URL)
def journal_index(request):
    """
    Journal list by active collection
    """
    filters = {}

    if request.GET.get('q'):
        filters['title__icontains'] = request.GET.get('q')

    if request.GET.get('letter'):
        filters['title__istartswith'] = request.GET.get('letter')

    if request.GET.get('jstatus'):
        filters['membership__status'] = request.GET.get('jstatus')

    journals = models.Journal.userobjects.active().filter(**filters)

    objects = get_paginated(journals, request.GET.get('page', 1))

    return render_to_response('journalmanager/journal_list.html', {
           'objects_journal': objects,
           'letters': get_first_letter(journals),
        },
        context_instance=RequestContext(request))


@permission_required('journalmanager.list_journal', login_url=settings.AUTHZ_REDIRECT_URL)
def dash_journal(request, journal_id=None):
    """
    Handles new and existing journals
    """

    journal = get_object_or_404(models.Journal.userobjects.active(), id=journal_id)

    return render_to_response('journalmanager/journal_dash.html', {
                              'journal': journal,
                              }, context_instance=RequestContext(request))


@permission_required('journalmanager.list_issue', login_url=settings.AUTHZ_REDIRECT_URL)
def issue_index(request, journal_id):
    journal = get_object_or_404(models.Journal.userobjects.active(), pk=journal_id)

    current_year = datetime.now().year
    previous_year = current_year - 1

    if request.method == 'POST':
        aheadform = AheadForm(request.POST, instance=journal, prefix='journal')
        if aheadform.is_valid():
            aheadform.save()
        else:
            messages.error(request, MSG_FORM_MISSING)
    else:
        aheadform = AheadForm(instance=journal, prefix='journal')


    aop_articles = [ArticleAttrGetter(art)
                    for art in journal.articles.filter(is_aop=True)]

    return render_to_response(
        'journalmanager/issue_list.html',
        {
            'journal': journal,
            'aheadform': aheadform,
            'current_year': current_year,
            'previous_year': previous_year,
            'issue_grid': journal.issues_as_grid(
                request.GET.get('is_available')),
            'aop_articles': aop_articles,
        },
        context_instance=RequestContext(request)
    )


@permission_required('journalmanager.list_section', login_url=settings.AUTHZ_REDIRECT_URL)
def section_index(request, journal_id=None):
    """
    Section list by active collection
    """
    journal = get_object_or_404(models.Journal.userobjects.active(), id=journal_id)

    sections = models.Section.userobjects.active().filter(journal=journal)
    sections = sorted(sections, key=lambda x: unicode(x))

    objects = get_paginated(sections, request.GET.get('page', 1))

    return render_to_response('journalmanager/section_list.html', {
           'objects_section': objects,
           'journal': journal },
           context_instance=RequestContext(request))


@permission_required('journalmanager.list_sponsor', login_url=settings.AUTHZ_REDIRECT_URL)
def sponsor_index(request):
    """
    Sponsor list by active collection
    """
    filters = {}

    if request.GET.get('q'):
        filters['name__icontains'] = request.GET.get('q')

    if request.GET.get('letter'):
        filters['name__istartswith'] = request.GET.get('letter')

    sponsors = models.Sponsor.userobjects.active().filter(**filters)

    objects = get_paginated(sponsors, request.GET.get('page', 1))

    return render_to_response('journalmanager/sponsor_list.html', {
           'objects_sponsor': objects,
           'letters': get_first_letter(sponsors)},
           context_instance=RequestContext(request))


@permission_required('journalmanager.list_article', login_url=settings.AUTHZ_REDIRECT_URL)
def article_index(request, issue_id):

    issue = get_object_or_404(models.Issue.userobjects.active(), pk=issue_id)

    articles = (ArticleAttrGetter(art) for art in issue.articles.all())

    return render_to_response(
        'journalmanager/article_list.html',
        {
            'journal': issue.journal,
            'issue': issue,
            'articles': articles,
        },
        context_instance=RequestContext(request)
    )


@permission_required('journalmanager.list_article', login_url=settings.AUTHZ_REDIRECT_URL)
def article_detail(request, article_pk):

    article = get_object_or_404(models.Article.userobjects.active(), pk=article_pk)

    def safe_read(file):
        with file as _file:
            return _file.read()

    previews = [{'lang': html.lang, 'html': safe_read(html.file)}
                for html in article.htmls.all()]

    return render_to_response(
        'journalmanager/article_detail.html',
        {
            'article': ArticleAttrGetter(article),
            'journal': article.journal,
            'packtools_version': packtools.__version__,
            'previews': previews,
        },
        context_instance=RequestContext(request)
    )


@permission_required('journalmanager.list_pressrelease', login_url=settings.AUTHZ_REDIRECT_URL)
def pressrelease_index(request, journal_id):
    journal = get_object_or_404(models.Journal, pk=journal_id)

    param_tab = request.GET.get('tab')
    pr_model = models.AheadPressRelease if param_tab == 'ahead' else models.RegularPressRelease

    preleases = pr_model.userobjects.active().journal(journal).select_related()
    objects = get_paginated(preleases, request.GET.get('page', 1))

    return render_to_response(
        'journalmanager/pressrelease_list.html',
        {
           'objects_pr': objects,
           'journal': journal,
        },
        context_instance=RequestContext(request))


@login_required
def generic_toggle_availability(request, object_id, model):

    if request.is_ajax():

        model = get_object_or_404(model.userobjects.active(), pk=object_id)
        model.is_trashed = not model.is_trashed
        model.save()

        return HttpResponse(mimetype="application/json")
    else:
        return HttpResponse(status=400)


@login_required
def toggle_active_collection(request, user_id, collection_id):
    '''
    Redefine the active collection, changing the administrative context to another collection.
    '''

    # Setting up all user collections.is_default to False
    user_collections = models.get_user_collections(request.user.id)

    collection = get_object_or_404(models.Collection, pk=collection_id)
    collection.make_default_to_user(request.user)

    return HttpResponseRedirect('/')


@login_required
def generic_bulk_action(request, model_name, action_name, value=None):
    info_msg = None
    MSG_MOVED = _('The selected documents had been moved to the Trash.')
    MSG_RESTORED = _('The selected documents had been restored.')

    model_refs = {
        'journal': models.Journal,
        'section': models.Section,
        'sponsor': models.Sponsor,
    }
    model = model_refs.get(model_name)

    if request.method == 'POST':
        items = request.POST.getlist('action')
        for doc_id in items:
            doc = get_object_or_404(model, pk=doc_id)

            #toggle doc availability
            if action_name == 'is_available':
                if isinstance(doc, models.Journal):
                    doc.is_trashed = True if int(value) == 0 else False
                    doc.save()
                    info_msg = MSG_MOVED if doc.is_trashed else MSG_RESTORED
                elif isinstance(doc, models.Section):
                    if not doc.is_used():
                        doc.is_trashed = True if int(value) == 0 else False
                        doc.save()
                        info_msg = MSG_MOVED if doc.is_trashed else MSG_RESTORED
                elif isinstance(doc, models.Institution):
                    doc.is_trashed = True if int(value) == 0 else False
                    doc.save()
                    info_msg = MSG_MOVED if doc.is_trashed else MSG_RESTORED

    if info_msg:
        messages.info(request, info_msg)
    return HttpResponseRedirect(get_referer_view(request))


@permission_required('auth.change_user', login_url=settings.AUTHZ_REDIRECT_URL)
def user_index(request):

    collection = models.Collection.userobjects.active()

    if not collection.is_managed_by_user(request.user):
        return HttpResponseRedirect(settings.AUTHZ_REDIRECT_URL)

    col_users = models.User.objects.filter(
        usercollections__collection__in=[collection]).distinct('username').order_by('username')

    users = get_paginated(col_users, request.GET.get('page', 1))

    elegible_users = models.User.objects.filter(
        ~Q(usercollections__collection__in=[collection])).distinct('username').order_by('username')

    t = loader.get_template('journalmanager/user_list.html')
    c = RequestContext(request, {
                       'users': users,
                       'elegible_users': elegible_users
                       })
    return HttpResponse(t.render(c))


@permission_required('auth.change_user', login_url=settings.AUTHZ_REDIRECT_URL)
@user_passes_test(_user_has_access, login_url=settings.AUTHZ_REDIRECT_URL)
def add_user_to_collection(request):
    """
    Add a existing user to the active collection
    """

    collection = models.Collection.userobjects.active()

    if request.method == 'POST':
        user_id = request.POST.get('user_id', None)
        if user_id:
            user = get_object_or_404(User, id=user_id)
            collection.add_user(user)
            return HttpResponseRedirect(reverse('user.index'))
        else:
            return HttpResponseRedirect(reverse('user.add'))


@permission_required('auth.change_user', login_url=settings.AUTHZ_REDIRECT_URL)
@user_passes_test(_user_has_access, login_url=settings.AUTHZ_REDIRECT_URL)
def exclude_user_from_collection(request, user_id=None):
    """
    Remove a user from the active collection
    """

    collection = models.Collection.userobjects.active()

    if user_id:
        user = get_object_or_404(User, id=user_id)

        collection.remove_user(user)

    return HttpResponseRedirect(reverse('user.index'))


@permission_required('auth.change_user', login_url=settings.AUTHZ_REDIRECT_URL)
@user_passes_test(_user_has_access, login_url=settings.AUTHZ_REDIRECT_URL)
def add_user(request, user_id=None):
    """
    Handles new and existing users
    """
    collection = models.Collection.userobjects.active()

    if user_id is None:
        user = User()
    else:
        user = get_object_or_404(User, id=user_id)

    # Getting Collections from the logged user.
    user_collections = models.get_user_collections(request.user.id)

    UserCollectionsFormSet = inlineformset_factory(
        User, models.UserCollections, form=UserCollectionsForm,
        extra=1, can_delete=True, formset=FirstFieldRequiredFormSet)

    # filter the collections the user is manager.
    UserCollectionsFormSet.form = staticmethod(curry(UserCollectionsForm, user=request.user))
    # user profile fomrset
    UserProfileFormSet = inlineformset_factory(User, models.UserProfile, form=UserProfileForm, extra=1, max_num=1, can_delete=False)

    if request.method == 'POST':
        userform = UserForm(request.POST, instance=user, prefix='user')
        usercollectionsformset = UserCollectionsFormSet(request.POST, instance=user, prefix='usercollections',)
        userprofileformset = UserProfileFormSet(request.POST, instance=user, prefix='userprofile')

        if userform.is_valid() and usercollectionsformset.is_valid() and userprofileformset.is_valid():
            new_user = userform.save()

            # force the first instance (collection) to be set as default
            instances = usercollectionsformset.save(commit=False)
            has_set_as_default = False
            for instance in instances:
                if not has_set_as_default:
                    instance.is_default = True
                    has_set_as_default = True
                instance.save()

            # work-around to solve bug: #1053
            new_user_profile = new_user.get_profile()
            profile_form = userprofileformset.forms[0] # only one form must exist (User <- OneToOne -> Profile)
            if profile_form.has_changed():
                for field in profile_form.changed_data:
                    changed_field_value = profile_form.cleaned_data[field]
                    if hasattr(new_user_profile, field):
                        setattr(new_user_profile, field, changed_field_value)
                        new_user_profile.save()

            # if it is a new user, mail him
            # requesting for password change
            if not user_id:
                password_form = accounts_forms.PasswordResetForm({'email': new_user.email})
                if password_form.is_valid():
                    opts = {
                        'use_https': request.is_secure(),
                        'request': request,
                    }
                    password_form.save(**opts)

            messages.info(request, MSG_FORM_SAVED)
            return HttpResponseRedirect(reverse('user.index'))
        else:
            messages.error(request, MSG_FORM_MISSING)
    else:
        userform = UserForm(instance=user, prefix='user')
        usercollectionsformset = UserCollectionsFormSet(instance=user, prefix='usercollections',)
        userprofileformset = UserProfileFormSet(instance=user, prefix='userprofile')

    return render_to_response('journalmanager/add_user.html', {
                              'add_form': userform,
                              'mode': 'user_journal',
                              'user_name': request.user.pk,
                              'usercollectionsformset': usercollectionsformset,
                              'userprofileformset': userprofileformset,
                              'user': user
                              },
                              context_instance=RequestContext(request))


@permission_required('journalmanager.change_journaltimeline', login_url=settings.AUTHZ_REDIRECT_URL)
def edit_journal_status(request, journal_id=None):
    """
    Handles Journal Status.

    Allow user just to update the status history of a specific journal.
    """

    journal = get_object_or_404(models.Journal.userobjects.active(), id=journal_id)

    current_user_collection = user_request_context.get_current_user_active_collection()
    journal_history = journal.statuses.filter(collection=current_user_collection)

    if request.method == "POST":
        membership = journal.membership_info(current_user_collection)
        membershipform = MembershipForm(request.POST, instance=membership)
        if membershipform.is_valid():
            membershipform.save_all(request.user, journal, current_user_collection)
            messages.info(request, MSG_FORM_SAVED)
            return HttpResponseRedirect(reverse(
                'journal_status.edit', kwargs={'journal_id': journal_id}))
        else:
            messages.error(request, MSG_FORM_MISSING)
    else:
        membershipform = MembershipForm()

    return render_to_response('journalmanager/edit_journal_status.html', {
                              'add_form': membershipform,
                              'journal_history': journal_history,
                              'journal': journal,
                              }, context_instance=RequestContext(request))


@permission_required('journalmanager.change_journal', login_url=settings.AUTHZ_REDIRECT_URL)
def add_journal(request, journal_id=None):
    """
    Handles new and existing journals
    """

    user_collections = models.get_user_collections(request.user.id)
    previous_journal_cover = None
    previous_journal_logo = None
    has_cover_url = has_logo_url = False

    is_new_journal = journal_id is None

    if is_new_journal:
        journal = models.Journal()
    else:
        journal = get_object_or_404(models.Journal, id=journal_id)
        # preserve the cover and logo urls before save in case of error when updating these fields

        try:
            previous_journal_cover = journal.cover.url
        except ValueError:
            previous_journal_cover = None

        try:
            previous_journal_logo = journal.logo.url
        except ValueError:
            previous_journal_logo = None

    form_hash = None

    JournalTitleFormSet = inlineformset_factory(models.Journal, models.JournalTitle, form=JournalTitleForm, extra=1, can_delete=True)
    JournalMissionFormSet = inlineformset_factory(models.Journal, models.JournalMission, form=JournalMissionForm, extra=1, can_delete=True)

    if request.method == "POST":
        journalform = JournalForm(request.POST, request.FILES, instance=journal, prefix='journal')
        titleformset = JournalTitleFormSet(request.POST, instance=journal, prefix='title')
        missionformset = JournalMissionFormSet(request.POST, instance=journal, prefix='mission')

        if not is_new_journal:
            audit_old_values = helpers.collect_old_values(journal, journalform, [titleformset, missionformset,])
        else:
            audit_old_values = None

        if 'pend' in request.POST:
            journal_form_hash = PendingPostData(request.POST).pend(resolve(request.get_full_path()).url_name, request.user)
            form_hash = journal_form_hash
            messages.info(request, MSG_FORM_SAVED_PARTIALLY)
        else:

            if journalform.is_valid() and titleformset.is_valid() and missionformset.is_valid():
                # Ensuring that journal doesnt exists on created journal form, so journal_id must be None
                filter_list = []

                if request.POST.get('journal-print_issn') != '':
                    filter_list.append(Q(print_issn__icontains=request.POST.get('journal-print_issn')))

                if request.POST.get('journal-eletronic_issn') != '':
                    filter_list.append(Q(eletronic_issn__icontains=request.POST.get('journal-eletronic_issn')))

                if journal_id is None and models.Journal.objects.filter(reduce(operator.or_, filter_list)).exists():
                    messages.error(request, _(u"This Journal already exists, please search the journal in the previous step"))
                else:
                    saved_journal = journalform.save_all(creator=request.user)

                    if not journal_id:
                        saved_journal.join(user_request_context.get_current_user_active_collection(), request.user)

                    titleformset.save()
                    missionformset.save()

                    audit_data = {
                        'user': request.user,
                        'obj': saved_journal,
                        'old_values': audit_old_values,
                        'new_values': helpers.collect_new_values(journalform, [titleformset, missionformset,]),
                    }
                    if is_new_journal:
                        audit_data['message'] = helpers.construct_create_message(journalform, [titleformset, missionformset,])
                        helpers.log_create(**audit_data)
                    else:
                        audit_data['message'] = helpers.construct_change_message(journalform, [titleformset, missionformset,])
                        helpers.log_change(**audit_data)

                    messages.info(request, MSG_FORM_SAVED)

                    if request.POST.get('form_hash', None) and request.POST['form_hash'] != 'None':
                        models.PendedForm.objects.get(form_hash=request.POST['form_hash']).delete()

                    return HttpResponseRedirect(reverse('journal.dash', args=[saved_journal.id]))
            else:
                messages.error(request, MSG_FORM_MISSING)

                # if conver or logo fail in validation, then override the has_xxx_url with
                # the value stored previously, or false if none

                if 'cover' in journalform.errors.keys():
                    has_cover_url = previous_journal_cover if previous_journal_cover else False
                else:
                    has_cover_url = journal.cover.url if hasattr(journal, 'cover') and hasattr(journal.cover, 'url') else False

                if 'logo' in journalform.errors.keys():
                    has_logo_url = previous_journal_logo if previous_journal_logo else False
                else:
                    has_logo_url = journal.logo.url if hasattr(journal, 'logo') and hasattr(journal.logo, 'url') else False

    else:
        if request.GET.get('resume', None):
            pended_post_data = PendingPostData.resume(request.GET.get('resume'))

            journalform = JournalForm(pended_post_data,  request.FILES, instance=journal, prefix='journal')
            titleformset = JournalTitleFormSet(pended_post_data, instance=journal, prefix='title')
            missionformset = JournalMissionFormSet(pended_post_data, instance=journal, prefix='mission')
        else:
            journalform = JournalForm(instance=journal, prefix='journal')
            titleformset = JournalTitleFormSet(instance=journal, prefix='title')
            missionformset = JournalMissionFormSet(instance=journal, prefix='mission')

        # Recovering Journal Cover url.
        try:
            has_cover_url = journal.cover.url
        except ValueError:
            has_cover_url = False

        # Recovering Journal Logo url.
        try:
            has_logo_url = journal.logo.url
        except ValueError:
            has_logo_url = False

    return render_to_response('journalmanager/add_journal.html', {
                              'journal': journal,
                              'add_form': journalform,
                              'titleformset': titleformset,
                              'missionformset': missionformset,
                              'has_cover_url': has_cover_url,
                              'has_logo_url': has_logo_url,
                              'form_hash': form_hash if form_hash else request.GET.get('resume', None),
                              'is_new': False if journal_id else True,
                              }, context_instance=RequestContext(request))


@login_required
def del_pended(request, form_hash):
    pended_form = get_object_or_404(models.PendedForm, form_hash=form_hash, user=request.user)
    pended_form.delete()
    messages.info(request, MSG_DELETE_PENDED)
    return HttpResponseRedirect(reverse('index'))


@permission_required('journalmanager.add_sponsor', login_url=settings.AUTHZ_REDIRECT_URL)
def add_sponsor(request, sponsor_id=None):
    """
    Handles new and existing sponsors
    """

    if sponsor_id is None:
        sponsor = models.Sponsor()
    else:
        sponsor = get_object_or_404(models.Sponsor.userobjects.active(), id=sponsor_id)

    user_collections = models.get_user_collections(request.user.id)

    if request.method == "POST":
        sponsorform = SponsorForm(request.POST, instance=sponsor, prefix='sponsor',
            collections_qset=user_collections)

        if sponsorform.is_valid():
            newsponsorform = sponsorform.save()

            if request.POST.get('popup', 0):
                return HttpResponse(u'<script type="text/javascript">\
                    opener.updateSelect(window, "%s", "%s", "id_journal-sponsor");</script>' % \
                    (escape(newsponsorform.id), escape(newsponsorform)))

            messages.info(request, MSG_FORM_SAVED)
            return HttpResponseRedirect(reverse('sponsor.index'))
        else:
            messages.error(request, MSG_FORM_MISSING)
    else:
        sponsorform = SponsorForm(instance=sponsor, prefix='sponsor',
            collections_qset=user_collections)

    return render_to_response('journalmanager/add_sponsor.html', {
                              'add_form': sponsorform,
                              'user_name': request.user.pk,
                              },
                              context_instance=RequestContext(request))


@permission_required('journalmanager.change_collection', login_url=settings.AUTHZ_REDIRECT_URL)
def add_collection(request, collection_id):
    """
    Handles existing collections
    """

    collection = get_object_or_404(models.Collection, id=collection_id)

    if not collection.is_managed_by_user(request.user):
        return HttpResponseRedirect(settings.AUTHZ_REDIRECT_URL)

    if request.method == "POST":
        collectionform = CollectionForm(request.POST, request.FILES, instance=collection, prefix='collection')

        if collectionform.is_valid():
            collectionform.save()
            messages.info(request, MSG_FORM_SAVED)
            return HttpResponseRedirect(reverse('collection.edit', kwargs={'collection_id': collection_id}))
        else:
            messages.error(request, MSG_FORM_MISSING)
    else:
        collectionform = CollectionForm(instance=collection, prefix='collection')

    try:
        collection_logo = collection.logo.url
    except ValueError:
        collection_logo = False

    return render_to_response('journalmanager/add_collection.html', {
                              'add_form': collectionform,
                              'collection_logo': collection_logo,
                              'user_name': request.user.pk,
                              },
                              context_instance=RequestContext(request))


@permission_required('journalmanager.add_issue', login_url=settings.AUTHZ_REDIRECT_URL)
def edit_issue(request, journal_id, issue_id=None):
    """
    Handles edition of existing issues
    """

    def get_issue_form_by_type(issue_type, request, journal, instance):
        """
            This method is useful to get the correct IssueForm based on issue_type.
        """
        form_kwargs = {
            'params': {
                'journal': journal,
            },
            'querysets': {
                'section': journal.section_set.filter(is_trashed=False),
                'use_license': models.UseLicense.objects.all(),
            },
            'instance': instance,
        }

        form_args = []
        if request.method == 'POST':
            form_args.append(request.POST)
            form_args.append(request.FILES)
        else:
            initial = issue.__dict__
            initial['suppl_type'] = issue.suppl_type if issue.type == 'supplement' else None
            initial['use_license'] = issue.use_license
            initial['section'] = issue.section.all()
            initial['journal'] = issue.journal
            form_kwargs['initial'] = initial

        if issue_type == 'supplement':
            return SupplementIssueForm(*form_args, **form_kwargs)
        elif issue_type == 'special':
            return SpecialIssueForm(*form_args, **form_kwargs)
        else:  # issue_type == 'regular':
            return RegularIssueForm(*form_args, **form_kwargs)

    issue = models.Issue.objects.get(pk=issue_id)

    IssueTitleFormSet = inlineformset_factory(models.Issue, models.IssueTitle,
                                              form=IssueTitleForm, extra=1, can_delete=True)

    if request.method == 'POST':
        form = get_issue_form_by_type(issue.type, request, issue.journal, issue)
        titleformset = IssueTitleFormSet(request.POST, instance=issue, prefix='title')

        audit_old_values = helpers.collect_old_values(issue, form, [titleformset, ])

        if form.is_valid() and titleformset.is_valid():
            saved_issue = form.save(commit=False)
            saved_issue.journal = issue.journal
            saved_issue.save()
            form.save_m2m()

            if titleformset.is_valid():
                titleformset.save()

            audit_data = {
                'user': request.user,
                'obj': issue,
                'message': helpers.construct_change_message(form, [titleformset, ]),
                'old_values': audit_old_values,
                'new_values': helpers.collect_new_values(form, [titleformset, ]),
            }
            helpers.log_change(**audit_data)

            messages.info(request, MSG_FORM_SAVED)

            return HttpResponseRedirect(reverse('issue.index', args=[journal_id]))
        else:
            messages.error(request, MSG_FORM_MISSING)
    else:
        form = get_issue_form_by_type(issue.type, request, issue.journal, issue)
        titleformset = IssueTitleFormSet(instance=issue, prefix='title')

    # Recovering Journal Cover url.
    try:
        has_cover_url = issue.cover.url
    except ValueError:
        has_cover_url = False

    # variable names are add_form, and add_issue_xxx is that way to re-use the template
    return render_to_response('journalmanager/add_issue_%s.html' % issue.type, {
                              'add_form': form,
                              'issue_type': issue.type,
                              'journal': issue.journal,
                              'titleformset': titleformset,
                              'user_name': request.user.pk,
                              'has_cover_url': has_cover_url,
                              },
                              context_instance=RequestContext(request))


@permission_required('journalmanager.add_issue', login_url=settings.AUTHZ_REDIRECT_URL)
def add_issue(request, issue_type, journal_id, issue_id=None):
    """
    Handles new and existing issues
    """

    def get_issue_form_by_type(issue_type, request, journal, instance=None, initial=None):
        """
            This method is useful to get the correct IssueForm based on issue_type.
            if initial == None then the request.POST and request.FILES is used to fill the form.
            if initial != None then the form is used in a GET request.
        """
        form_kwargs = {
            'params': {
                'journal': journal,
            },
            'querysets': {
                'section': journal.section_set.filter(is_trashed=False),
                'use_license': models.UseLicense.objects.all(),
            },
            'instance': instance,
            'initial': initial,
        }
        form_args = []
        if initial is None:
            form_args.append(request.POST)
            form_args.append(request.FILES)
        if issue_type == 'supplement':
            return SupplementIssueForm(*form_args, **form_kwargs)
        elif issue_type == 'special':
            return SpecialIssueForm(*form_args, **form_kwargs)
        else:  # issue_type == 'regular':
            return RegularIssueForm(*form_args, **form_kwargs)

    journal = get_object_or_404(models.Journal.userobjects.active(), pk=journal_id)

    if issue_id is None:
        data_dict = {'use_license': journal.use_license.id,
                     'editorial_standard': journal.editorial_standard,
                     'ctrl_vocabulary': journal.ctrl_vocabulary}
        issue = models.Issue()

        # get last issue of the journal
        last_issue = journal.get_last_issue()
    else:
        data_dict = None
        issue = models.Issue.objects.get(pk=issue_id)

    IssueTitleFormSet = inlineformset_factory(models.Issue, models.IssueTitle,
                                              form=IssueTitleForm, extra=1, can_delete=True)

    if request.method == 'POST':
        add_form = get_issue_form_by_type(issue_type, request, journal, instance=issue)
        titleformset = IssueTitleFormSet(request.POST, instance=issue, prefix='title')

        if add_form.is_valid() and titleformset.is_valid():
            saved_issue = add_form.save(commit=False)
            saved_issue.journal = journal
            saved_issue.type = issue_type
            saved_issue.save()
            add_form.save_m2m()
            # the backward relation is created only
            # if title is given.
            if titleformset.is_valid():
                titleformset.save()

            # if is a new issue copy editorial board from the last issue
            if issue_id is None and last_issue:
                try:
                    members = last_issue.editorialboard.editorialmember_set.all()
                except ObjectDoesNotExist:
                    messages.info(request,
                        _(u"Issue created successfully, however we can not create the editorial board."))
                    notifications.issue_board_replica(issue, 'issue_add_no_replicated_board')
                else:
                    ed_board = EditorialBoard()
                    ed_board.issue = saved_issue
                    ed_board.save()

                    for member in members:
                        member.board = ed_board
                        member.pk = None
                        member.save()

                    notifications.issue_board_replica(issue, 'issue_add_replicated_board')

            audit_data = {
                'user': request.user,
                'obj': issue,
                'message': helpers.construct_create_message(add_form, [titleformset, ]),
                'old_values': '',
                'new_values': helpers.collect_new_values(add_form, [titleformset, ]),
            }
            helpers.log_create(**audit_data)

            messages.info(request, MSG_FORM_SAVED)

            return HttpResponseRedirect(reverse('issue.index', args=[journal_id]))
        else:
            messages.error(request, MSG_FORM_MISSING)
    else:
        add_form = get_issue_form_by_type(issue_type, request, journal, instance=issue, initial=data_dict)
        titleformset = IssueTitleFormSet(instance=issue, prefix='title')

    # Recovering Journal Cover url.
    try:
        has_cover_url = issue.cover.url
    except ValueError:
        has_cover_url = False
    return render_to_response('journalmanager/add_issue_%s.html' % issue_type, {
                              'add_form': add_form,
                              'issue_type': issue_type,
                              'journal': journal,
                              'titleformset': titleformset,
                              'user_name': request.user.pk,
                              'has_cover_url': has_cover_url,
                              },
                              context_instance=RequestContext(request))


@permission_required('journalmanager.change_section', login_url=settings.AUTHZ_REDIRECT_URL)
def add_section(request, journal_id, section_id=None):
    """
    Handles new and existing sections
    """
    journal = get_object_or_404(
        models.Journal.objects.all_by_user(request.user), pk=journal_id)

    if section_id is None:
        section = models.Section()
        has_relation = False
    else:
        section = get_object_or_404(models.Section, pk=section_id)
        has_relation = section.is_used()

    all_forms = get_all_section_forms(request.POST, journal, section)

    add_form = all_forms['section_form']
    section_title_formset = all_forms['section_title_formset']

    if request.method == 'POST':

        if add_form.is_valid() and section_title_formset.is_valid():
            add_form = add_form.save_all(journal)
            section_title_formset.save()

            if request.POST.get('popup', 0):
                return HttpResponse(u'<script type="text/javascript">\
                    opener.updateSelect(window, "%s", "%s", "id_section");</script>' % \
                    (escape(add_form.id), escape(add_form)))

            messages.info(request, MSG_FORM_SAVED)
            return HttpResponseRedirect(reverse('section.index', args=[journal_id]))
        else:
            messages.error(request, MSG_FORM_MISSING)

    return render_to_response('journalmanager/add_section.html', {
                              'add_form': add_form,
                              'section_title_formset': section_title_formset,
                              'user_name': request.user.pk,
                              'journal': journal,
                              'has_relation': has_relation,
                              }, context_instance=RequestContext(request))


@permission_required('journalmanager.delete_section', login_url=settings.AUTHZ_REDIRECT_URL)
def del_section(request, journal_id, section_id):
    section = get_object_or_404(models.Section, pk=section_id)

    if not section.is_used():
        section.is_trashed = True
        section.save()
        messages.success(request, _(u'Section removed successfully'))
    else:
        messages.info(
            request,
            _(u"Can't delete, some issues are using this Section")
        )

    return HttpResponseRedirect(
        reverse('section.index', args=[section.journal.id])
    )


@login_required
def toggle_user_availability(request, user_id):

    if request.is_ajax():
        user = get_object_or_404(models.User, pk=user_id)
        user.is_active = not user.is_active
        user.save()

        response_data = json.dumps({
          "result": str(user.is_active),
          "object_id": user.id
          })

        return HttpResponse(response_data, mimetype="application/json")
    else:
        return HttpResponse(status=400)


@login_required
def trash_listing(request):
    listing_ref = {
        'journal': models.Journal,
        'section': models.Section,
        'sponsor': models.Sponsor,
    }

    if request.GET.get('show', None) in listing_ref:
        doc_entity = listing_ref[request.GET['show']]
    else:
        doc_entity = models.Journal

    try:
        trashed_docs = doc_entity.objects.all_by_user(request.user, is_available=False)
    except AttributeError:
        trashed_docs = models.Journal.objects.all_by_user(request.user, is_available=False)

    trashed_docs_paginated = get_paginated(trashed_docs, request.GET.get('page', 1))

    return render_to_response(
        'journalmanager/trash_listing.html',
        {'trashed_docs': trashed_docs_paginated},
        context_instance=RequestContext(request))


@login_required
def ajx_list_users(request):
    """
    Lists the users acoording to the a given string.
    """
    if not request.is_ajax():
        return HttpResponse(status=400)

    users = User.objects.all()

    response_data = [user.username for user in users]

    return HttpResponse(json.dumps(response_data), mimetype="application/json")


@login_required
def ajx_search_journal(request):
    """
    Ajax view function to search the journal by: ``title``, ``short_title``,
    ``print_issn``, ``eletronic_issn`` and ``acronym``
    """

    if not request.is_ajax():
        return HttpResponse(status=400)

    if 'q' in request.GET:
        query = request.GET.get('q')
        journals = models.Journal.objects.filter(Q(title__icontains=query) |
                                                 Q(short_title__icontains=query) |
                                                 Q(print_issn__icontains=query) |
                                                 Q(eletronic_issn__icontains=query) |
                                                 Q(acronym__icontains=query))

        data = [{'id': journal.id, 'title': journal.title, 'print_issn': journal.print_issn,
                 'eletronic_issn': journal.eletronic_issn, 'short_title': journal.short_title,
                 'acronym': journal.acronym, 'collections': [collection.name
                     for collection in journal.collections.all()]}
                 for journal in journals]

        return HttpResponse(json.dumps({'data': data}),
                            mimetype='application/json')


@login_required
def ajx_add_journal_to_user_collection(request, journal_id):
    """
    Add journal to the user current collection
    """

    if not request.is_ajax():
        return HttpResponse(status=400)

    user_collection = user_request_context.get_current_user_active_collection()

    journal = models.Journal.objects.get(id=journal_id)

    if journal.is_member(user_collection):
        response = {
            'journal': journal.title,
            'collection': user_collection.name,
            'assignment': False,
        }

    else:
        # The journal is join to new collection with status ``inprogress``
        journal.join(user_collection, request.user)
        messages.error(request, _(u'{journal} add to collection {collection}'.format(
            journal=journal,
            collection=user_collection)))

        response = {
            'journal': journal.title,
            'collection': user_collection.name,
            'assignment': True,
        }

    return HttpResponse(json.dumps(response), mimetype='application/json')


@login_required
def ajx_list_issues_for_markup_files(request):
    """
    Lists the issues of a given journal to be used by the
    ``markup_files.html`` page.

    The following values must be passed as querystring parameters:

    ``j`` is a journal's id

    ``all`` is a boolean value that returns all issues
    from a journal or only the ones containing the attribute
    ``is_marked_up`` set to False.
    """
    if not request.is_ajax():
        return HttpResponse(status=400)

    journal_id = request.GET.get('j', None)

    if journal_id is None:
        # journal id is required is None -> Raise Bad Request
        return HttpResponse(status=400)

    journal = get_object_or_404(models.Journal, pk=journal_id)

    all_issues = asbool(request.GET.get('all', True))

    journal_issues = journal.issue_set.all()

    if not all_issues:
        journal_issues = journal_issues.filter(is_marked_up=False)

    issues = []
    for issue in journal_issues:
        text = u'{0} - {1}'.format(issue.publication_year, issue.label)
        issues.append({'id': issue.pk, 'text': text})

    response_data = json.dumps(issues)

    return HttpResponse(response_data, mimetype="application/json")


@login_required
def ajx_lookup_for_section_translation(request):
    """
    Says if a given translation already exists in a given
    journal.

    The following values must be passed as querystring parameters:

    ``j`` is a journal's id
    ``t`` is a urlencoded section translation
    ``exc`` is a section id to be excluded from the search
    """
    MSG_EXISTS = _('The section already exists.')
    MSG_NOT_EXISTS = _('This is a new section.')

    if not request.is_ajax():
        return HttpResponse(status=400)

    journal_id = request.GET.get('j', None)
    if not journal_id:
        return HttpResponse(status=400)

    section_title = request.GET.get('t', None)
    if not section_title:
        return HttpResponse(status=400)

    try:
        exclude = int(request.GET.get('exc', 0))
    except ValueError:
        return HttpResponse(status=400)

    found_secs = models.Section.userobjects.all().available().filter(
        journal__pk=journal_id, titles__title=section_title)

    sections = [[unicode(sec), sec.actual_code] for sec in found_secs if sec.pk != exclude]
    has_sections = bool(sections)
    data = {
        'exists': has_sections,
        'sections': sections,
        'message': MSG_EXISTS if has_sections else MSG_NOT_EXISTS,
    }

    response_data = json.dumps(data)

    return HttpResponse(response_data, mimetype="application/json")


@permission_required('journalmanager.add_pressrelease', login_url=settings.AUTHZ_REDIRECT_URL)
def add_pressrelease(request, journal_id, prelease_id=None):
    journal = get_object_or_404(models.Journal, pk=journal_id)

    if prelease_id:
        pressrelease = get_object_or_404(models.RegularPressRelease,
                                         pk=prelease_id)
    else:
        pressrelease = models.RegularPressRelease()

    pr_forms = get_all_pressrelease_forms(request.POST, journal, pressrelease)

    pressrelease_form = pr_forms['pressrelease_form']
    translation_formset = pr_forms['translation_formset']
    article_formset = pr_forms['article_formset']

    if request.method == 'POST':

        if pressrelease_form.is_valid() and translation_formset.is_valid():
            pressrelease_form.save()
            translation_formset.save()

            if article_formset.is_valid():
                article_formset.save()

            messages.info(request, MSG_FORM_SAVED)

            return HttpResponseRedirect(reverse('prelease.index', args=[journal_id]))
        else:
            messages.error(request, MSG_FORM_MISSING)

    return render_to_response(
        'journalmanager/add_pressrelease.html',
        {
            'pressrelease_form': pressrelease_form,
            'translation_formset': translation_formset,
            'article_formset': article_formset,
            'journal': journal,
        },
        context_instance=RequestContext(request)
    )


@permission_required('journalmanager.add_pressrelease', login_url=settings.AUTHZ_REDIRECT_URL)
def add_aheadpressrelease(request, journal_id, prelease_id=None):
    journal = get_object_or_404(models.Journal, pk=journal_id)

    if prelease_id:
        pressrelease = get_object_or_404(models.AheadPressRelease,
                                         pk=prelease_id)
    else:
        pressrelease = models.AheadPressRelease()

    pr_forms = get_all_ahead_pressrelease_forms(request.POST, journal, pressrelease)

    pressrelease_form = pr_forms['pressrelease_form']
    translation_formset = pr_forms['translation_formset']
    article_formset = pr_forms['article_formset']

    if request.method == 'POST':
        if (pressrelease_form.is_valid() and
                translation_formset.is_valid() and
                article_formset.is_valid()):

            pr = pressrelease_form.save(commit=False)
            pr.journal = journal
            pr.save()
            translation_formset.save()
            article_formset.save()

            messages.info(request, MSG_FORM_SAVED)

            return HttpResponseRedirect(reverse('prelease.index', args=[journal_id]) + '?tab=ahead')
        else:
            messages.error(request, MSG_FORM_MISSING)

    return render_to_response(
        'journalmanager/add_pressrelease.html',
        {
            'pressrelease_form': pressrelease_form,
            'translation_formset': translation_formset,
            'article_formset': article_formset,
            'journal': journal,
        },
        context_instance=RequestContext(request)
    )
