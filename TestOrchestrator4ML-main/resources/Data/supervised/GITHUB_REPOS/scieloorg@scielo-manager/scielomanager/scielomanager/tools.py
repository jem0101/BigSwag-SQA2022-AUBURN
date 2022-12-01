# coding: utf-8
try:
    from hashlib import md5
except ImportError:
    from md5 import new as md5
import re

from django.db.models.sql.datastructures import EmptyResultSet
from django.core.paginator import EmptyPage
from django.core.paginator import Paginator
from django.contrib.auth.models import Group, User
from django.core import exceptions

from django.conf import settings


class NullPaginator(object):
    """
    A null object implementation for a Paginator.
    http://en.wikipedia.org/wiki/Null_Object_pattern
    """
    def __getattr__(self, name):
        return None


def has_changed(instance, field):
    """
    This function return the boolean value ``True`` if field of any instance
    was changed and ``False`` otherwise.
    Raise DoesNotExist except if the instance doesn`t exist in the database and
    raise FieldError if field doesn`t exist.
    """

    if not instance.pk:
        raise instance.DoesNotExist('%s must be saved in order to use this function' % instance)

    old_value = instance.__class__._default_manager.filter(pk=instance.pk).values(field).get()[field]

    return not getattr(instance, field) == old_value


def get_paginated(items, page_num, items_per_page=settings.PAGINATION__ITEMS_PER_PAGE):
    """
    Wraps django core pagination object
    """
    paginator = Paginator(items, items_per_page)

    try:
        page_num = int(page_num)
    except ValueError:
        raise TypeError('page_num must be integer')

    try:
        paginated = paginator.page(page_num)
    except EmptyPage:
        paginated = paginator.page(paginator.num_pages)
    except EmptyResultSet:
        paginated = NullPaginator()

    return paginated


# Copyright (c) 2009 Arthur Furlan <arthur.furlan@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# On Debian systems, you can find the full text of the license in
# /usr/share/common-licenses/GPL-2


def get_referer_view(request, default=None):
    '''
    Return the referer view of the current request

    Example:

        def some_view(request):
            ...
            referer_view = get_referer_view(request)
            return HttpResponseRedirect(referer_view, '/accounts/login/')
    '''

    # if the user typed the url directly in the browser's address bar
    referer = request.META.get('HTTP_REFERER')
    if not referer:
        return default

    # remove the protocol and split the url at the slashes
    referer = re.sub('^https?:\/\/', '', referer).split('/')
    #if referer[0] != request.META.get('SERVER_NAME'):
        #return default

    # add the slash at the relative path's view and finished
    referer = u'/' + u'/'.join(referer[1:])
    return referer

# Taken from Pyramid framework
# https://github.com/Pylons/pyramid/blob/master/pyramid/settings.py
truthy = frozenset(('t', 'true', 'y', 'yes', 'on', '1'))


def asbool(s):
    """ Return the boolean value ``True`` if the case-lowered value of string
    input ``s`` is any of ``t``, ``true``, ``y``, ``on``, or ``1``, otherwise
    return the boolean value ``False``. If ``s`` is the value ``None``,
    return ``False``. If ``s`` is already one of the boolean values ``True``
    or ``False``, return it."""
    if s is None:
        return False
    if isinstance(s, bool):
        return s
    s = str(s).strip()
    return s.lower() in truthy


def get_users_by_group(group):
    """
    Get all users from a Group with name:``group`` or raise a ObjectDoesNotExist
    """
    target_group = Group.objects.get(name=group)
    return target_group.user_set.all()

def get_users_by_group_by_collections(group_name, collections):
    """
    return a list of users that belong to a group with name:
    @param: group_name and associated with ANY collection in:
    @param: collections list
    """
    users = User.objects.filter(groups__name=group_name, user_collection__in=collections)
    return users

def user_receive_emails(user):
    if user.get_profile():
        return user.get_profile().email_notifications
    else:
        return False


def get_setting_or_raise(name):
    """ Retorna a diretiva de configuração `name` ou levanta exceção.
    """
    try:
        setting = getattr(settings, name)
    except AttributeError:
        raise exceptions.ImproperlyConfigured('Setting "%s" is missing' % name)

    return setting
