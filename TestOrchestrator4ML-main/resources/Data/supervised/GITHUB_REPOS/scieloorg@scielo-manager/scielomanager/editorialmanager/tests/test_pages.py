# coding: utf-8

from django_webtest import WebTest
from django.core.urlresolvers import reverse
from django.conf import settings
from journalmanager.tests import modelfactories
from . import modelfactories as editorial_modelfactories
from .test_forms import _makePermission
from editorialmanager.models import EditorialMember, EditorialBoard


def _set_permission_to_group(perm, group):
    perm_add_editorialmember = _makePermission(perm=perm, model='editorialmember')
    group.permissions.add(perm_add_editorialmember)


class PagesAsEditorTests(WebTest):

    def setUp(self):
        # create a group 'Editors'
        self.group = modelfactories.GroupFactory(name="Editors")
        # create a user and set group 'Editors'
        self.user = modelfactories.UserFactory(is_active=True)
        self.user.groups.add(self.group)

        self.collection = modelfactories.CollectionFactory.create()
        self.collection.add_user(self.user, is_manager=False)
        self.collection.make_default_to_user(self.user)

        self.journal = modelfactories.JournalFactory.create()
        self.journal.join(self.collection, self.user)

        # set the user as editor of the journal
        self.journal.editor = self.user

        # create an issue
        self.issue = modelfactories.IssueFactory.create()
        self.issue.journal = self.journal
        self.journal.save()
        self.issue.save()

    def tearDown(self):
        pass

    def test_logged_user_access_to_index(self):
        """
        User loggin and access 'editorial.index' page,
        the result must show the journal who is editable by the logged user
        """
        # when
        response = self.app.get(reverse('editorial.index'), user=self.user)

        # then
        self.assertTrue(self.user.get_profile().is_editor or self.user.get_profile().is_librarian)
        self.assertTemplateUsed(response, 'journal/journal_list.html')

        journals_in_response = response.context['objects_journal']
        self.assertIsNotNone(journals_in_response)
        journals_in_response = journals_in_response.object_list
        self.assertIn(self.journal, journals_in_response)
        response.mustcontain(self.journal.title)
        journal_detail_link_url = reverse("editorial.journal.detail", args=[self.journal.pk])
        response.mustcontain(journal_detail_link_url)

    def test_logged_user_access_to_journal_detail(self):
        # when
        response = self.app.get(reverse("editorial.journal.detail", args=[self.journal.pk]), user=self.user)
        # then
        self.assertTemplateUsed(response, 'journal/journal_detail.html')
        journal_in_response = response.context['journal']
        self.assertIsNotNone(journal_in_response)
        response.mustcontain(self.journal.title)

    def test_user_access_links_to_add_when_have_permissions(self):
        # with
        member = editorial_modelfactories.EditorialMemberFactory.create()
        member.board = EditorialBoard.objects.create(issue=self.issue)
        member.save()
        add_url = reverse("editorial.board.add", args=[self.journal.id, self.issue.id])
        board_url = reverse("editorial.board", args=[self.journal.pk])
        # when
        response = self.app.get(board_url, user=self.user)

        # then

        # have no link to ADD, EDIT or DELETE a board member
        self.assertNotIn(add_url, response.body)
        # when gain permission to ADD editorial member, the link is shown
        _set_permission_to_group('add_editorialmember', self.group)
        response = self.app.get(board_url, user=self.user)
        self.assertEqual(response.status_code, 200)
        self.assertIn(add_url, response.body)

    def test_user_access_links_to_edit_when_have_permissions(self):
        # with
        member = editorial_modelfactories.EditorialMemberFactory.create()
        member.board = EditorialBoard.objects.create(issue=self.issue)
        member.save()
        edit_url = reverse("editorial.board.edit", args=[self.journal.id, member.id])
        board_url = reverse("editorial.board", args=[self.journal.pk])
        # when
        response = self.app.get(board_url, user=self.user)

        # then
        # have no link to ADD, EDIT or DELETE a board member
        self.assertNotIn(edit_url, response.body)
        # when gain permission to EDIT editorial member, the link is shown
        _set_permission_to_group('change_editorialmember', self.group)
        response = self.app.get(board_url, user=self.user)
        self.assertEqual(response.status_code, 200)
        self.assertIn(edit_url, response.body)

    def test_user_access_links_to_delete_when_have_permissions(self):
        # with
        member = editorial_modelfactories.EditorialMemberFactory.create()
        member.board = EditorialBoard.objects.create(issue=self.issue)
        member.save()
        delete_url = reverse("editorial.board.delete", args=[self.journal.id, member.id])
        board_url = reverse("editorial.board", args=[self.journal.pk])
        # when
        response = self.app.get(reverse("editorial.board", args=[self.journal.pk]), user=self.user)

        # then

        # have no link to ADD, EDIT or DELETE a board member
        self.assertNotIn(delete_url, response.body)

        # when gain permission to DELETE editorial member, the link is shown
        _set_permission_to_group('delete_editorialmember', self.group)
        response = self.app.get(board_url, user=self.user)
        self.assertEqual(response.status_code, 200)
        self.assertIn(delete_url, response.body)

    def test_unauthorized_user_access_to_add_member_page(self):
        """
        A user without permissions to EDIT board member, when try to access the *EDIT* page,
        will be redirected to ``settings.AUTHZ_REDIRECT_URL``
        """
        # when
        response = self.app.get(
            reverse("editorial.board.add", args=[self.journal.id, self.issue.id]),
            user=self.user,
            expect_errors=True)
        # then
        self.assertEqual(response.status_code, 302)
        self.assertIn(settings.AUTHZ_REDIRECT_URL, response.location)
        self.assertTemplateUsed(response.follow(), 'accounts/unauthorized.html')

    def test_authorized_user_access_to_add_member_page(self):
        """
        A user with permissions to ADD board member, when try to access the *ADD* page,
        will see the correct page
        """
        # with
        _set_permission_to_group('add_editorialmember', self.group)
        # when
        response = self.app.get(
            reverse("editorial.board.add", args=[self.journal.id, self.issue.id]),
            user=self.user)
        # when
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/board_member_edit.html')

    def test_unauthorized_user_access_to_edit_member_page(self):
        """
        A user without permissions to EDIT board member, when try to access the *EDIT* page,
        will be redirected to ``settings.AUTHZ_REDIRECT_URL``
        """
        # with
        member = editorial_modelfactories.EditorialMemberFactory.create()
        member.board = EditorialBoard.objects.create(issue=self.issue)
        member.save()
        # when
        response = self.app.get(
            reverse("editorial.board.edit", args=[self.journal.id, member.id]),
            user=self.user,
            expect_errors=True)
        # then
        self.assertEqual(response.status_code, 302)
        self.assertIn(settings.AUTHZ_REDIRECT_URL, response.location)
        self.assertTemplateUsed(response.follow(), 'accounts/unauthorized.html')

    def test_authorized_user_access_to_edit_member_page(self):
        """
        A user with permissions to EDIT board member, when try to access the *EDIT* page,
        will see the correct page
        """
        # with
        member = editorial_modelfactories.EditorialMemberFactory.create()
        member.board = EditorialBoard.objects.create(issue=self.issue)
        member.save()
        _set_permission_to_group('change_editorialmember', self.group)
        # when
        response = self.app.get(
            reverse("editorial.board.edit", args=[self.journal.id, member.id]),
            user=self.user)
        # when
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/board_member_edit.html')

    def test_unauthorized_user_access_to_delete_member_page(self):
        """
        A user without permissions to DELETE board member, when try to access the *DELETE* page,
        will be redirected to ``settings.AUTHZ_REDIRECT_URL``
        """
        # with
        member = editorial_modelfactories.EditorialMemberFactory.create()
        member.board = EditorialBoard.objects.create(issue=self.issue)
        member.save()
        # when
        response = self.app.get(
            reverse("editorial.board.delete", args=[self.journal.id, member.id]),
            user=self.user,
            expect_errors=True)
        # then
        self.assertEqual(response.status_code, 302)
        self.assertIn(settings.AUTHZ_REDIRECT_URL, response.location)
        self.assertTemplateUsed(response.follow(), 'accounts/unauthorized.html')

    def test_authorized_user_access_to_delete_member_page(self):
        """
        A user with permissions to DELETE board member, when try to access the *DELETE* page,
        will see the correct page
        """
        # with
        member = editorial_modelfactories.EditorialMemberFactory.create()
        member.board = EditorialBoard.objects.create(issue=self.issue)
        member.save()
        _set_permission_to_group('delete_editorialmember', self.group)
        # when
        response = self.app.get(
            reverse("editorial.board.delete", args=[self.journal.id, member.id]),
            user=self.user)
        # when
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/board_member_delete.html')


class PagesAsLibrarianTests(PagesAsEditorTests):
    """
    Excecute the same tests that an Editors (PagesAsEditorTests), the setUp is almost the same.
    Only change is that the self.user is assigned as a member of "Librarian" group instead of "Editors" group.
    """
    def setUp(self):
        super(PagesAsLibrarianTests, self).setUp()
        # change user group to belong to Librarian group
        self.user.groups.clear()
        self.group = modelfactories.GroupFactory(name="Librarian")
        self.user.groups.add(self.group)
        self.user.save()

    def tearDown(self):
        super(PagesAsLibrarianTests, self).tearDown()


class RoleType(WebTest):

    def setUp(self):
        # create a group 'Editors'
        group = modelfactories.GroupFactory(name="Editors")
        # create a user and set group 'Editors'
        self.user = modelfactories.UserFactory(is_active=True)
        self.user.groups.add(group)

        self.collection = modelfactories.CollectionFactory.create()
        self.collection.add_user(self.user, is_manager=False)
        self.collection.make_default_to_user(self.user)

        self.journal = modelfactories.JournalFactory.create()
        self.journal.join(self.collection, self.user)

        # set the user as editor of the journal
        self.journal.editor = self.user

        # create an issue
        self.issue = modelfactories.IssueFactory.create()
        self.issue.journal = self.journal
        self.journal.save()
        self.issue.save()

    def test_access_to_ADD_ROLE_button_is_DISABLED(self):
        """
        User must have the permission: editorialmanager.add_roletype to see the button
        """
        # when
        response = self.app.get(reverse("editorial.board", args=[self.journal.id, ]), user=self.user)

        # then
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/board_list.html')
        add_role_url = reverse("editorial.role.add", args=[self.journal.id])
        self.assertNotIn(add_role_url, response.body)

    def test_access_to_ADD_ROLE_button_is_ENABLE(self):
        """
        User must have the permission: editorialmanager.add_roletype to see the button
        """
        # with
        perm_add_roletype = _makePermission(perm='add_roletype', model='roletype')
        self.user.user_permissions.add(perm_add_roletype)
        # when
        response = self.app.get(reverse("editorial.board", args=[self.journal.id, ]), user=self.user)

        # then
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/board_list.html')
        add_role_url = reverse("editorial.role.add", args=[self.journal.id])
        self.assertIn(add_role_url, response.body)

    def test_access_to_EDIT_ROLE_button_is_DISABLED(self):
        """
        User must have the permission: editorialmanager.change_roletype to see the button
        """
        # with
        board = EditorialBoard.objects.create(issue=self.issue)
        role = editorial_modelfactories.RoleTypeFactory.create()
        member = editorial_modelfactories.EditorialMemberFactory.create(board=board, role=role)
        # when
        response = self.app.get(reverse("editorial.board", args=[self.journal.id, ]), user=self.user)

        # then
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/board_list.html')
        edit_role_url = reverse("editorial.role.edit", args=[self.journal.id, role.id])
        self.assertNotIn(edit_role_url, response.body)

    def test_access_to_EDIT_ROLE_button_is_ENABLE(self):
        """
        User must have the permission: editorialmanager.change_roletype to see the button
        """
        # with
        board = EditorialBoard.objects.create(issue=self.issue)
        role = editorial_modelfactories.RoleTypeFactory.create()
        member = editorial_modelfactories.EditorialMemberFactory.create(board=board, role=role)
        # add perms
        perm_change_roletype = _makePermission(perm='change_roletype', model='roletype')
        self.user.user_permissions.add(perm_change_roletype)
        # when
        response = self.app.get(reverse("editorial.board", args=[self.journal.id, ]), user=self.user)

        # then
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/board_list.html')
        edit_role_url = reverse("editorial.role.edit", args=[self.journal.id, role.id])
        self.assertIn(edit_role_url, response.body)

    def test_access_to_role_list_link(self):
        """
        User must not have any particular permission
        """
        # with
        board = EditorialBoard.objects.create(issue=self.issue)
        # when
        response = self.app.get(reverse("editorial.board", args=[self.journal.id, ]), user=self.user)

        # then
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/board_list.html')
        list_role_url = reverse("editorial.role.list", args=[self.journal.id, ])
        self.assertIn(list_role_url, response.body)

    def test_access_to_EDIT_and_TRANSLATE_from_role_list_DISABLE(self):
        """
        User must not have any particular permission, but cant see the edit nor translate buttons
        """
        # with
        board = EditorialBoard.objects.create(issue=self.issue)
        role = editorial_modelfactories.RoleTypeFactory.create(name='blaus!!!')
        # when
        response = self.app.get(reverse("editorial.role.list", args=[self.journal.id, ]), user=self.user)

        # then
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/role_type_list.html')
        self.assertIn(role.name, response.body)
        self.assertIn(role, response.context['roles'])
        edit_role_url = reverse("editorial.role.edit", args=[self.journal.id, role.id])
        self.assertNotIn(edit_role_url, response.body)
        translate_role_url = reverse('editorial.role.translate', args=[self.journal.id, role.id])
        self.assertNotIn(translate_role_url, response.body)

    def test_access_to_EDIT_and_TRANSLATE_from_role_list_ENABLE(self):
        """
        If user have permissions to change_roletype, must see EDIT and TRANSLATE buttons in role's list
        """
        # with
        board = EditorialBoard.objects.create(issue=self.issue)
        role = editorial_modelfactories.RoleTypeFactory.create(name='blaus!!!')
        # add perms
        perm_change_roletype = _makePermission(perm='change_roletype', model='roletype')
        self.user.user_permissions.add(perm_change_roletype)
        # when
        response = self.app.get(reverse("editorial.role.list", args=[self.journal.id, ]), user=self.user)

        # then
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'board/role_type_list.html')
        self.assertIn(role.name, response.body)
        edit_role_url = reverse("editorial.role.edit", args=[self.journal.id, role.id])
        self.assertIn(edit_role_url, response.body)
        translate_role_url = reverse('editorial.role.translate', args=[self.journal.id, role.id])
        self.assertIn(translate_role_url, response.body)


class DownloadMemberCSVFileTests(WebTest):

    def setUp(self):
        # create a group 'Librarian'
        self.group = modelfactories.GroupFactory(name="Librarian")
        # create a user and set group 'Librarian'
        self.user = modelfactories.UserFactory(is_active=True)
        self.user.groups.add(self.group)
        self.user.save()

        self.collection = modelfactories.CollectionFactory.create()
        self.collection.add_user(self.user, is_manager=False)
        self.collection.make_default_to_user(self.user)

        self.journal = modelfactories.JournalFactory.create()
        self.journal.join(self.collection, self.user)

        # create an issue
        self.issue = modelfactories.IssueFactory.create()
        self.issue.journal = self.journal
        self.journal.save()
        self.issue.save()

    def test_non_authenticated_users_are_redirected_to_login_page(self):
        response = self.app.get(
            reverse('editorial.export.csv.journal', args=[self.journal.id]),
            status=302
        ).follow()

        self.assertTemplateUsed(response, 'registration/login.html')

    def test_authenticated_users_can_access(self):
        from django.template.defaultfilters import slugify

        response = self.app.get(
            reverse('editorial.export.csv.issue', args=[self.journal.id, self.issue.id]),
            user=self.user
        )
        journal_slug = slugify(self.journal.title)
        issue_year = self.issue.publication_year
        issue_volume = self.issue.volume
        issue_number = self.issue.number
        filename = u'board_%s_%s_v%s_n%s' % (journal_slug, issue_year, issue_volume, issue_number)
        self.assertEquals(response.content_disposition,
                          'attachment; filename="%s.csv"' % filename)

    def test_authenticated_users_download_content(self):

        member = editorial_modelfactories.EditorialMemberFactory.create()
        member.board = EditorialBoard.objects.create(issue=self.issue)
        member.save()

        response = self.app.get(
            reverse('editorial.export.csv.issue', args=[self.journal.id, self.issue.id]),
            user=self.user
        )

        expected = u'journal, issn_print, issn_eletronic, issue_year, issue_volume, issue_number, role_name, first_name, last_name, full_name, email, institution, link_cv, state, country, country_code, country_code_alpha3, research_id, orcid\r\n"%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s"\r\n' % (self.journal.title, self.journal.print_issn, self.journal.eletronic_issn, self.issue.publication_year, self.issue.volume, self.issue.number, member.role.name, member.first_name, member.last_name, member.first_name + ' ' + member.last_name, member.email, member.institution, member.link_cv, member.state, member.country.name, member.country, member.country.alpha3, member.research_id, member.orcid)

        self.assertEqual(response.content, expected.encode('utf-8'))
