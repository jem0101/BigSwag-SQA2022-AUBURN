
"""
    When you start having a lot of routes, you can give them names, and
    refer to this name later, when you want a link for them.

    Here we will use the SAME URLs and views as in the previous app,
    but we will give them names.
"""

from django.urls import path, include

# it's possible use whole prefixing as well to avoid name conflicts
# The choice is yours, there is no better way
import app1_hello.views
import app3_basic_routing.views
import app4_links.views


urlpatterns = [
    # Giving a name is as simple as passing a "name" parameter.
    # We will use this name in views.py, so go to this file to see what
    # we do with it.
    path("prefix/", app3_basic_routing.views.prefix, name="prefix"),

    # Starting from here, we will use names in the template,
    # so look at templates/app4_index.html to see how we use this name.
    path("hello_from_app1/", app1_hello.views.hello, name="hello"),

    # Naming routes doesn't work with include. This will not do what you
    # expect. To benefit from name, you should name routes from
    # the included urls.py.
    path("app2_included/", include("app2_hello_again.urls"), name="include"),

    path(
        "<str:name>/<str:prefix>/", app3_basic_routing.views.hello, name="hello_prefix"
    ),
    path("<str:name>/", app3_basic_routing.views.hello, name="hello_name"),
    path("", app4_links.views.index),
]
