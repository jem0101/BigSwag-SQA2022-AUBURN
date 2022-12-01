
from django.urls import path
from django.views.generic import TemplateView

import app8_base.views

urlpatterns = [

    path(
        'example1/',
        TemplateView.as_view(template_name="app8_base/example_child.html"),
        name="useless_view1"
    ),

    path(
        'example2/',
        TemplateView.as_view(template_name="app8_base/example_child2.html"),
        name="useless_view2"
    ),

    # Here we use our index view. It's a reusable view, in the sense that
    # it's very configurable using parameters.
    # All values in "kwargs" will be passed as parameters to index(). Each
    # call to "index()" will look like something like this :
    # index(request, **kwargs).
    # We use this to pass a header, a footer and the examples to list in
    # our index. By changing just these values in the next apps, we will be
    # able to reuse the Python code and the template.
    path(
        '', app8_base.views.index,
        kwargs={
            'title': 'Making a base application',
            'examples': [
                ('useless_view1', 'This does very little'),
                ('useless_view2', 'So does this')
            ],
            'header': 'This index view is coded to be generic and reusable.',
            'footer': 'Here we can optionally write a footer',
        }
    ),

    # Don't be afraid by the illusion of complexity here. It's still just
    # path(pattern, view, parameters). There parameters are here "kwargs" and
    # it's value. It's value is a dictionary. The dictionary has 4 keys : title,
    # examples, header, and footer. The 'examples' key has a list of tuples for
    # value, while 'title', 'header' and 'footer' have a string for value. This
    # is all nested, and that's why it looks packed, but it really is just
    # ordinary Python objects mixed together in a function call.

]
