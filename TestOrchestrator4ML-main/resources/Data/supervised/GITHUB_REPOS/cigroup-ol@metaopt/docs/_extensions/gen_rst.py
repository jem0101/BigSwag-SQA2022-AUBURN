import os
from example_builder import ExampleBuilder

RST_TEMPLATE = """

.. _%(sphinx_tag)s:

%(docstring)s

%(image_list)s

.. raw:: html

    <div class="toggle_trigger"><a href="#">

**Code output:**

.. raw:: html

    </a></div>
    <div class="toggle_container">

.. literalinclude:: %(stdout)s

.. raw:: html

    </div>
    <div class="toggle_trigger" id="start_open"><a href="#">

**Python source code:**

.. raw:: html

    </a></div>
    <div class="toggle_container">

.. literalinclude:: %(fname)s
    :lines: %(end_line)s-

.. raw:: html

    </div>
    <div align="right">

:download:`[download source: %(fname)s] <%(fname)s>`

.. raw:: html

    </div>

"""

class CustomExampleBuilder(ExampleBuilder):
    def subdir_contents(self, path, subdirs):
        subdirs = [os.path.join(path, subdir) for subdir in subdirs]

        subdir_contents = ("\n\n"
                           ".. toctree::\n"
                           "   :maxdepth: 2\n\n")

        # for subdir in subdirs:
        #     index = os.path.splitext(self.rst_index_filename(subdir))[0]
        #     subdir_contents += '   %s\n' % os.path.relpath(index, path)

        subdir_contents += '\n'
        return subdir_contents


def main(app):
    target_dir = os.path.join(app.builder.srcdir, 'examples')
    source_dir = os.path.abspath(app.builder.srcdir +  '/../'
        + 'examples/showcase')

    try:
        plot_gallery = eval(app.builder.config.plot_gallery)
    except TypeError:
        plot_gallery = bool(app.builder.config.plot_gallery)

    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    EB = CustomExampleBuilder(source_dir, target_dir,
                              execute_files=plot_gallery,
                              dir_info_file='README.rst',
                              sphinx_tag_base='example',
                              template_example=RST_TEMPLATE)
    EB.run()

def setup(app):
    app.connect('builder-inited', main)
    app.add_config_value('plot_gallery', True, 'html')
