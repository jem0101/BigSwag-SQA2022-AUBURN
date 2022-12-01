from __future__ import absolute_import
"""
==============================
Add-on Management (``addons``)
==============================

.. index:: add-ons

Orange.utils.addons module provides a framework for Orange add-on management. As
soon as it is imported, the following initialization takes place: the list of
installed add-ons is loaded, their directories are added to python path
(:obj:`sys.path`) the callback list is initialized, the stored repository list is
loaded. The most important consequence of importing the module is thus the
injection of add-ons into the namespace.

"""

#TODO Document this module.

import shelve
import anydbm
import xmlrpclib
import warnings
import re
import pkg_resources
import tempfile
import tarfile
import zipfile
import shutil
import os
import sys
import platform
import subprocess
import errno
import urllib2
import site
import itertools
import pipes

from collections import namedtuple, defaultdict
from contextlib import closing

import Orange.utils.environ

ADDONS_ENTRY_POINT = "orange.addons"


OrangeAddOn = namedtuple(
    'OrangeAddOn',
    ['name', 'available_version', 'installed_version', 'summary',
     'description', 'author', 'docs_url', 'keywords', 'homepage',
     'package_url', 'release_urls']
)

ReleaseUrl = namedtuple(
    "ReleaseUrl",
    ["filename", "url", "size", "python_version", "packagetype"]
)

# It'd be great if we could somehow read a list and descriptions of
# widgets, show them in the dialog and enable search of add-ons
# based on keywords in widget names and descriptions.

INDEX_RE = "[^a-z0-9-']"  # RE for splitting entries in the search index

AOLIST_FILE = os.path.join(Orange.utils.environ.orange_settings_dir,
                           "addons_v2.shelve")


def open_addons(flag="r"):
    try:
        addons = shelve.open(AOLIST_FILE, flag)
    except anydbm.error as ex:
        if flag in ["r", "w"] and ex.message.startswith("need 'c'"):
            # Need to create it it first.
            s = shelve.open(AOLIST_FILE, "c")
            s.close()
            addons = shelve.open(AOLIST_FILE, flag)
        else:
            if os.path.isfile(AOLIST_FILE):
                os.remove(AOLIST_FILE)
            addons = shelve.open(AOLIST_FILE, 'n')
    except ImportError:
        if os.path.isfile(AOLIST_FILE):
            os.remove(AOLIST_FILE)
        addons = shelve.open(AOLIST_FILE, 'n')
    else:
        # Try to read the whole list and check for sanity.
        if any(name != name.lower() for name, _ in addons.items()):
            addons.close()
            if os.path.isfile(AOLIST_FILE):
                os.remove(AOLIST_FILE)
            addons = shelve.open(AOLIST_FILE, 'n')

    return addons


def addons_corrupted():
    with closing(open_addons(flag="r")) as addons:
        return len(addons) == 0

addon_refresh_callback = []

global index
index = defaultdict(list)
def rebuild_index():
    global index

    index = defaultdict(list)
    with closing(open_addons(flag="r")) as addons:
        for name, ao in addons.items():
            for s in [name, ao.summary, ao.description, ao.author] + (ao.keywords if ao.keywords else []):
                if not s:
                    continue
                words = [word for word in re.split(INDEX_RE, s.lower())
                         if len(word)>1]
                for word in words:
                    for i in range(len(word)):
                        index[word[:i+1]].append(name)


def search_index(query):
    global index
    result = set()
    words = [word for word in re.split(INDEX_RE, query.lower()) if len(word)>1]
    if not words:
        with closing(open_addons(flag="r")) as addons:
            return addons.keys()
    for word in words:
        result.update(index[word])
    return result


def refresh_available_addons(force=False, progress_callback=None):
    pypi = xmlrpclib.ServerProxy('https://pypi.python.org/pypi',
                                 transport=xmlrpclib.SafeTransport())
    if progress_callback:
        progress_callback(1, 0)

    pkg_dict = {}
    for data in pypi.search({'keywords': 'orange'}):
        name = data['name']
        order = data['_pypi_ordering']
        if name not in pkg_dict or pkg_dict[name][0] < order:
            pkg_dict[name] = (order, data['version'])

    try:
        import slumber
        readthedocs = slumber.API(base_url='http://readthedocs.org/api/v1/')
    except:
        readthedocs = None

    docs = {}
    if progress_callback:
        progress_callback(len(pkg_dict) + 1, 1)

    with closing(open_addons(flag="c")) as addons:
        for i, (name, (_, version)) in enumerate(pkg_dict.items()):
            installed = addons[name.lower()] if name.lower() in addons else None
            if force or not installed or installed.available_version != version:
                try:
                    data = pypi.release_data(name, version)
                    urls = pypi.release_urls(name, version)
                    release_urls = \
                        [ReleaseUrl(url["filename"], url["url"],
                                    url["size"], url["python_version"],
                                    url["packagetype"])
                         for url in urls]
                    if readthedocs:
                        try:
                            docs = readthedocs.project.get(
                                slug=name.lower())['objects'][0]
                        except:
                            docs = {}
                    addons[name.lower()] = OrangeAddOn(
                        name=name,
                        available_version=data['version'],
                        installed_version=installed.installed_version if installed else None,
                        summary=data['summary'],
                        description=data.get('description', ''),
                        author=str((data.get('author', '') or '') + ' ' +
                                   (data.get('author_email', '') or '')).strip(),
                        docs_url=data.get('docs_url', docs.get('subdomain', '')),
                        keywords=data.get('keywords', "").split(","),
                        homepage=data.get('home_page', ''),
                        package_url=data.get('package_url', ''),
                        release_urls=release_urls
                    )
                except Exception:
                    import traceback
                    traceback.print_exc()
                    warnings.warn(
                        'Could not load data for the add-on: %s' % name)

            if progress_callback:
                progress_callback(len(pkg_dict) + 1, i + 2)

    rebuild_index()


def load_installed_addons():
    found = set()
    with closing(open_addons(flag="c")) as addons:
        for entry_point in pkg_resources.iter_entry_points(ADDONS_ENTRY_POINT):
            name = entry_point.dist.project_name
            version = entry_point.dist.version

            if name.lower() in addons:
                addons[name.lower()] = addons[name.lower()]._replace(installed_version=version)
            else:
                addons[name.lower()] = OrangeAddOn(
                    name=name,
                    available_version=None,
                    installed_version=version,
                    summary="",
                    description="",
                    author="",
                    docs_url="",
                    keywords="",
                    homepage="",
                    package_url="",
                    release_urls=[])
            found.add(name.lower())

        for name in set(addons).difference(found):
            addons[name.lower()] = addons[name.lower()]._replace(installed_version=None)

    rebuild_index()


def open_archive(path, mode="r"):
    """
    Return an open archive file object (zipfile.ZipFile or tarfile.TarFile).
    """
    _, ext = os.path.splitext(path)
    if ext == ".zip":
        # TODO: should it also open .egg, ...
        archive = zipfile.ZipFile(path, mode)

    elif ext in (".tar", ".gz", ".bz2", ".tgz", ".tbz2", ".tb2"):
        archive = tarfile.open(path, mode)

    return archive


member_info = namedtuple(
    "member_info",
    ["info",  # original info object (Tar/ZipInfo)
     "path",  # filename inside the archive
     "linkname",  # linkname if applicable
     "issym",  # True if sym link
     "islnk",  # True if hardlink
     ]
)


def archive_members(archive):
    """
    Given an open archive return an iterator of `member_info` instances.
    """
    if isinstance(archive, zipfile.ZipFile):
        def converter(info):
            return member_info(info, info.filename, None, False, False)

        return itertools.imap(converter, archive.infolist())
    elif isinstance(archive, tarfile.TarFile):
        def converter(info):
            return member_info(info, info.name, info.linkname,
                               info.issym(), info.islnk())
        return itertools.imap(converter, archive.getmembers())
    else:
        raise TypeError


def resolve_path(path):
    """
    Return a normalized real path.
    """
    return os.path.normpath(os.path.realpath(os.path.abspath(path)))


def is_badfile(member, base_dir):
    """
    Would extracting `member_info` instance write outside of `base_dir`.
    """
    path = member.path
    full_path = resolve_path(os.path.join(base_dir, path))
    return not full_path.startswith(base_dir)


def is_badlink(member, base_dir):
    """
    Would extracting `member_info` instance create a link to outside
    of `base_dir`.

    """
    if member.issym or member.islnk:
        dirname = os.path.dirname(member.path)
        full_path = resolve_path(os.path.join(dirname, member.linkname))
        return not full_path.startswith(base_dir)
    else:
        return False


def check_safe(member, base_dir):
    """
    Check if member is safe to extract to base_dir or raise an exception.
    """
    path = member.path
    drive, path = os.path.splitdrive(path)

    if drive != "":
        raise ValueError("Absolute path in archive")

    if path.startswith("/"):
        raise ValueError("Absolute path in archive")

    base_dir = resolve_path(base_dir)

    if is_badfile(member, base_dir):
        raise ValueError("Extract outside %r" % base_dir)
    if is_badlink(member, base_dir):
        raise ValueError("Link outside %r" % base_dir)

    return True


def extract_archive(archive, path="."):
    """
    Extract the contents of `archive` to `path`.
    """
    if isinstance(archive, basestring):
        archive = open_archive(archive)

    members = archive_members(archive)

    for member in members:
        if check_safe(member, path):
            archive.extract(member.info, path)


def run_setup(setup_script, args):
    """
    Run `setup_script` with `args` in a subprocess, using
    :ref:`subprocess.check_output`.

    """
    source_root = os.path.dirname(setup_script)
    executable = sys.executable
    extra_kwargs = {}
    if os.name == "nt" and os.path.basename(executable) == "pythonw.exe":
        dirname, _ = os.path.split(executable)
        executable = os.path.join(dirname, "python.exe")
        # by default a new console window would show up when executing the
        # script
        startupinfo = subprocess.STARTUPINFO()
        if hasattr(subprocess, "STARTF_USESHOWWINDOW"):
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        else:
            # This flag was missing in inital releases of 2.7
            startupinfo.dwFlags |= subprocess._subprocess.STARTF_USESHOWWINDOW

        extra_kwargs["startupinfo"] = startupinfo

    process = subprocess.Popen([executable, setup_script] + args,
                               cwd=source_root,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               bufsize=1,  # line buffered
                               **extra_kwargs)
    output = []
    while process.poll() is None:
        try:
            line = process.stdout.readline()
        except (OSError, IOError) as ex:
            if ex.errno != errno.EINTR:
                raise
        else:
            output.append(line)
            print line,

    if process.returncode:
        raise subprocess.CalledProcessError(
                  process.returncode,
                  setup_script,
                  "".join(output)
              )


def install(name, progress_callback=None):
    with closing(open_addons(flag="r")) as addons:
        addon = addons[name.lower()]

    source_urls = [url for url in addon.release_urls
                   if url.packagetype == "sdist"]
    release_url = source_urls[0]

    try:
        tmpdir = tempfile.mkdtemp()

        stream = urllib2.urlopen(release_url.url, timeout=120)

        package_path = os.path.join(tmpdir, release_url.filename)

        progress_cb = (lambda value: progress_callback(value, 0)) \
                      if progress_callback else None
        with open(package_path, "wb") as package_file:
            Orange.utils.copyfileobj(
                stream, package_file, progress=progress_cb)

        extract_archive(package_path, tmpdir)

        setup_py = os.path.join(tmpdir, name + '-' + addon.available_version,
                                'setup.py')

        if not os.path.isfile(setup_py):
            raise Exception("Unable to install add-on - it is not properly "
                            "packed.")

        switches = []
        if not hasattr(sys, "real_prefix"):
            # we're not in a virtualenv
            switches.append('--user')
        run_setup(setup_py, ['install'] + switches)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    for p in list(sys.path):
        site.addsitedir(p)
    reload(pkg_resources)
    for p in list(sys.path):
        pkg_resources.find_distributions(p)
    from orngRegistry import load_new_addons
    load_new_addons()
    load_installed_addons()
    for func in addon_refresh_callback:
        func()


def easy_install_process(args, bufsize=-1):
    from setuptools.command import easy_install
    # Check if easy_install supports '--user' switch
    options = [opt[0] for opt in easy_install.easy_install.user_options]
    has_user_site = "user" in options

    if has_user_site and not hasattr(sys, "real_prefix"):
        # we're not in a virtualenv
        # (why are we assuming we have write permissions in the
        # virtualenv's site dir?)
        args = ["--user"] + args

    # properly quote arguments if necessary
    args = map(pipes.quote, args)

    script = """
import sys
from setuptools.command.easy_install import main
sys.exit(main({args!r}))
"""
    script = script.format(args=args)

    return python_process(["-c", script], bufsize=bufsize)


def python_process(args, script_name=None, cwd=None, env=None, **kwargs):
    """
    Run a `sys.executable` in a subprocess with `args`.
    """
    executable = sys.executable
    if os.name == "nt" and os.path.basename(executable) == "pythonw.exe":
        dirname, _ = os.path.split(executable)
        executable = os.path.join(dirname, "python.exe")
        # by default a new console window would show up when executing the
        # script
        startupinfo = subprocess.STARTUPINFO()
        if hasattr(subprocess, "STARTF_USESHOWWINDOW"):
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        else:
            # This flag was missing in inital releases of 2.7
            startupinfo.dwFlags |= subprocess._subprocess.STARTF_USESHOWWINDOW

        kwargs["startupinfo"] = startupinfo

    if script_name is not None:
        script = script_name
    else:
        script = executable

    process = subprocess.Popen(
        [script] + args,
        executable=executable,
        cwd=cwd,
        env=env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        **kwargs
    )

    return process


def uninstall(name, progress_callback=None):
    try:
        import pip.req
        ao = pip.req.InstallRequirement(name, None)
        ao.uninstall(True)
    except ImportError:
        raise Exception("Pip is required for add-on uninstallation. "
                        "Install pip and try again.")


def upgrade(name, progress_callback=None):
    install(name, progress_callback)


# Support for loading legacy "registered" add-ons
def __read_addons_list(addons_file, systemwide):
    if os.path.isfile(addons_file):
        return [tuple([x.strip() for x in lne.split("\t")])
                for lne in file(addons_file, "rt")]
    else:
        return []

registered = __read_addons_list(os.path.join(Orange.utils.environ.orange_settings_dir, "add-ons.txt"), False) + \
             __read_addons_list(os.path.join(Orange.utils.environ.install_dir, "add-ons.txt"), True)

if registered:
    warnings.warn("'add-ons.txt' is deprecated. " +
                  "Please use setuptools/entry points.",
                  UserWarning)

for name, path in registered:
    for p in [os.path.join(path, "widgets", "prototypes"),
          os.path.join(path, "widgets"),
          path,
          os.path.join(path, "lib-%s" % "-".join(( sys.platform, "x86" if (platform.machine()=="")
          else platform.machine(), ".".join(map(str, sys.version_info[:2])) )) )]:
        if os.path.isdir(p) and not any([Orange.utils.environ.samepath(p, x)
                                         for x in sys.path]):
            if p not in sys.path:
                sys.path.insert(0, p)

#TODO Show some progress to the user at least during the installation procedure.
