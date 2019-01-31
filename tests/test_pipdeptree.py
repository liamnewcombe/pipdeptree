import json
import os
import pickle
import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from operator import attrgetter

import pytest

from pipdeptree import (build_dist_index, construct_tree, filter_tree,
                        DistPackage, ReqPackage, render_tree, _mark_roots,
                        reverse_tree, cyclic_deps, conflicting_deps,
                        get_parser, render_json, render_json_tree,
                        dump_graphviz, dump_enhanced_graphviz, _get_dist_package,
                        build_package_tree,
                        _mark_include_exclude, _mark_versions, TreeWalker,
                        _label_conflicted_packages, _node_style, _edge_style,
                        MissingPackage, print_graphviz, main)


def venv_fixture(pickle_file):
    """Loads required virtualenv pkg data from a pickle file

    :param pickle_file: path to a .pickle file
    :returns: a tuple of pkgs, pkg_index, req_map
    :rtype: tuple

    """
    with open(pickle_file, 'rb') as f:
        pkgs = pickle.load(f)
        dist_index = build_dist_index(pkgs)
        tree = construct_tree(dist_index)
        return pkgs, dist_index, tree


pkgs, dist_index, tree = venv_fixture('tests/virtualenvs/testenv.pickle')


def find_dist(key):
    return dist_index[key]


def find_req(key, parent_key):
    parent = [x for x in tree.keys() if x.key == parent_key][0]
    return [x for x in tree[parent] if x.key == key][0]


def test_build_dist_index():
    assert len(dist_index) == len(pkgs)
    assert all(isinstance(x, str) for x in dist_index.keys())
    assert all(isinstance(x, DistPackage) for x in dist_index.values())


def test_tree():
    assert len(tree) == len(pkgs)
    assert all((isinstance(k, DistPackage) and
                all(isinstance(v, ReqPackage) for v in vs))
               for k, vs in tree.items())


def test_reverse_tree():
    rtree = reverse_tree(tree)
    assert all(isinstance(k, ReqPackage) for k, vs in rtree.items())
    assert all(all(isinstance(v, DistPackage) for v in vs)
               for k, vs in rtree.items())
    assert all(all(v.req is not None for v in vs)
               for k, vs in rtree.items())



########################################################################################


def test_missing_package():
    """Test constructing a MissingPackage from a ReqPackage."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    alembic = this_index['alembic']
    for req in this_tree[alembic]:
        if req.version_spec:
            missing = MissingPackage().from_req(req=req)
            break

    assert req.is_missing
    assert missing.key == req.key
    assert missing.project_name == req.project_name
    assert req.version_spec in missing.required_versions
    assert missing.version == 'MISSING'
    assert missing.include
    assert not missing.exclude


def test_get_dist_package():
    """Test get dist package finds a package as req.dist."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    alembic = this_index['alembic']
    req = this_tree[alembic][1]
    test_target = _get_dist_package(tree=this_tree, req=req, index=this_index)
    assert isinstance(test_target, DistPackage)
    assert test_target.key == req.key


def test_get_dist_package_by_index_key():
    """Test get dist package finds a package by key in the index."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    alembic = this_index['alembic']
    req = this_tree[alembic][1]
    req.dist = None
    test_target = _get_dist_package(tree=this_tree, req=req, index=this_index)
    assert isinstance(test_target, DistPackage)
    assert test_target.project_name == req.project_name
    # Assert that we repaired the tree
    assert req.dist == test_target


def test_get_dist_package_by_searching_key():
    """Test get dist package finds a package by key searching the tree."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    alembic = this_index['alembic']
    req = this_tree[alembic][1]
    req.dist = None
    test_target = _get_dist_package(tree=tree, req=req)
    assert isinstance(test_target, DistPackage)
    assert test_target.project_name == req.project_name
    # Assert that we repaired the tree
    assert req.dist == test_target


def test_get_dist_package_creates_missing():
    """Test get dist package creates a missing package and repairs tree."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/unsatisfiedenv.pickle')
    uritemplate = this_index['uritemplate']
    req = this_tree[uritemplate][0]
    test_target = _get_dist_package(tree=this_tree, req=req, index=this_index)
    assert isinstance(test_target, MissingPackage)
    # Assert that we repaired the tree
    assert req.dist == test_target
    assert test_target in this_tree
    assert test_target.key == 'simplejson'
    assert test_target.version == 'MISSING'


def test_build_package_tree():
    """Build a PackageTree and use get_nodes to test the node keys added."""
    _, _, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    test_package_tree = build_package_tree(tree=this_tree)
    test_keys = test_package_tree.get_nodes()
    target_keys = [
        'wheel', 'werkzeug', 'sqlalchemy', 'slugify', 'six', 'redis',
        'python-editor', 'python-dateutil', 'psycopg2', 'markupsafe',
        'mako', 'jinja2', 'itsdangerous', 'gnureadline', 'flask',
        'flask-script', 'click', 'alembic', 'lookupy']
    # Check it has all the packages, details later
    assert len(test_keys) == len(target_keys)
    for key in target_keys:
        assert key in test_keys


def test_package_tree_roots():
    """Test the identification of roots in the package tree."""
    _, _, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    test_package_tree = build_package_tree(tree=this_tree)
    test_roots = test_package_tree.roots()
    target_roots = ['alembic', 'wheel', 'flask-script', 'slugify',
                    'redis', 'psycopg2', 'gnureadline', 'lookupy']
    for root in target_roots:
        assert root in test_roots
    assert len(test_roots) == len(target_roots)


def test_package_tree_leaves():
    """Test the identification of leaves in the package tree."""
    _, _, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    test_package_tree = build_package_tree(tree=this_tree)
    test_leaves = test_package_tree.leaves()
    target_leaves = [
        'wheel', 'slugify', 'sqlalchemy', 'python-editor',
        'six', 'markupsafe', 'werkzeug', 'itsdangerous', 'click',
        'redis', 'psycopg2', 'gnureadline', 'lookupy']
    for leaf in target_leaves:
        assert leaf in test_leaves
    assert len(test_leaves) == len(target_leaves)


def test_package_tree_get_requirements():
    """Test the recursive requirement getter. Walk from Alembic and get the subtree."""
    _, _, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    test_package_tree = build_package_tree(tree=this_tree)
    test_node = test_package_tree.get_node('alembic')
    target_reqs = {
        "alembic":
            (None, "alembic", ['alembic']),
        "mako":
            ("mako", "mako", ['alembic', 'mako']),
        "markupsafe":
            ("markupsafe", "markupsafe", ['alembic', 'mako', 'markupsafe']),
        "sqlalchemy":
            ("sqlalchemy", "sqlalchemy", ['alembic', 'sqlalchemy']),
        "python-dateutil":
            ("python-dateutil", "python-dateutil", ['alembic', 'python-dateutil']),
        "six":
            ("six", "six", ['alembic', 'python-dateutil', 'six']),
        "python-editor":
            ("python-editor", "python-editor", ['alembic', 'python-editor'])}
    listy = list(test_node.get_requirements())
    assert len(listy) == len(target_reqs)
    for req, dist, stack in test_node.get_requirements():
        target_req, target_dist, target_stack = target_reqs[dist.key]
        if target_req:
            assert target_req == req.key
        assert dist.key == target_dist
        assert stack == target_stack


def test_package_tree_delete():
    """Test deleting a node from the PackageTree."""
    _, _, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    test_package_tree = build_package_tree(tree=this_tree)
    test_package_tree.delete(key='mako')
    # Test that mako is gone
    with pytest.raises(KeyError) as pytest_wrapped_e:
        node = test_package_tree.get_node(key='mako')
    assert pytest_wrapped_e.type == KeyError
    # Test the dependency edge from alembic is gone
    alembic = test_package_tree.get_node(key='alembic')
    assert len(alembic.requirements) == 3
    for req in alembic.requirements:
        assert req.key != 'mako'
    # Test that markupsafe knows it's not required by mako any more
    markupsafe = test_package_tree.get_node(key='markupsafe')
    assert len(markupsafe.required_by) == 1
    # but markupsafe knows it's still required by jinja2
    jinja2 = test_package_tree.get_node(key='jinja2')
    assert jinja2 in markupsafe.required_by

def test_tree_walker():
    """Test the tree walker generator walks the branches to leaves."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    alembic = this_index['alembic']
    walker = TreeWalker(tree=this_tree)
    traversed_keys = set()
    for req, dist, stack in walker.walk(package=alembic):
        assert isinstance(req, ReqPackage)
        assert isinstance(dist, DistPackage)
        assert dist.key == req.key
        traversed_keys.add(dist.key)
    target_keys = {'python-dateutil', 'markupsafe', 'sqlalchemy',
                   'six', 'mako', 'python-editor'}
    assert traversed_keys == target_keys


def test_tree_walker_no_index():
    """Test the tree walker generator walks the branches to leaves.
    Assert it works the same without the key index."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    alembic = this_index['alembic']
    walker = TreeWalker(tree=this_tree)
    for dep, pkg, _ in walker.walk(package=alembic):
        assert isinstance(dep, ReqPackage)
        assert isinstance(pkg, DistPackage)
    pkgs = list(walker.walk(package=alembic))
    assert len(pkgs) == 6


def test_tree_walker_visit_once():
    """Test the tree walker generator walks the branches to leaves.
    Assert it skips nodes already visited."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    alembic = this_index['alembic']
    flask = this_index['flask']
    walker = TreeWalker(tree=this_tree, visit_once=True)
    pkgs = []
    for req, dist, stack in walker.walk(package=alembic):
        pkgs.append(dist)
        assert isinstance(req, ReqPackage)
        assert isinstance(dist, DistPackage)
    assert len(pkgs) == 6
    # Now walk the other branch from Flask, should not include MarkupSafe again:
    pkgs = list(walker.walk(package=flask))
    assert len(pkgs) == 4
    # Walk the Flask branch again, forgetting what we've visited
    walker.forget_visited()
    pkgs = list(walker.walk(package=flask))
    assert len(pkgs) == 5


def test_tree_walker_circular():
    """Test the tree walker stops walking for circular
    dependencies in the tree."""
    cyclic_pkgs, cyclic_index, cyclic_tree = venv_fixture(
            'tests/virtualenvs/cyclicenv.pickle')
    cyclic_pkg = cyclic_index['circulardependencya']
    keys = []
    walker = TreeWalker(tree=cyclic_tree)
    for req, dist, stack in walker.walk(package=cyclic_pkg):
        keys.append(dist.key)
    assert len(keys) == 2


def test_tree_walker_reverse():
    """Test the tree walker reverses the tree and walks leaf to root"""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    walker = TreeWalker(tree=this_tree, reverse=True)
    markupsafe = this_index['markupsafe']
    for req, dist, stack in walker.walk(package=markupsafe):
        dist.include = True
    # Assert we hit the nodes up to root(s)
    targets = ['flask-script', 'flask', 'jinja2', 'mako', 'alembic']
    for name in targets:
        test_pkg = this_index[name]
        assert test_pkg.include
    # Assert we missed all the others
    for package in tree.keys():
        if package.key not in targets:
            assert not package.include


def test_label_conflicted_packages():
    """Test the reverse walker labelling conflicted_reqs"""
    _, conflicting_index, conflicting_tree = venv_fixture(
            'tests/virtualenvs/unsatisfiedenv.pickle')
    # Set up the tree
    _mark_include_exclude(tree=conflicting_tree)
    _mark_versions(tree=conflicting_tree, index=conflicting_index)
    # Label from one package
    jinja2 = conflicting_index['jinja2']
    _label_conflicted_packages(
            tree=conflicting_tree, package=jinja2)
    flask = conflicting_index['flask']
    assert flask.conflicted_reqs == {'Jinja2'}
    # Now walk the whole tree
    for pkg in conflicting_tree:
        _label_conflicted_packages(
                tree=conflicting_tree, package=pkg)
    # Assert that only two packages are labelled, flask and uritemplate
    for pkg in conflicting_tree:
        if not pkg.key in ['flask', 'uritemplate']:
            assert pkg.conflicted_reqs == set()
        if pkg.key in ['flask', 'uritemplate']:
            assert len(pkg.conflicted_reqs) == 1


def test_package_filter():
    """Test that the show_only={} works on the enhanced tree."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    tree_str = render_tree(this_tree, show_only={'alembic'})
    lines = set(tree_str.split('\n'))
    target_set = {'  - SQLAlchemy [required: >=0.7.6, installed: 1.2.9]',
                  '  - Mako [required: Any, installed: 1.0.7]',
                  '  - python-editor [required: >=0.3, installed: 1.0.3]',
                  'alembic==0.9.10',
                  '  - python-dateutil [required: Any, installed: 2.7.3]',
                  '    - MarkupSafe [required: >=0.9.2, installed: 1.0]',
                  '    - six [required: >=1.5, installed: 1.11.0]'}
    assert target_set == lines


def test_mark_versions():
    """Test the version labelling tree traversal."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    _mark_versions(tree=this_tree, index=this_index)
    # Should have marked all packages with the versions
    # their parents require, MarkupSafe should have two as it's conflicted
    pkg = this_index['python-dateutil']
    assert pkg.required_versions == set()
    pkg = this_index['sqlalchemy']
    assert pkg.required_versions == {'>=0.7.6'}
    pkg = this_index['itsdangerous']
    assert pkg.required_versions == {'>=0.24'}
    pkg = this_index['markupsafe']
    assert pkg.required_versions == {'>=0.23', '>=0.9.2'}


def test_mark_roots():
    """Test that mark roots marks all non-roots as is_root=False."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    _mark_roots(tree=this_tree)
    # Assert that these packages are roots
    for name in ['flask-script', 'alembic', 'gnureadline', 'lookupy',
                 'psycopg2', 'redis', 'slugify', 'wheel']:
        pkg = this_index[name]
        assert pkg.is_root
    # Assert that these packages are not roots
    for name in ['click', 'flask', 'itsdangerous', 'jinja2',
                 'mako', 'markupsafe', 'python-dateutil', 'python-editor',
                 'six', 'sqlalchemy', 'werkzeug']:
        pkg = this_index[name]
        assert not pkg.is_root


def test_mark_include_exclude():
    """Test walking the tree and applying the include / exclude lists."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    _mark_include_exclude(tree=this_tree,
                          show_only={'alembic', 'flask'},
                          exclude={'mako'})
    # Excludes
    mako = this_index['mako']
    assert mako.exclude
    # Assert we didn't exclude the already included leaf 'markupsafe'
    # Which is excluded from alembic / mako but included from flask / Jinja2
    for name in ['alembic', 'click', 'flask', 'itsdangerous', 'jinja2',
                 'markupsafe', 'python-dateutil', 'python-editor',
                 'six', 'sqlalchemy', 'werkzeug']:
        dist = this_index[name]
        assert dist.include
    for name in ['gnureadline', 'lookupy', 'mako',
                 'psycopg2', 'redis', 'slugify', 'wheel']:
        dist = this_index[name]
        assert not dist.include


def test_node_style_basic():
    """Assert the node styling for each param. - Regular package"""
    _ , this_index, _ = venv_fixture('tests/virtualenvs/testenv.pickle')
    pkg_dateutil = this_index['python-dateutil']
    pkg_dateutil.required_versions = {'>=1.0.0', '==1.5.0'}
    test_style = _node_style(pkg_dateutil)
    assert test_style['label'] == 'python-dateutil\nVer: 2.7.3\nReq: ==1.5.0, >=1.0.0' or \
           test_style['label'] == 'python-dateutil\nVer: 2.7.3\nReq: >=1.0.0, ==1.5.0'
    assert test_style['project_name'] == 'python-dateutil'
    assert test_style['version'] == '2.7.3'
    assert test_style['required_versions'] == '>=1.0.0, ==1.5.0' or '==1.5.0, >=1.0.0'


def test_node_style_conflicted_reqs():
    """Assert the node styling for conflicted reqs"""
    _, this_index, _ = venv_fixture('tests/virtualenvs/testenv.pickle')
    pkg_dateutil = this_index['python-dateutil']
    pkg_dateutil.required_versions = {'>=1.0.0', '==1.5.0'}
    pkg_dateutil.conflicted_reqs.add('some-other-package')
    test_style = _node_style(pkg_dateutil)
    assert test_style['label'] == \
           'python-dateutil\nVer: 2.7.3\nReq: ==1.5.0, >=1.0.0\nConflicted: some-other-package' or \
           test_style['label'] == \
           'python-dateutil\nVer: 2.7.3\nReq: >=1.0.0, ==1.5.0\nConflicted: some-other-package'
    assert test_style['project_name'] == 'python-dateutil'
    assert test_style['version'] == '2.7.3'
    assert test_style['required_versions'] == '>=1.0.0, ==1.5.0' or '==1.5.0, >=1.0.0'
    assert test_style['color'] == 'khaki'
    assert test_style['style'] == 'filled'
    assert test_style['conflicted_reqs'] == 'some-other-package'


def test_node_style_conflict_target():
    """Assert the node styling for conflict target"""
    _, this_index, _ = venv_fixture('tests/virtualenvs/testenv.pickle')
    pkg_dateutil = this_index['python-dateutil']
    pkg_dateutil.required_versions = {'>=1.0.0', '==1.5.0'}
    pkg_dateutil.conflicted_reqs.add('some-other-package')
    pkg_dateutil.conflict_target = True
    test_style = _node_style(pkg_dateutil)
    assert test_style['label'] == \
           'python-dateutil\nVer: 2.7.3\nReq: ==1.5.0, >=1.0.0\nConflicted: some-other-package' or \
           test_style['label'] == \
           'python-dateutil\nVer: 2.7.3\nReq: >=1.0.0, ==1.5.0\nConflicted: some-other-package'
    assert test_style['project_name'] == 'python-dateutil'
    assert test_style['version'] == '2.7.3'
    assert test_style['required_versions'] == '>=1.0.0, ==1.5.0' or '==1.5.0, >=1.0.0'
    # Assert we overrode the conficted reqs color
    assert test_style['color'] == 'orange'
    assert test_style['style'] == 'filled'


def test_node_style_missing_package():
    """Assert the node styling for Missing package"""
    _, this_index, _ = venv_fixture('tests/virtualenvs/testenv.pickle')
    pkg_dateutil = this_index['python-dateutil']
    pkg_dateutil.version = 'MISSING'
    pkg_dateutil.required_versions = {'>=1.0.0', '==1.5.0'}
    pkg_dateutil.conflict_target = True
    test_style = _node_style(pkg_dateutil)
    assert test_style['label'] == 'python-dateutil\nVer: MISSING\nReq: ==1.5.0, >=1.0.0' or \
           test_style['label'] == 'python-dateutil\nVer: MISSING\nReq: >=1.0.0, ==1.5.0'
    assert test_style['project_name'] == 'python-dateutil'
    assert test_style['version'] == 'MISSING'
    assert test_style['required_versions'] == '>=1.0.0, ==1.5.0' or '==1.5.0, >=1.0.0'
    assert test_style['missing'] == 'True'
    # Assert we overrode the conflicted_reqs and conflict_target colors
    assert test_style['color'] == 'red'
    assert test_style['style'] == 'filled'
    assert test_style['fontcolor'] == 'white'


def test_edge_style_base():
    """Assert the base edge style."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    pkg = this_index['flask']
    deps = this_tree[pkg]
    for dep in deps:
        test_style = _edge_style(dep)
        assert test_style == {}


def test_edge_style_conflict():
    """Assert the conflict edge style."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/unsatisfiedenv.pickle')
    pkg = this_index['flask']
    deps = this_tree[pkg]
    dep = [dep for dep in deps if dep.key == 'jinja2'][0]
    assert dep.key == 'jinja2'
    test_style = _edge_style(dep)
    assert test_style == {'color': 'orange', 'fontcolor': 'orange', 'conflict': 'True'}


def test_edge_style_missing():
    """Assert the conflict edge style."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/unsatisfiedenv.pickle')
    filter_tree(tree=this_tree)
    pkg = this_index['uritemplate']
    deps = this_tree[pkg]
    dep = [dep for dep in deps if dep.key == 'simplejson'][0]
    assert dep.key == 'simplejson'
    test_style = _edge_style(dep)
    assert test_style == {'color': 'red', 'fontcolor': 'red',
                          'conflict': 'True', 'missing': 'True'}


# def test_graph():
#     output = dump_graphviz(tree, output_format='png')
#     with open('graph.png', 'wb') as f:
#         f.write(output)


# def test_filtered_graph():
#     output = dump_enhanced_graphviz(tree,
#                                     show_only={'alembic', 'flask'},
#                                     exclude={'mako', 'python-dateutil'},
#                                     output_format='png')
#     with open('filtered_graph.png', 'wb') as f:
#         f.write(output)


def test_filtered_tree():
    """Do a full enhanced-graph filter and test the resulting tree nodes."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    filtered_tree = filter_tree(
            tree=this_tree,
            show_only={'alembic', 'flask'},
            exclude={'mako', 'python-dateutil'})
    # Assert that these packages were not included
    for name in ['mako', 'gnureadline', 'lookupy',
                 'psycopg2', 'redis', 'slugify', 'wheel']:
        pkg = this_index[name]
        assert pkg not in filtered_tree
    # Assert that these packages were excluded
    for name in ['mako', 'python-dateutil', 'six']:
        pkg = this_index[name]
        assert pkg not in filtered_tree
    # Assert that these packages are included
    for name in ['alembic', 'click', 'flask', 'itsdangerous', 'jinja2',
                 'python-editor', 'markupsafe', 'sqlalchemy', 'werkzeug']:
        pkg = this_index[name]
        assert pkg in filtered_tree


def test_filtered_tree_package_state():
    """Do a full enhanced-graph filter and test no nodes are conflicted."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    filtered_tree = filter_tree(
            tree=this_tree,
            show_only={'alembic', 'flask'},
            exclude={'mako'})
    for node in filtered_tree.keys():
        assert not node.conflict_target
        assert node.conflicted_reqs == set()


def test_filtered_tree_reqs():
    """Do a full enhanced-graph filter and test that no nodes are is_missing."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    filtered_tree = filter_tree(
            tree=this_tree,
            show_only={'alembic', 'flask'},
            exclude={'mako'})
    for node in filtered_tree.keys():
        assert node.version != "MISSING"
        assert not node.conflict_target
        for req in filtered_tree[node]:
            assert not req.is_missing
            assert not req.is_conflicting()


def test_filtered_tree_no_includes():
    """Test a filtered tree with only excludes, no active includes."""
    _, this_index, this_tree = venv_fixture('tests/virtualenvs/testenv.pickle')
    filtered_tree = filter_tree(
            tree=this_tree,
            exclude={'mako', 'python-dateutil'})
    # Assert that these packages are included
    for name in ['alembic', 'click', 'flask', 'itsdangerous', 'jinja2',
                 'python-editor', 'markupsafe', 'sqlalchemy', 'werkzeug',
                 'gnureadline', 'lookupy', 'psycopg2', 'redis', 'slugify',
                 'wheel']:
        pkg = this_index[name]
        assert pkg in filtered_tree
    # Assert that these packages were excluded
    for name in ['mako', 'python-dateutil', 'six']:
        pkg = this_index[name]
        assert pkg not in filtered_tree


# def test_conflicting_graph_colour():
#     """Test the conflicts get coloured in."""
#     _, _, conflicting_tree = venv_fixture('tests/virtualenvs/unsatisfiedenv.pickle')
#     output = dump_enhanced_graphviz(tree=conflicting_tree, output_format='png')
#     with open('conflicting_graph.png', 'wb') as f:
#         f.write(output)


# def test_romopackages_graph():
#     pkgs, dist_index, romo_tree = venv_fixture('tests/virtualenvs/romopackages.pickle')
#     output = dump_enhanced_graphviz(tree=romo_tree,
#                                     # show_only={'api-importaler'},
#                                     exclude={'graphviz', 'pytest', 'wheel'},
#                                     output_format='png')
#     with open('romopackages.png', 'wb') as f:
#         f.write(output)


def test_main_enhanced_graph(monkeypatch, capsys):
    """Test the enhanced graph from --enhanced-graph to dot output"""
    parser = get_parser()
    args = parser.parse_args('--graph-output dot --enhanced-graph'.split())

    def _get_args():
        return args
    monkeypatch.setattr('pipdeptree._get_args', _get_args)
    con_pkgs, con_index, con_tree = venv_fixture('tests/virtualenvs/unsatisfiedenv.pickle')
    def get_installed_distributions(local_only=args.local_only, user_only=args.user_only):
        return con_pkgs
    monkeypatch.setattr('pipdeptree.get_installed_distributions', get_installed_distributions)

    assert main() == 0
    out, _ = capsys.readouterr()
    # Assert a simple package with no issues and no dependencies
    argparse = ("""argparse [label="argparse\nVer: 1.4.0" """
                """project_name=argparse required_versions="" version="1.4.0"]""")
    assert argparse in out
    # Assert simpleJSON which is Missing in this venv
    simplejson = ("""simplejson [label="simplejson\nVer: MISSING\nReq: >=2.5.0" """
                  """color=red conflict_target=True fontcolor=white """
                  """missing=True project_name=simplejson required_versions=">=2.5.0" """
                  """style=filled version=MISSING]""")
    assert simplejson in out
    # Assert Flask has conflicted dependencies
    flask = ("""flask [label="Flask\nVer: 0.10.1\nConflicted: Jinja2" """
             """color=khaki conflicted_reqs=Jinja2 project_name=Flask """
             """required_versions="" style=filled version="0.10.1"]\n"""
             """	flask -> itsdangerous [label=">=0.21"]\n"""
             """	flask -> jinja2 [label=">=2.4" color=orange conflict=True fontcolor=orange]\n"""
             """	flask -> werkzeug [label=">=0.7"]""")
    assert flask in out
    conflicted = ("""jinja2 [label="Jinja2\nVer: 2.3\nReq: >=2.4" """
                  """color=orange conflict_target=True project_name=Jinja2 """
                  """required_versions=">=2.4" style=filled version=2.3]""")
    assert conflicted in out


########################################################################################



def test_DistPackage_render_as_root():
    alembic = find_dist('alembic')
    assert alembic.version == '0.9.10'
    assert alembic.project_name == 'alembic'
    assert alembic.render_as_root(frozen=False) == 'alembic==0.9.10'


def test_DistPackage_render_as_branch():
    sqlalchemy = find_req('sqlalchemy', 'alembic')
    alembic = find_dist('alembic').as_required_by(sqlalchemy)
    assert alembic.project_name == 'alembic'
    assert alembic.version == '0.9.10'
    assert sqlalchemy.project_name == 'SQLAlchemy'
    assert sqlalchemy.version_spec == '>=0.7.6'
    assert sqlalchemy.installed_version == '1.2.9'
    result_1 = alembic.render_as_branch(False)
    result_2 = alembic.render_as_branch(False)
    assert result_1 == result_2 == 'alembic==0.9.10 [requires: SQLAlchemy>=0.7.6]'


def test_ReqPackage_render_as_root():
    flask = find_req('flask', 'flask-script')
    assert flask.project_name == 'Flask'
    assert flask.installed_version == '1.0.2'
    assert flask.render_as_root(frozen=False) == 'Flask==1.0.2'


def test_ReqPackage_render_as_branch():
    mks1 = find_req('markupsafe', 'jinja2')
    assert mks1.project_name == 'MarkupSafe'
    assert mks1.installed_version == '1.0'
    assert mks1.version_spec == '>=0.23'
    assert mks1.render_as_branch(False) == 'MarkupSafe [required: >=0.23, installed: 1.0]'
    assert mks1.render_as_branch(True) == 'MarkupSafe==1.0'
    mks2 = find_req('markupsafe', 'mako')
    assert mks2.project_name == 'MarkupSafe'
    assert mks2.installed_version == '1.0'
    assert mks2.version_spec == '>=0.9.2'
    assert mks2.render_as_branch(False) == 'MarkupSafe [required: >=0.9.2, installed: 1.0]'
    assert mks2.render_as_branch(True) == 'MarkupSafe==1.0'


def test_render_tree_only_top():
    tree_str = render_tree(tree, list_all=False)
    lines = set(tree_str.split('\n'))
    assert 'Flask-Script==2.0.6' in lines
    assert '  - SQLAlchemy [required: >=0.7.6, installed: 1.2.9]' in lines
    assert 'Lookupy==0.1' in lines
    assert 'itsdangerous==0.24' not in lines


def test_render_tree_list_all():
    tree_str = render_tree(tree, list_all=True)
    lines = set(tree_str.split('\n'))
    assert 'Flask-Script==2.0.6' in lines
    assert '  - SQLAlchemy [required: >=0.7.6, installed: 1.2.9]' in lines
    assert 'Lookupy==0.1' in lines
    assert 'itsdangerous==0.24' in lines


def test_render_tree_exclude():
    tree_str = render_tree(tree, list_all=True, exclude={'itsdangerous', 'SQLAlchemy', 'Flask', 'markupsafe', 'wheel'})
    expected = """alembic==0.9.10
  - Mako [required: Any, installed: 1.0.7]
  - python-dateutil [required: Any, installed: 2.7.3]
    - six [required: >=1.5, installed: 1.11.0]
  - python-editor [required: >=0.3, installed: 1.0.3]
click==6.7
Flask-Script==2.0.6
gnureadline==6.3.8
Jinja2==2.10
Lookupy==0.1
Mako==1.0.7
psycopg2==2.7.5
python-dateutil==2.7.3
  - six [required: >=1.5, installed: 1.11.0]
python-editor==1.0.3
redis==2.10.6
six==1.11.0
slugify==0.0.1
Werkzeug==0.14.1"""
    assert expected == tree_str


def test_render_tree_exclude_reverse():
    rtree = reverse_tree(tree)
    tree_str = render_tree(rtree, list_all=True, exclude={'itsdangerous', 'SQLAlchemy', 'Flask', 'markupsafe', 'wheel'})
    expected = """alembic==0.9.10
click==6.7
Flask-Script==2.0.6
gnureadline==6.3.8
Jinja2==2.10
Lookupy==0.1
Mako==1.0.7
  - alembic==0.9.10 [requires: Mako]
psycopg2==2.7.5
python-dateutil==2.7.3
  - alembic==0.9.10 [requires: python-dateutil]
python-editor==1.0.3
  - alembic==0.9.10 [requires: python-editor>=0.3]
redis==2.10.6
six==1.11.0
  - python-dateutil==2.7.3 [requires: six>=1.5]
    - alembic==0.9.10 [requires: python-dateutil]
slugify==0.0.1
Werkzeug==0.14.1"""
    assert expected == tree_str


def test_render_tree_freeze():
    tree_str = render_tree(tree, list_all=False, frozen=True)
    lines = set()
    for line in tree_str.split('\n'):
        # Workaround for https://github.com/pypa/pip/issues/1867
        # When hash randomization is enabled, pip can return different names
        # for git editables from run to run
        line = line.replace('origin/master', 'master')
        line = line.replace('origin/HEAD', 'master')
        lines.add(line)
    assert 'Flask-Script==2.0.6' in lines
    assert '  SQLAlchemy==1.2.9' in lines
    # TODO! Fix the following failing test
    # assert '-e git+https://github.com/naiquevin/lookupy.git@cdbe30c160e1c29802df75e145ea4ad903c05386#egg=Lookupy-master' in lines
    assert 'itsdangerous==0.24' not in lines


def test_render_json(capsys):
    output = render_json(tree, indent=4)
    print_graphviz(output)
    out, _ = capsys.readouterr()
    assert out.startswith('[\n    {\n        "')
    assert out.strip().endswith('}\n]')
    data = json.loads(out)
    assert 'package' in data[0]
    assert 'dependencies' in data[0]


def test_render_json_tree():
    output = render_json_tree(tree, indent=4)
    data = json.loads(output)

    # @TODO: This test fails on travis because gnureadline doesn't
    # appear as a dependency of ipython (which it is)
    #
    # ignored_pkgs = {'pip', 'pipdeptree', 'setuptools', 'wheel'}
    # pkg_keys = set([d['key'].lower() for d in data
    #                 if d['key'].lower() not in ignored_pkgs])
    # expected = {'alembic', 'flask-script', 'ipython',
    #             'lookupy', 'psycopg2', 'redis', 'slugify'}
    # assert pkg_keys - expected == set()

    matching_pkgs = [p for p in data if p['key'] == 'flask-script']
    assert matching_pkgs
    flask_script = matching_pkgs[0]

    matching_pkgs = [p for p in flask_script['dependencies'] if p['key'] == 'flask']
    assert matching_pkgs
    flask = matching_pkgs[0]

    matching_pkgs = [p for p in flask['dependencies'] if p['key'] == 'jinja2']
    assert matching_pkgs
    jinja2 = matching_pkgs[0]

    assert [p for p in jinja2['dependencies'] if p['key'] == 'markupsafe']


def test_render_pdf():
    output = dump_graphviz(tree, output_format='pdf')

    @contextmanager
    def redirect_stdout(new_target):
        old_target, sys.stdout = sys.stdout, new_target
        try:
            yield new_target
        finally:
            sys.stdout = old_target

    f = NamedTemporaryFile(delete=False)
    with redirect_stdout(f):
        print_graphviz(output)
    with open(f.name, 'rb') as rf:
        out = rf.read()
    os.remove(f.name)
    assert out[:4] == b'%PDF'


def test_render_svg(capsys):
    output = dump_graphviz(tree, output_format='svg')
    print_graphviz(output)
    out, _ = capsys.readouterr()
    assert out.startswith('<?xml')
    assert '<svg' in out
    assert out.strip().endswith('</svg>')


def test_parser_default():
    parser = get_parser()
    args = parser.parse_args([])
    assert not args.json
    assert args.output_format is None


def test_parser_j():
    parser = get_parser()
    args = parser.parse_args(['-j'])
    assert args.json
    assert args.output_format is None


def test_parser_json():
    parser = get_parser()
    args = parser.parse_args(['--json'])
    assert args.json
    assert args.output_format is None


def test_parser_json_tree():
    parser = get_parser()
    args = parser.parse_args(['--json-tree'])
    assert args.json_tree
    assert not args.json
    assert args.output_format is None


def test_parser_pdf():
    parser = get_parser()
    args = parser.parse_args(['--graph-output', 'pdf'])
    assert args.output_format == 'pdf'
    assert not args.json


def test_parser_svg():
    parser = get_parser()
    args = parser.parse_args(['--graph-output', 'svg'])
    assert args.output_format == 'svg'
    assert not args.json


def test_cyclic_dependencies():
    cyclic_pkgs, dist_index, tree = venv_fixture('tests/virtualenvs/cyclicenv.pickle')
    cyclic = [map(attrgetter('key'), cs) for cs in cyclic_deps(tree)]
    assert len(cyclic) == 2
    a, b, c = cyclic[0]
    x, y, z = cyclic[1]
    assert a == c == y
    assert x == z == b


def test_render_tree_cyclic_dependency():
    cyclic_pkgs, dist_index, tree = venv_fixture('tests/virtualenvs/cyclicenv.pickle')
    tree_str = render_tree(tree, list_all=True)
    lines = set(tree_str.split('\n'))
    assert 'CircularDependencyA==0.0.0' in lines
    assert '  - CircularDependencyB [required: Any, installed: 0.0.0]' in lines
    assert 'CircularDependencyB==0.0.0' in lines
    assert '  - CircularDependencyA [required: Any, installed: 0.0.0]' in lines


def test_render_tree_freeze_cyclic_dependency():
    cyclic_pkgs, dist_index, tree = venv_fixture('tests/virtualenvs/cyclicenv.pickle')
    tree_str = render_tree(tree, list_all=True, frozen=True)
    lines = set(tree_str.split('\n'))
    assert 'CircularDependencyA==0.0.0' in lines
    assert '  CircularDependencyB==0.0.0' in lines
    assert 'CircularDependencyB==0.0.0' in lines
    assert '  CircularDependencyA==0.0.0' in lines


def test_conflicting_deps():
    # the custom environment has a bad jinja version and it's missing simplejson
    _, _, conflicting_tree = venv_fixture('tests/virtualenvs/unsatisfiedenv.pickle')
    flask = next((x for x in conflicting_tree.keys() if x.key == 'flask'))
    jinja = next((x for x in conflicting_tree[flask] if x.key == 'jinja2'))
    uritemplate = next((x for x in conflicting_tree.keys() if x.key == 'uritemplate'))
    simplejson = next((x for x in conflicting_tree[uritemplate] if x.key == 'simplejson'))
    assert jinja
    assert flask
    assert uritemplate
    assert simplejson

    unsatisfied = conflicting_deps(conflicting_tree)
    assert unsatisfied == {
        flask: [jinja],
        uritemplate: [simplejson],
    }


def test_main_basic(monkeypatch):
    parser = get_parser()
    args = parser.parse_args('')

    def _get_args():
        return args
    monkeypatch.setattr('pipdeptree._get_args', _get_args)

    assert main() == 0


def test_main_show_only_and_exclude_ok(monkeypatch):
    parser = get_parser()
    args = parser.parse_args('--packages Flask --exclude Jinja2'.split())

    def _get_args():
        return args
    monkeypatch.setattr('pipdeptree._get_args', _get_args)

    assert main() == 0


def test_main_show_only_and_exclude_fails(monkeypatch):
    parser = get_parser()
    args = parser.parse_args('--packages Flask --exclude Jinja2,Flask'.split())

    def _get_args():
        return args
    monkeypatch.setattr('pipdeptree._get_args', _get_args)

    with pytest.raises(SystemExit):
        main()
