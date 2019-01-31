from __future__ import print_function
import os
import sys
from itertools import chain
from collections import defaultdict
import argparse
from operator import attrgetter
import json
from importlib import import_module

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

try:
    from pip._internal.utils.misc import get_installed_distributions
    from pip._internal.operations.freeze import FrozenRequirement
except ImportError:
    from pip import get_installed_distributions, FrozenRequirement

import pkg_resources
# inline:
# from graphviz import backend, Digraph


__version__ = '0.13.2'


flatten = chain.from_iterable


def build_dist_index(pkgs):
    """Build an index pkgs by their key as a dict.

    :param list pkgs: list of pkg_resources.Distribution instances
    :returns: index of the pkgs by the pkg key
    :rtype: dict

    """
    return dict((p.key, DistPackage(p)) for p in pkgs)


def construct_tree(index):
    """Construct tree representation of the pkgs from the index.

    The keys of the dict representing the tree will be objects of type
    DistPackage and the values will be list of ReqPackage objects.

    :param dict index: dist index ie. index of pkgs by their keys
    :returns: tree of pkgs and their dependencies
    :rtype: dict

    """
    return dict((p, [ReqPackage(r, index.get(r.key))
                     for r in p.requires()])
                for p in index.values())


def sorted_tree(tree):
    """Sorts the dict representation of the tree

    The root packages as well as the intermediate packages are sorted
    in the alphabetical order of the package names.

    :param dict tree: the pkg dependency tree obtained by calling
                     `construct_tree` function
    :returns: sorted tree
    :rtype: collections.OrderedDict

    """
    return OrderedDict(sorted([(k, sorted(v, key=attrgetter('key')))
                               for k, v in tree.items()],
                              key=lambda kv: kv[0].key))


def find_tree_root(tree, key):
    """Find a root in a tree by it's key

    :param dict tree: the pkg dependency tree obtained by calling
                     `construct_tree` function
    :param str key: key of the root node to find
    :returns: a root node if found else None
    :rtype: mixed

    """
    result = [p for p in tree.keys() if p.key == key]
    assert len(result) in [0, 1]
    return None if len(result) == 0 else result[0]


def reverse_tree(tree):
    """Reverse the dependency tree.

    ie. the keys of the resulting dict are objects of type
    ReqPackage and the values are lists of DistPackage objects.

    :param dict tree: the pkg dependency tree obtained by calling
                      `construct_tree` function
    :returns: reversed tree
    :rtype: dict

    """
    rtree = defaultdict(list)
    child_keys = set(c.key for c in flatten(tree.values()))
    for k, vs in tree.items():
        for v in vs:
            node = find_tree_root(rtree, v.key) or v
            rtree[node].append(k.as_required_by(v))
        if k.key not in child_keys:
            rtree[k.as_requirement()] = []
    return rtree


def guess_version(pkg_key, default='?'):
    """Guess the version of a pkg when pip doesn't provide it

    :param str pkg_key: key of the package
    :param str default: default version to return if unable to find
    :returns: version
    :rtype: string

    """
    try:
        m = import_module(pkg_key)
    except ImportError:
        return default
    else:
        return getattr(m, '__version__', default)


def frozen_req_from_dist(dist):
    try:
        return FrozenRequirement.from_dist(dist)
    except TypeError:
        return FrozenRequirement.from_dist(dist, [])


class Package(object):
    """Abstract class for wrappers around objects that pip returns.

    This class needs to be subclassed with implementations for
    `render_as_root` and `render_as_branch` methods.

    """

    def __init__(self, obj):
        self._obj = obj
        self.project_name = obj.project_name
        self.key = obj.key

    def render_as_root(self, frozen):
        return NotImplementedError

    def render_as_branch(self, frozen):
        return NotImplementedError

    def render(self, parent=None, frozen=False):
        if not parent:
            return self.render_as_root(frozen)
        else:
            return self.render_as_branch(frozen)

    @staticmethod
    def frozen_repr(obj):
        fr = frozen_req_from_dist(obj)
        return str(fr).strip()

    def __getattr__(self, key):
        return getattr(self._obj, key)

    def __repr__(self):
        return '<{0}("{1}")>'.format(self.__class__.__name__, self.key)


class DistPackage(Package):
    """Wrapper class for pkg_resources.Distribution instances

      :param obj: pkg_resources.Distribution to wrap over
      :param req: optional ReqPackage object to associate this
                  DistPackage with. This is useful for displaying the
                  tree in reverse
    """

    def __init__(self, obj, req=None):
        super(DistPackage, self).__init__(obj)
        self.version_spec = None
        self.req = req
        # Metadata properties for filtered_graph
        self.include = False
        self.exclude = False
        self.required_versions = set()
        self.conflict_target = False
        self.conflicted_reqs = set()
        self.is_root = True

    def render_as_root(self, frozen):
        if not frozen:
            return '{0}=={1}'.format(self.project_name, self.version)
        else:
            return self.__class__.frozen_repr(self._obj)

    def render_as_branch(self, frozen):
        assert self.req is not None
        if not frozen:
            parent_ver_spec = self.req.version_spec
            parent_str = self.req.project_name
            if parent_ver_spec:
                parent_str += parent_ver_spec
            return (
                '{0}=={1} [requires: {2}]'
            ).format(self.project_name, self.version, parent_str)
        else:
            return self.render_as_root(frozen)

    def as_requirement(self):
        """Return a ReqPackage representation of this DistPackage"""
        return ReqPackage(self._obj.as_requirement(), dist=self)

    def as_required_by(self, req):
        """Return a DistPackage instance associated to a requirement

        This association is necessary for displaying the tree in
        reverse.

        :param ReqPackage req: the requirement to associate with
        :returns: DistPackage instance

        """
        return self.__class__(self._obj, req)

    def as_dict(self):
        return {'key': self.key,
                'package_name': self.project_name,
                'installed_version': self.version}


class ReqPackage(Package):
    """Wrapper class for Requirements instance

      :param obj: The `Requirements` instance to wrap over
      :param dist: optional `pkg_resources.Distribution` instance for
                   this requirement
    """

    UNKNOWN_VERSION = '?'

    def __init__(self, obj, dist=None):
        super(ReqPackage, self).__init__(obj)
        self.dist = dist
        # Metadata properties for filtered_graph
        self.is_missing = False

    @property
    def version_spec(self):
        specs = sorted(self._obj.specs, reverse=True)  # `reverse` makes '>' prior to '<'
        return ','.join([''.join(sp) for sp in specs]) if specs else None

    @property
    def installed_version(self):
        if not self.dist:
            return guess_version(self.key, self.UNKNOWN_VERSION)
        return self.dist.version

    def is_conflicting(self):
        """If installed version conflicts with required version"""
        # unknown installed version is also considered conflicting
        if self.installed_version == self.UNKNOWN_VERSION:
            return True
        ver_spec = (self.version_spec if self.version_spec else '')
        req_version_str = '{0}{1}'.format(self.project_name, ver_spec)
        req_obj = pkg_resources.Requirement.parse(req_version_str)
        return self.installed_version not in req_obj

    def render_as_root(self, frozen):
        if not frozen:
            return '{0}=={1}'.format(self.project_name, self.installed_version)
        elif self.dist:
            return self.__class__.frozen_repr(self.dist._obj)
        else:
            return self.project_name

    def render_as_branch(self, frozen):
        if not frozen:
            req_ver = self.version_spec if self.version_spec else 'Any'
            return (
                '{0} [required: {1}, installed: {2}]'
                ).format(self.project_name, req_ver, self.installed_version)
        else:
            return self.render_as_root(frozen)

    def as_dict(self):
        return {'key': self.key,
                'package_name': self.project_name,
                'installed_version': self.installed_version,
                'required_version': self.version_spec}


########################################################################################


class MissingPackage(Package):
    """Placeholder package class for missing dependencies."""

    def __init__(self, name=None):
        # No super() as this is just a placeholder, not a real Package
        self.project_name = name
        self.key = name
        self.include = True
        self.exclude = False
        self.conflict_target = True
        self.conflicted_reqs = set()
        self.required_versions = set()

    @property
    def version(self):
        """Read only property, version can only be MISSING."""
        return 'MISSING'

    def from_req(self, req):
        """Build a MissingPackage from a ReqPackage.
        Labels your req as missing and builds a new MissingPackage.

        :param ReqPackage req: The ReqPackage with no target DistPackage
        :returns missing: A new MissingPackage matching the Req
        :rtype MissingPackage
        """
        req.is_missing = True
        self.key = req.key
        self.project_name = req.project_name
        if req.version_spec is not None:
            self.required_versions.add(req.version_spec)
        return self


def _get_dist_package(tree, req, index=None):
    """Helper, tries to find the DistPackage in the tree by name.

    :param collections.OrderedDict tree: The tree to search
    :param ReqPackage req: The required package to find a DistPackage for
    :param dict index: The tree index, if available
    :returns node: The tree DistPackage whose name was supplied or a MissingPackage
    :rtype DistPackage
    """
    def by_key(key):
        if index and key in index:
            return index[key]
        for node in tree.keys():
            if node.key == key:
                return node
        return None

    if req.dist:
        return req.dist
    # The DistPackage is probably missing repair the tree if necessary
    dist = by_key(req.key)
    if dist is None:
        dist = MissingPackage().from_req(req=req)
        tree[dist] = []
    req.dist = dist
    return dist


class TreeWalker(object):
    """Helper to walk up or down the tree yielding ReqPackages & DistPackages.
    Walks from the supplied package returning:
        req - the ReqPackage view of the current package
        dist - the DistPackage view of the current package
        stack - the list of keys on the branch from the start to the current package
    """

    def __init__(self, tree, visit_once=False, reverse=False):
        """Init the generator with the tree and index.

        :param collections.OrderedDict tree: The Tree to walk
        :param bool visit_once: Whether to only yield each node once overall
        :param bool reverse: Whether to invert the tree and walk leaf to root
        """
        self._reverse = reverse
        self._tree = tree
        self._index = {p.key: p for p in self._tree}
        if reverse:
            self._rev_tree = reverse_tree(tree)
            self._rev_index = {p.key: p for p in self._rev_tree}
        self._visited = []
        self.visit_once = visit_once

    def forget_visited(self):
        """Purge the visited list."""
        self._visited = []

    def walk(self, package):
        """Walk the branches and leaves from the given package.

        Yields all the branches and leaves below the given package,
        not the package itself.

        :param DistPackage package: The package to start from
        :returns (ReqPackage, DistPackage, Stack): Both views of the package and stack
        :rtype Tuple
        """
        # TODO - Change Yield Froms into Python2 compatible nastiness
        branches = []
        if self._reverse:
            branches = list(self._reverse_walk(package=package, stack=[]))
        else:
            branches = list(self._normal_walk(package=package, stack=[]))
        for (req, dist, stack) in branches:
            yield (req, dist, stack)

    def _check_visited(self, package):
        """If in visit once mode, have we already been here?"""
        if self.visit_once:
            if package.key in self._visited:
                return True
            self._visited.append(package.key)
        return False

    @staticmethod
    def _check_loop(package, stack):
        """Detect loops and don't continue looping."""
        if package.key in stack:
            return True
        stack.append(package.key)

    def _normal_walk(self, package, stack):
        """Walk in the 'normal' direction down the tree.
        Add Missing packages to the tree if found."""
        if self._check_loop(package=package, stack=stack):
            return
        for req in self._tree[package]:
            if req.key not in self._visited:
                if not self._check_visited(package=req):
                    dist = _get_dist_package(
                            self._tree, req=req, index=self._index)
                    yield (req, dist, stack)
                    leaves = list(self._normal_walk(package=dist, stack=stack.copy()))
                    for (req, dist, stack) in leaves:
                        yield (req, dist, stack)

    def _reverse_walk(self, package, stack):
        """Walk up the reversed tree. No missing packages."""
        if self._check_loop(package=package, stack=stack):
            return
        package = self._rev_index.get(package.key, None)
        if package is not None:
            for pkg in self._rev_tree[package]:
                # if pkg.key not in self._visited:
                if not self._check_visited(package=pkg):
                    req = self._rev_index[pkg.key]
                    dist = self._index[pkg.key]
                    yield (req, dist, stack)
                    yield from self._reverse_walk(package=dist, stack=stack.copy())


def dump_enhanced_graphviz(
        tree, show_only=None, exclude=None, output_format='dot'):
    """Output a filtered dependency graph, enhanced wrapper for dump_graphviz().

    :param dict tree: dependency graph
    :param set show_only: set of select packages to be shown in the
                          output. This is optional arg, default: None.
    :param set exclude: set of select packages to be excluded from the
                          output. This is optional arg, default: None.
    :param string output_format: output format
    :returns: representation of tree in the specified output format
    :rtype: str or binary representation depending on the output format
    """
    filtered_tree = filter_tree(tree=tree, show_only=show_only, exclude=exclude)
    output = dump_graphviz(
            tree=filtered_tree,
            output_format=output_format,
            enhanced_graph=True)
    return output


def _label_conflicted_packages(tree, package):
    """Walk up the tree from a conflicted package and label affected packages.

    :param collections.OrderedDict tree: The tree to walk
    :param DistPackage package: The package to work from
    """
    if package.conflict_target:
        walker = TreeWalker(tree=tree, visit_once=True, reverse=True)
        for req, dist, stack in walker.walk(package=package):
            dist.conflicted_reqs.add(package.project_name)


def _mark_versions(tree, index=None):
    """Work the tree, mark the various requested versions on each DistPackage.

    :param collections.OrderedDict tree: The tree to process
    :param dict index: The tree key index if available
    """
    for package, reqs in tree.items():
        for req in reqs:
            dist = req.dist or _get_dist_package(tree=tree, req=req, index=index)
            if req.version_spec is not None:
                dist.required_versions.add(req.version_spec)
            if req.is_conflicting():
                dist.conflict_target = True


def _mark_roots(tree):
    """Walks the tree and marks the root nodes as is_root=True.

    A root is any package which is not required by another package.

    :param tree (collections.OrderedDict): The DistPackage Tree to label
    """
    walker = TreeWalker(tree=tree, visit_once=True)
    for pkg in list(tree.keys()):
        for _, dist, _ in walker.walk(pkg):
            dist.is_root = False


def _mark_include_exclude(tree,
                          show_only=None,
                          exclude=None):
    """Recursively marks the packages in the tree with include or exclude flags.
    Marks top level exclusions
    Only marks branch or leaf dependencies if it is not already actively
    included by an include request.

    :param collections.OrderedDict tree: The tree of DistPackage s to process
    :param set show_only: set of select packages to be shown in the
                          output. This is optional arg, default: None.
    :param set exclude: set of select packages to be excluded from the
                          output. This is optional arg, default: None.
    """
    nodes = list(tree.keys())
    includes = []
    excludes = []

    # Filter the nodes based on the list of packages to include
    if show_only:
        includes = [p for p in nodes
                    if p.key in show_only or p.project_name in show_only]
    if exclude:
        excludes = [p for p in tree.keys()
                    if p.key in exclude or p.project_name in exclude]

    walker = TreeWalker(tree=tree, visit_once=False)

    # Mark nodes and branches the user included with -p
    # Do not mark explicitly excluded nodes or their children
    for package in includes:
        package.include = True
        for req, dist, stack in walker.walk(package=package):
            if dist not in excludes:
                if not any([pkg.key in stack for pkg in excludes]):
                    dist.include = True

    # Mark nodes and branches the user excluded with -e, avoid included branches
    for package in excludes:
        package.exclude = True
        for req, dist, stack in walker.walk(package=package):
            if not dist.include:
                dist.exclude = True

    # If there were no explicit includes, mark everything except excluded branches
    if not includes:
        _mark_roots(tree=tree)
        roots = [p for p in tree.keys() if p.is_root]
        for package in roots:
            # Need to walk down each branch, ask if we're on an exclude branch
            if not package.exclude:
                package.include = True
                for req, dist, stack in walker.walk(package=package):
                    if not any([pkg.key in stack for pkg in excludes]):
                        if dist not in excludes:
                            dist.include = True
                            dist.exclude = False


def filter_tree(tree, show_only=None, exclude=None):
    """Filter the tree using packages and exclude lists for graph rendering.
    Walk the tree and label the packages affected by conflicts.

    :param dict tree: The tree to filter
    :param set show_only: set of select packages to be shown in the
                          output. This is optional arg, default: None.
    :param set exclude: set of select packages to be excluded from the
                          output. This is optional arg, default: None.
    :returns tree: The input tree, filtered
    :rtype collections.OrderedDict
    """
    tree = sorted_tree(tree)
    _mark_include_exclude(tree=tree, show_only=show_only, exclude=exclude)

    # Prune the tree, prune packages, then dependencies (or they reappear in the graph)
    for package in list(tree.keys()):
        if not package.include or package.exclude:
            del tree[package]

    keys = [package.key for package in tree]
    for dist, deps in tree.items():
        tree[dist] = [dep for dep in deps if dep.key in keys]

    local_index = {p.key: p for p in tree.keys()}
    _mark_versions(tree=tree, index=local_index)

    for package in list(tree.keys()):
        _label_conflicted_packages(package=package, tree=tree)

    return tree


def build_package_tree(tree):
    """Builds up a traversable PackageTree of the Dist and Req Packages."""
    package_tree = PackageTree()
    for dist in tree.keys():
        package_tree.add_package(dist_package=dist)
    for dist, reqs in tree.items():
        package_tree.add_requirements(dist_package=dist, req_packages=reqs)
    return package_tree


class PackageTree:
    """Traversable tree wrapper for the Distribution and Requirement packages."""

    def __init__(self):
        self._nodes = {}

    def add_package(self, dist_package):
        """Add a package node to the tree.

        :param DistPackage dist_package: The DistPackage to add to the tree
        """
        if dist_package.key in self._nodes:
            raise KeyError('DistPackage {0} already in tree'.format(dist_package.key))
        self._nodes[dist_package.key] = TreeNode(dist_package)

    def add_requirements(self, dist_package, req_packages):
        """Add the requirement edges to the distribution nodes.

        :param DistPackage dist_package: The DistPackage to add edges from
        :param list req_packages: The ReqPackages of this DistPackage
        """
        if req_packages:
            source_node = self._nodes[dist_package.key]

            for req in req_packages:
                if req.key not in self._nodes:
                    # TODO - Insert MissingPackage builder here
                    raise KeyError('No node found for ReqPackage: {0}'.format(req.key))
                target_node = self._nodes[req.key]
                source_node.add_requirement(req_package=req, target_node=target_node)
                target_node.add_required_by(source_node=source_node)

    def get_nodes(self):
        """Get a list of all node keys.

        :returns list: A list of the node keys
        """
        return self._nodes.keys()

    def get_node(self, key):
        """Get a TreeNode by it's key.

        :returns TreeNode: The node with the matching key
        """
        return self._nodes[key]

    def roots(self):
        """Returns the keys of all the root nodes.

        :returns list: A list of node keys
        """
        return [key for key, node in self._nodes.items() if node.is_root]

    def leaves(self):
        """Returns the keys of all the leaf nodes.

        :returns list: A list of node keys
        """
        return [key for key, node in self._nodes.items() if node.is_leaf]

    def delete(self, key):
        """Delete a specified node by key.
        Deletes the node and the requirements edges that point to it.

        :param str: The key of the node to delete
        """
        target_node = self._nodes[key]
        for impacted in target_node.required_by:
            impacted.delete_requirement(key)
        for impacted in target_node.requirements.values():
            impacted.delete_required_by(target_node)
        del self._nodes[key]


class TreeNode:
    """Package node for the Tree."""

    def __init__(self, dist_package):
        """A PackageTree Node container for a DistPackage and it's ReqPackages.

        :param DistPackage dist_package: The DistPackage to wrap
        """
        if not isinstance(dist_package, DistPackage):
            raise TypeError('TreeNode requires DistPackages')
        self._dist_package = dist_package
        self.key = dist_package.key
        self.requirements = {}
        self.required_by = []
        self.include = False
        self.exclude = False

    @property
    def dist_package(self):
        """Returns the DistPackage.

        :returns DistPackage
        """
        return self._dist_package

    def add_requirement(self, req_package, target_node):
        """Adds a requirement edge to the graph.

        :param ReqPackage req_package: The requirement package to add to the graph
        :param TreeNode target_node: The TreeNode containing the target DistPackage
        """
        if not req_package in self.requirements:
            self.requirements[req_package] = target_node

    def delete_requirement(self, key):
        """Delete a requirement edge from the graph."""
        for req in list(self.requirements.keys()):
            if req.key == key:
                del self.requirements[req]

    def add_required_by(self, source_node):
        """Adds a required by backward edge to the graph.

        :param TreeNode source_node: The source node requiring this node
        """
        self.required_by.append(source_node)

    def delete_required_by(self, node):
        """Delete a required by edge from the graph.

        :param TreeNode node: The node to remove
        """
        if node in self.required_by:
            self.required_by.remove(node)

    @property
    def is_root(self):
        """Is this node a 'root' in the tree? i.e. is not a Requirement of any other.

        :returns bool
        """
        return len(self.required_by) == 0

    @property
    def is_leaf(self):
        """Is this node a 'leaf' in the tree? i.e. is not required by any other.

        :returns bool
        """
        return len(self.requirements) == 0

    def get_requirements(self, req=None, stack=None):
        """Recursively gets a list of tuples of the requirements below this node.

        :param ReqPackage req: The requirement package that requires this node
        :param list stack: The node key stack to get to here
        :returns list: Tuples of (DistPackage, ReqPackage, stack)
        """
        stack = stack or []
        if self.key in stack:
            return                  # Exit circular dependencies
        stack.append(self.key)
        yield (req, self.dist_package, stack)
        for req, node in self.requirements.items():
            for req, inner_node, inner_stack in node.get_requirements(
                    req=req, stack=stack.copy()):
                yield (req, inner_node, inner_stack)


########################################################################################


def render_tree(tree, list_all=True, show_only=None, frozen=False, exclude=None):
    """Convert tree to string representation

    :param dict tree: the package tree
    :param bool list_all: whether to list all the pgks at the root
                          level or only those that are the
                          sub-dependencies
    :param set show_only: set of select packages to be shown in the
                          output. This is optional arg, default: None.
    :param bool frozen: whether or not show the names of the pkgs in
                        the output that's favourable to pip --freeze
    :param set exclude: set of select packages to be excluded from the
                          output. This is optional arg, default: None.
    :returns: string representation of the tree
    :rtype: str

    """
    tree = sorted_tree(tree)
    branch_keys = set(r.key for r in flatten(tree.values()))
    nodes = tree.keys()
    use_bullets = not frozen

    key_tree = dict((k.key, v) for k, v in tree.items())
    get_children = lambda n: key_tree.get(n.key, [])

    if show_only:
        nodes = [p for p in nodes
                 if p.key in show_only or p.project_name in show_only]
    elif not list_all:
        nodes = [p for p in nodes if p.key not in branch_keys]

    def aux(node, parent=None, indent=0, chain=None):
        if exclude and (node.key in exclude or node.project_name in exclude):
            return []
        if chain is None:
            chain = [node.project_name]
        node_str = node.render(parent, frozen)
        if parent:
            prefix = ' '*indent + ('- ' if use_bullets else '')
            node_str = prefix + node_str
        result = [node_str]
        children = [aux(c, node, indent=indent+2,
                        chain=chain+[c.project_name])
                    for c in get_children(node)
                    if c.project_name not in chain]
        result += list(flatten(children))
        return result

    lines = flatten([aux(p) for p in nodes])
    return '\n'.join(lines)


def render_json(tree, indent):
    """Converts the tree into a flat json representation.

    The json repr will be a list of hashes, each hash having 2 fields:
      - package
      - dependencies: list of dependencies

    :param dict tree: dependency tree
    :param int indent: no. of spaces to indent json
    :returns: json representation of the tree
    :rtype: str

    """
    return json.dumps([{'package': k.as_dict(),
                        'dependencies': [v.as_dict() for v in vs]}
                       for k, vs in tree.items()],
                      indent=indent)


def render_json_tree(tree, indent):
    """Converts the tree into a nested json representation.

    The json repr will be a list of hashes, each hash having the following fields:
      - package_name
      - key
      - required_version
      - installed_version
      - dependencies: list of dependencies

    :param dict tree: dependency tree
    :param int indent: no. of spaces to indent json
    :returns: json representation of the tree
    :rtype: str

    """
    tree = sorted_tree(tree)
    branch_keys = set(r.key for r in flatten(tree.values()))
    nodes = [p for p in tree.keys() if p.key not in branch_keys]
    key_tree = dict((k.key, v) for k, v in tree.items())
    get_children = lambda n: key_tree.get(n.key, [])

    def aux(node, parent=None, chain=None):
        if chain is None:
            chain = [node.project_name]

        d = node.as_dict()
        if parent:
            d['required_version'] = node.version_spec if node.version_spec else 'Any'
        else:
            d['required_version'] = d['installed_version']

        d['dependencies'] = [
            aux(c, parent=node, chain=chain+[c.project_name])
            for c in get_children(node)
            if c.project_name not in chain
        ]

        return d

    return json.dumps([aux(p) for p in nodes], indent=indent)

def _node_style(package):
    """Style the graphviz node for enhanced graph based on the package info."""

    def build_label():
        """Builds long descriptive label for enhanced graph."""
        label = [package.project_name,
                 'Ver: {0}'.format(package.version)]
        if package.required_versions:
            label.append('Req: {0}'.format(required_versions))
        if package.conflicted_reqs:
            label.append(
                    'Conflicted: {0}'.format(conflicted_reqs))
        return '\n'.join(label)

    style = {}
    required_versions = ', '.join(package.required_versions)
    conflicted_reqs = '\n'.join(package.conflicted_reqs)

    style['label'] = build_label()
    style['project_name'] = package.project_name
    style['version'] = str(package.version)
    style['required_versions'] = required_versions

    if package.conflicted_reqs:
        style['conflicted_reqs'] = conflicted_reqs
        style['color'] = 'khaki'
        style['style'] = 'filled'

    if package.conflict_target:
        style['color'] = 'orange'
        style['style'] = 'filled'
        style['conflict_target'] = 'True'

    if package.version == 'MISSING':
        style['missing'] = 'True'
        style['color'] = 'red'
        style['style'] = 'filled'
        style['fontcolor'] = 'white'

    return style


def _edge_style(dep):
    """Style the graph edges for enhanced graph."""
    style = {}
    if dep.is_conflicting():
        style['color'] = 'orange'
        style['fontcolor'] = 'orange'
        style['conflict'] = 'True'
    if dep.is_missing:
        style['color'] = 'red'
        style['fontcolor'] = 'red'
        style['missing'] = 'True'
    return style


def dump_graphviz(tree, output_format='dot', enhanced_graph=False):
    """Output dependency graph as one of the supported GraphViz output formats.

    :param dict tree: dependency graph
    :param string output_format: output format
    :param bool enhanced_graph: Add the graph extras
    :returns: representation of tree in the specified output format
    :rtype: str or binary representation depending on the output format

    """
    try:
        from graphviz import backend, Digraph
    except ImportError:
        print('graphviz is not available, but necessary for the output '
              'option. Please install it.', file=sys.stderr)
        sys.exit(1)

    if output_format not in backend.FORMATS:
        print('{0} is not a supported output format.'.format(output_format),
              file=sys.stderr)
        print('Supported formats are: {0}'.format(
            ', '.join(sorted(backend.FORMATS))), file=sys.stderr)
        sys.exit(1)

    graph = Digraph(format=output_format)

    for package, deps in tree.items():
        # Case insensitive package names, https://www.python.org/dev/peps/pep-0426/#name
        key = package.key.lower()
        label = '{0}\n{1}'.format(package.project_name, package.version)
        # Add more attributes to the graph nodes
        if enhanced_graph:
            node_style = _node_style(package)
            graph.node(key, **node_style)
        else:
            graph.node(key, label=label)

        for dep in deps:
            label = dep.version_spec or 'any'
            if enhanced_graph:
                edge_style = _edge_style(dep)
                graph.edge(key, dep.key, label=label, **edge_style)
            else:
                graph.edge(key, dep.key, label=label)

    # Allow output of dot format, even if GraphViz isn't installed.
    if output_format == 'dot':
        return graph.source

    # As it's unknown if the selected output format is binary or not, try to
    # decode it as UTF8 and only print it out in binary if that's not possible.
    try:
        return graph.pipe().decode('utf-8')
    except UnicodeDecodeError:
        return graph.pipe()


def print_graphviz(dump_output):
    """Dump the data generated by GraphViz to stdout.

    :param dump_output: The output from dump_graphviz
    """
    if hasattr(dump_output, 'encode'):
        print(dump_output)
    else:
        with os.fdopen(sys.stdout.fileno(), 'wb') as bytestream:
            bytestream.write(dump_output)


def conflicting_deps(tree):
    """Returns dependencies which are not present or conflict with the
    requirements of other packages.

    e.g. will warn if pkg1 requires pkg2==2.0 and pkg2==1.0 is installed

    :param tree: the requirements tree (dict)
    :returns: dict of DistPackage -> list of unsatisfied/unknown ReqPackage
    :rtype: dict

    """
    conflicting = defaultdict(list)
    for p, rs in tree.items():
        for req in rs:
            if req.is_conflicting():
                conflicting[p].append(req)
    return conflicting


def cyclic_deps(tree):
    """Return cyclic dependencies as list of tuples

    :param list pkgs: pkg_resources.Distribution instances
    :param dict pkg_index: mapping of pkgs with their respective keys
    :returns: list of tuples representing cyclic dependencies
    :rtype: generator

    """
    key_tree = dict((k.key, v) for k, v in tree.items())
    get_children = lambda n: key_tree.get(n.key, [])
    cyclic = []
    for p, rs in tree.items():
        for req in rs:
            if p.key in map(attrgetter('key'), get_children(req)):
                cyclic.append((p, req, p))
    return cyclic


def get_parser():
    parser = argparse.ArgumentParser(description=(
        'Dependency tree of the installed python packages'
    ))
    parser.add_argument('-v', '--version', action='version',
                        version='{0}'.format(__version__))
    parser.add_argument('-f', '--freeze', action='store_true',
                        help='Print names so as to write freeze files')
    parser.add_argument('-a', '--all', action='store_true',
                        help='list all deps at top level')
    parser.add_argument('-l', '--local-only',
                        action='store_true', help=(
                            'If in a virtualenv that has global access '
                            'do not show globally installed packages'
                        ))
    parser.add_argument('-u', '--user-only', action='store_true',
                        help=(
                            'Only show installations in the user site dir'
                        ))
    parser.add_argument('-w', '--warn', action='store', dest='warn',
                        nargs='?', default='suppress',
                        choices=('silence', 'suppress', 'fail'),
                        help=(
                            'Warning control. "suppress" will show warnings '
                            'but return 0 whether or not they are present. '
                            '"silence" will not show warnings at all and '
                            'always return 0. "fail" will show warnings and '
                            'return 1 if any are present. The default is '
                            '"suppress".'
                        ))
    parser.add_argument('-r', '--reverse', action='store_true',
                        default=False, help=(
                            'Shows the dependency tree in the reverse fashion '
                            'ie. the sub-dependencies are listed with the '
                            'list of packages that need them under them.'
                        ))
    parser.add_argument('-p', '--packages',
                        help=(
                            'Comma separated list of select packages to show '
                            'in the output. If set, --all will be ignored.'
                        ))
    parser.add_argument('-e', '--exclude',
                        help=(
                            'Comma separated list of select packages to exclude '
                            'from the output. If set, --all will be ignored.'
                        ), metavar='PACKAGES')
    parser.add_argument('-j', '--json', action='store_true', default=False,
                        help=(
                            'Display dependency tree as json. This will yield '
                            '"raw" output that may be used by external tools. '
                            'This option overrides all other options.'
                        ))
    parser.add_argument('--json-tree', action='store_true', default=False,
                        help=(
                            'Display dependency tree as json which is nested '
                            'the same way as the plain text output printed by default. '
                            'This option overrides all other options (except --json).'
                        ))
    parser.add_argument('--graph-output', dest='output_format',
                        help=(
                            'Print a dependency graph in the specified output '
                            'format. Available are all formats supported by '
                            'GraphViz, e.g.: dot, jpeg, pdf, png, svg'
                        ))
    parser.add_argument('-n', '--enhanced-graph', action='store_true', default=False,
                        help=(
                            'Add enhanced graph features to graph output, '
                            'color labelling for missing and conflicting dependencies, '
                            'additional graph metadata (e.g. for Gephi rendering), '
                            'labels for required versions.'
                        ))
    return parser


def _get_args():
    parser = get_parser()
    return parser.parse_args()


def main():
    args = _get_args()
    pkgs = get_installed_distributions(local_only=args.local_only,
                                       user_only=args.user_only)

    dist_index = build_dist_index(pkgs)
    tree = construct_tree(dist_index)

    show_only = set(args.packages.split(',')) if args.packages else None
    exclude = set(args.exclude.split(',')) if args.exclude else None

    if args.json:
        print(render_json(tree, indent=4))
        return 0
    elif args.json_tree:
        print(render_json_tree(tree, indent=4))
        return 0
    elif args.output_format:
        if args.enhanced_graph:
            output = dump_enhanced_graphviz(
                    tree,
                    show_only=show_only,
                    exclude=exclude,
                    output_format=args.output_format)
        else:
            output = dump_graphviz(tree, output_format=args.output_format)
        print_graphviz(output)
        return 0

    return_code = 0

    # show warnings about possibly conflicting deps if found and
    # warnings are enabled
    if args.warn != 'silence':
        conflicting = conflicting_deps(tree)
        if conflicting:
            print('Warning!!! Possibly conflicting dependencies found:',
                  file=sys.stderr)
            for p, reqs in conflicting.items():
                pkg = p.render_as_root(False)
                print('* {}'.format(pkg), file=sys.stderr)
                for req in reqs:
                    req_str = req.render_as_branch(False)
                    print(' - {}'.format(req_str), file=sys.stderr)
            print('-'*72, file=sys.stderr)

        cyclic = cyclic_deps(tree)
        if cyclic:
            print('Warning!! Cyclic dependencies found:', file=sys.stderr)
            for a, b, c in cyclic:
                print('* {0} => {1} => {2}'.format(a.project_name,
                                                   b.project_name,
                                                   c.project_name),
                      file=sys.stderr)
            print('-'*72, file=sys.stderr)

        if args.warn == 'fail' and (conflicting or cyclic):
            return_code = 1

    if show_only and exclude and (show_only & exclude):
        print('Conflicting packages found in --packages and --exclude lists.', file=sys.stderr)
        sys.exit(1)

    tree = render_tree(tree if not args.reverse else reverse_tree(tree),
                       list_all=args.all, show_only=show_only,
                       frozen=args.freeze, exclude=exclude)
    print(tree)
    return return_code


if __name__ == '__main__':
    sys.exit(main())
