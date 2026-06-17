"""
    Hierarchical constraint groups, used as input for :func:`hierarchical_marco`.

    =================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        ConstraintNode
        constraint_node_to_dict
        constraint_node_from_dict
"""

from dataclasses import dataclass, field

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list


@dataclass
class ConstraintNode:
    """
        A node in a hierarchy of constraint groups.

        Each node has a name, a (possibly empty) list of constraints that live directly
        at this node, and a list of child nodes. The constraints of a node's subtree
        are the constraints at the node itself plus those of all its descendants.

        Used as the input hierarchy for :func:`hierarchical_marco`.
    """
    name: str
    constraints: list = field(default_factory=list)
    children: list = field(default_factory=list)
    parent: "ConstraintNode" = field(default=None, repr=False, compare=False)

    def add_child(self, name):
        """Get or create a child node with the given name."""
        for child in self.children:
            if child.name == name:
                return child
        child = ConstraintNode(name, parent=self)
        self.children.append(child)
        return child

    def iter_nodes(self):
        """Pre-order iterator over this node and all its descendants."""
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def leaves(self):
        """All nodes in this subtree that have no children."""
        if not self.children:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def all_constraints(self):
        """All constraints in this node's subtree, recursively."""
        result = list(self.constraints)
        for child in self.children:
            result.extend(child.all_constraints())
        return result

    def get_grouped_constraint(self):
        """The conjunction of all constraints in this subtree, or None if there are none."""
        cons = toplevel_list(self.all_constraints(), merge_and=False)
        if not cons:
            return None
        if len(cons) == 1:
            return cons[0]
        return cp.all(cons)

    def get_full_name(self):
        """Dotted path name from the root (exclusive) to this node, for reporting."""
        parts = []
        node = self
        while node is not None and node.parent is not None:
            parts.append(node.name)
            node = node.parent
        return " ".join(reversed(parts))

    def level(self):
        """Depth of this node below the root (root is level 0, its children level 1, ...)."""
        lvl = 0
        node = self.parent
        while node is not None:
            lvl += 1
            node = node.parent
        return lvl


def activate_descendants_at_level(node, target_level):
    """
        Refine `node` to `target_level`, mirroring the "scripted" refinement of the
        defense-rostering experiments: descend from `node` until reaching nodes at depth
        ``>= target_level`` (or a leaf reached earlier) — those become **active** groups —
        while every strictly-intermediate node becomes **partitioned** (a pass-through used
        only for map-solver bookkeeping). This can *skip* intermediate levels (e.g. a
        single-child "renaming" level), unlike one-level-at-a-time refinement.

        Only children that carry constraints (``get_grouped_constraint() is not None``) are
        considered.

        :return: ``(active, partitioned)``, two lists of :class:`ConstraintNode`.
    """
    active, partitioned = [], []

    def recurse(n):
        children = [c for c in n.children if c.get_grouped_constraint() is not None]
        if n.level() >= target_level or not children:
            active.append(n)
        else:
            partitioned.append(n)
            for c in children:
                recurse(c)

    recurse(node)
    return active, partitioned


def constraint_node_to_dict(root, index_of):
    """
        Serialize a :class:`ConstraintNode` tree to a JSON-friendly dict.

        :param: root: the root :class:`ConstraintNode` of the tree
        :param: index_of: mapping from `id(constraint)` to an integer index into a flat
            list of constraints (e.g. as produced when writing out a benchmark instance)
    """
    return {
        "name": root.name,
        "constraints": [index_of[id(c)] for c in root.constraints],
        "children": [constraint_node_to_dict(child, index_of) for child in root.children],
    }


def constraint_node_from_dict(spec, constraints):
    """
        Reconstruct a :class:`ConstraintNode` tree from the dict produced by
        :func:`constraint_node_to_dict`.

        :param: spec: the serialized tree
        :param: constraints: flat list of constraints, indexed as referenced in `spec`
    """
    node = ConstraintNode(
        name=spec["name"],
        constraints=[constraints[i] for i in spec.get("constraints", [])],
    )
    for child_spec in spec.get("children", []):
        child = constraint_node_from_dict(child_spec, constraints)
        child.parent = node
        node.children.append(child)
    return node
