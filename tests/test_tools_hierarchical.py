import cpmpy as cp
from cpmpy.tools.explain import (
    ConstraintNode,
    constraint_node_from_dict,
    constraint_node_to_dict,
    hierarchical_marco,
    marco,
    umus,
)
from cpmpy.transformations.normalize import toplevel_list


def _php_constraints():
    x = cp.boolvar(shape=(5, 3), name="x")
    model = cp.Model()
    model += cp.cpm_array(x.sum(axis=1)) >= 1
    model += cp.cpm_array(x.sum(axis=0)) <= 1
    return toplevel_list(model.constraints, merge_and=False)


class TestUMUS:
    def test_circular(self):
        x = cp.intvar(0, 3, shape=4, name="x")
        # circular "bigger than", UNSAT
        cons = [
            x[0] > x[1],
            x[1] > x[2],
            x[2] > x[0],

            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]

        cumu = umus(cons)
        # the only MUS is the first 3 constraints, so CUMU == that MUS
        assert set(cumu) == set(cons[:3])

    def test_union_of_muses(self):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b, d]
        mus2 = [a, b, c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        cumu = umus([a, b, c, d], hard)

        # CUMU must be a superset of at least one MUS, and a subset of all soft constraints
        assert set(cumu) <= {a, b, c, d}
        assert set(mus1) <= set(cumu) or set(mus2) <= set(cumu)

    def test_php(self):
        cons = _php_constraints()
        cumu = umus(cons)
        assert not cp.Model(cumu).solve()
        assert set(cumu) <= set(cons)


class TestConstraintNode:
    def test_basic_tree(self):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        root = ConstraintNode("root")
        g1 = root.add_child("g1")
        g1.add_child("g1a").constraints.append(a)
        g1.add_child("g1b").constraints.append(b)
        g2 = root.add_child("g2")
        g2.constraints.extend([c, d])

        assert set(root.all_constraints()) == {a, b, c, d}
        assert {leaf.name for leaf in root.leaves()} == {"g1a", "g1b", "g2"}
        assert {node.name for node in root.iter_nodes()} == {"root", "g1", "g1a", "g1b", "g2"}

        g1a = g1.add_child("g1a")  # should return existing child
        assert g1a.get_full_name() == "g1 g1a"

    def test_serialization_roundtrip(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        constraints = [a, b, c]
        index_of = {id(con): i for i, con in enumerate(constraints)}

        root = ConstraintNode("root")
        g1 = root.add_child("g1")
        g1.constraints.append(constraints[0])
        g2 = root.add_child("g2")
        g2.add_child("g2a").constraints.append(constraints[1])
        g2.add_child("g2b").constraints.append(constraints[2])

        spec = constraint_node_to_dict(root, index_of)
        rebuilt = constraint_node_from_dict(spec, constraints)

        assert set(rebuilt.all_constraints()) == set(root.all_constraints())
        names_orig = sorted(node.get_full_name() for node in root.iter_nodes())
        names_new = sorted(node.get_full_name() for node in rebuilt.iter_nodes())
        assert names_orig == names_new


class TestHierarchicalMarco:
    def _build_tree(self, cons):
        # group php "row" constraints under one node, "column" constraints under another
        root = ConstraintNode("root")
        rows = root.add_child("rows")
        cols = root.add_child("cols")
        for i, con in enumerate(cons[:5]):
            rows.add_child(f"row{i}").constraints.append(con)
        for i, con in enumerate(cons[5:]):
            cols.add_child(f"col{i}").constraints.append(con)
        return root

    def test_matches_flat_marco(self):
        cons = _php_constraints()
        root = self._build_tree(cons)

        flat_subsets = list(marco(soft=cons))
        flat_musses = [frozenset(ss) for kind, ss in flat_subsets if kind == "MUS"]
        flat_mcses = [frozenset(ss) for kind, ss in flat_subsets if kind == "MCS"]

        hier_subsets = list(hierarchical_marco(root, initial_level=2))
        hier_musses = [frozenset(c) for kind, c, _ in hier_subsets if kind == "MUS"]
        hier_mcses = [frozenset(c) for kind, c, _ in hier_subsets if kind == "MCS"]

        assert set(hier_musses) == set(flat_musses)
        assert set(hier_mcses) == set(flat_mcses)

    def test_refines_to_leaves(self):
        cons = _php_constraints()
        root = self._build_tree(cons)

        # start coarse: each top-level group ("rows"/"cols") is one constraint
        hier_subsets = list(hierarchical_marco(root, initial_level=1))

        all_names = set()
        for _, _, names in hier_subsets:
            all_names.update(names)

        # refinement should reach individual row/col leaves, not just the two top groups
        assert any(name.startswith("rows row") for name in all_names)
        assert any(name.startswith("cols col") for name in all_names)
