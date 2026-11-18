import numpy as np
import math
import random


class linear_transformation(np.ndarray):
    # This class allows us to work more directly with linear transformations between
    # ordered bases that aren't necessarily indexed by integers. It also changes the
    # way matrices are displayed
    def __new__(cls, input_matrix: np.ndarray,
                codomain_gens: list = [],  # the objects that index the rows
                domain_gens: list = []):  # the objects that index the columns
        # if no list of generators is given, the default indices
        # will be the inherrited from the input_matrix
        if codomain_gens == []:
            codomain_gens = list(range(np.shape(input_matrix)[0]))
        if domain_gens == []:
            domain_gens = list(range(np.shape(input_matrix)[1]))
        if np.shape(input_matrix) != (len(codomain_gens), len(domain_gens)):
            return
        # the rest of this until __get_item__ is boiler plate for subclassing np.ndarray
        obj = np.asarray(input_matrix).view(cls)
        obj.input_matrix = input_matrix
        obj.domain_gens = domain_gens
        obj.codomain_gens = codomain_gens
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.input_matrix: np.ndarray = getattr(obj, 'input_matrix', None)
        self.domain_gens: list = getattr(obj, 'domain_gens', None)
        self.codomain_gens: list = getattr(obj, 'codomain_gens', None)

    # This change allows us to access the entries of our matrix more easily.
    def __getitem__(self, key) -> int | np.ndarray:
        if isinstance(key, tuple):
            row, col = key
            if row not in self.codomain_gens:
                raise IndexError(f"Row index '{row}' does not exist.")
            if col not in self.domain_gens:
                raise IndexError(f"Column index '{col}' does not exist.")
            numerical_indices: tuple[int] = (self.codomain_gens.index(row),
                                             self.domain_gens.index(col))
            return self.input_matrix.__getitem__(numerical_indices)
        return self.input_matrix.__getitem__(key)

    # The string representation is upgraded to show row and column labels.
    def __str__(self) -> str:
        m, n = self.input_matrix.shape
        # check for trivial (co)domain
        if m * n == 0:
            return str(0)
        # the sign kwarg in the line below ensures consistent spacing between columns
        default_string: str = np.array2string(self.input_matrix, sign=' ')
        # we first need to find where the line breaks happen in the default string representation of mat
        line_br: int = default_string.find("\n") + 1
        # if default_string only has one row, default_string.find("\n") will return -1
        if line_br == 0:
            line_br: int = len(default_string)
        # this allows us to break the string into its rows and add labels to each one
        row_labels: list[str] = [str(gen) for gen in self.codomain_gens]
        l_margin: int = max([len(label) for label in row_labels]) + 1
        lines: list[str] = [row_labels[i]
                            + (l_margin - len(row_labels[i])) * " "
                            + default_string[line_br * i: line_br * (i + 1)]
                            for i in range(len(row_labels))]
        # now we make the column labels to add on top
        spacing: int = 3
        col_labels: list[str] = [str(gen) for gen in self.domain_gens]
        t_margin: int = max([len(label) for label in col_labels])
        col_labels: list[str] = [label + (t_margin - len(label)) * " "
                                 for label in col_labels]
        verticalized_col_labels: str = "".join([(l_margin + len("[[")) * " "
                                                + "".join([f"{label[j * spacing: (j + 1) * spacing]}"
                                                           for label in col_labels]) + "\n"
                                                for j in range(t_margin // spacing)])
        return verticalized_col_labels + "".join(lines)


# a simplex is just an ordered tuple of positive integers
# which are assumed to be written in their natural order
def boundary(simplex: tuple[int],
             j: int) -> tuple[int]:
    boundary_face: list[int] = list(simplex)
    del boundary_face[j]
    return tuple(boundary_face)


# intersection for tuples
def intersect(simplex1: tuple[int],
              simplex2: tuple[int]) -> tuple[int]:
    return tuple(set(simplex1) & set(simplex2))


# When called without a third argument, produces a list of all subfaces
# of simplex that aren't contained in any faces from relative_faces.
def list_of_subfaces(simplex: tuple[int],
                     relative_faces: list[tuple[int]] = [],
                     j: float = math.inf) -> list[tuple[int]]:
    # To check whether our simplex is contained in a relative face we only
    # need to consider the intersections of the relative faces with our simplex.
    intersected_relative_faces: list[tuple[int]] = list({intersect(face, simplex)
                                                         for face in relative_faces})
    # Our simplex was contained in a face from relative_faces if
    # and only if it is an element of intersected_relative_faces.
    # In this case, we don't want to include simplex in our list.
    if simplex in intersected_relative_faces:
        return []
    # Otherwise, our set of subfaces should contain at least simplex.
    subfaces: list[tuple[int]] = [simplex]
    # Now we're going to search the codimension 1 subfaces of simplex
    # to see which of them and their subfaces should be added to the list.
    for i in range(min(len(simplex), j)):
        # The use of the third argument below restricts the search
        # to subfaces obtained by deleting at indices less than i. This
        # prevents the function from looking for the same subface multiple times.
        subfaces += list_of_subfaces(boundary(simplex, i),
                                     intersected_relative_faces,
                                     i)
    return subfaces


class Simplicial_Complex:
    # Produces the simplicial complex generated by the provided list, generators,
    # of simplices. The list of simplices need not be incomparable w.r.t. inclusion;
    # a list of proper facets self.facets() will be extracted from the list of simplices.

    # quotient_gens is an optional argument that when used allows us to
    # consider spaces formed by taking the quotient of two simplicial complexes
    def __init__(self, generators: list[tuple[int]],
                 quotient_gens: list[tuple[int]] = []):
        # the list of simplices is easier to process if it is ordered so
        # that the dimensions of the simplices in the list are nonincreasing
        generators.sort(key=len)
        generators.reverse()

        self.generators: list[tuple[int]] = generators
        self.quotient_gens: list[tuple[int]] = quotient_gens

        self.dimension: int = len(generators[0]) - 1

    # Produces a simple string representation
    # which presents the faces in each dimension.
    def __str__(self) -> str:
        string: str = f"-1-faces : {self.dfaces(-1)}\n"
        for d in range(self.dimension + 1):
            string += f" {d}-faces : {self.dfaces(d)}\n"
        return string

    # Extracts from the input list of simplices a list of
    # facets by weeding out nonmaximal faces in the list.
    def facets(self) -> list[tuple[int]]:
        # top-dimensional faces are guaranteed to be facets. We
        # therefore skip checking these confirmed facets, significantly
        # speeding up this function whenever the simplicial complex is pure.
        confirmed_facets: list[tuple[int]] = [face for face in self.generators
                                              if len(face) == self.dimension + 1]
        # The start index in the for loop here skips confirmed
        # facets due to how we've ordered our set of generators.
        for i in range(len(confirmed_facets), len(self.generators)):
            ith_generator: tuple[int] = self.generators[i]
            # ith generator is assumed maximal until demonstrated otherwise.
            ith_generator_contained: bool = False
            # Check for containment within any facets confirmed so far.
            for facet in confirmed_facets:
                if set(ith_generator) < set(facet):
                    ith_generator_contained: bool = True
                    break
            # Since all generators beyond the ith are no longer than the
            # ith, ith_generator_contained comes out of this loop false if
            # and only if ith_generator is maximal with respect to inclusion.
            if not ith_generator_contained:
                confirmed_facets += ith_generator
        return [facet for facet in confirmed_facets if facet not in self.quotient_gens]

    # Produces a list of all faces in our simplicial complex
    # minus those faces contained in one or more quotient generator.
    def faces(self) -> list[tuple[int]]:
        # We'll be using our list of facets many times.
        list_of_facets: list[tuple[int]] = self.facets()
        list_of_faces: list[tuple[int]] = []
        # We iterate over indices instead of directly
        # over list_of_facets so we can easily slice it.
        for i in range(len(list_of_facets)):
            # We're trying to compile this list without needing to deal with repetitions. This is why,
            # in our search for subfaces of the ith facet in the list, we ignore subfaces contained in
            # any of the first i - 1 facets (or the quotient generators) because these were already added.
            list_of_faces += list_of_subfaces(list_of_facets[i],
                                              list_of_facets[:i] + self.quotient_gens)
        # These next few lines are just to sort the faces into a pleasing order.
        if list_of_facets != [()]:
            max_vertex: int = max(max(facet) for facet in list_of_facets)
            sort_function: function = lambda x: sum([(x[i] - i - 1) * max_vertex**(len(x) - i - 1)
                                                     for i in range(len(x))])
            list_of_faces.sort(key=sort_function)
        return list_of_faces

    # Produces a list of all d-faces in our simplicial complex
    # minus those faces contained in one or more quotient generator.
    def dfaces(self, d: int) -> list[tuple[int]]:
        return [face for face in self.faces() if len(face) == d + 1]

    # Produces the vertices of the complex (as integers, not simplices).
    def vertices(self) -> list[int]:
        return [v for (v,) in self.dfaces(0)]

    # Gives the f_vector.
    def f_vector(self) -> list[int]:
        return [len(self.dfaces(d)) for d in range(-1, self.dimension + 1)]

    # Gives the h_vector.
    def h_vector(self) -> list[int]:
        return [sum([(-1)**(k - i) * math.comb(self.dimension - i + 1, k - i) * self.f_vector()[i]
                     for i in range(k + 1)])
                for k in range(self.dimension + 2)]

    # Gives the dimension d boundary map.
    def boundary_matrix(self, d: int) -> linear_transformation:
        mat: linear_transformation = linear_transformation(np.zeros((len(self.dfaces(d-1)),
                                                                     len(self.dfaces(d))), dtype=int),
                                                           self.dfaces(d-1),
                                                           self.dfaces(d))
        for face in self.dfaces(d):
            for j in range(d+1):
                jth_boundary: tuple[int] = boundary(face, j)
                if jth_boundary in self.dfaces(d-1):
                    mat[self.dfaces(d-1).index(boundary(face, j)),
                        self.dfaces(d).index(face)] = (-1)**j
        return mat

    # Gives the dimension d up-down Laplacian.
    def UD_Laplacian(self, d: int) -> linear_transformation:
        return linear_transformation(self.boundary_matrix(d + 1).input_matrix @ self.boundary_matrix(d + 1).input_matrix.transpose(),
                                     self.dfaces(d),
                                     self.dfaces(d))

    # Gives the dimension d down-up Laplacian.
    def DU_Laplacian(self, d: int) -> linear_transformation:
        return linear_transformation(self.boundary_matrix(d).input_matrix.transpose() @ self.boundary_matrix(d).input_matrix,
                                     self.dfaces(d),
                                     self.dfaces(d))

    # Gives the dimension d total Laplacian.
    def total_Laplacian(self, d: int) -> linear_transformation:
        return linear_transformation(self.UD_Laplacian(d).input_matrix + self.DU_Laplacian(d).input_matrix,
                                     self.dfaces(d),
                                     self.dfaces(d))

    # Gives the star of simplex in the complex.
    def star(self, simplex: tuple):
        return Simplicial_Complex([facet for facet in self.generators if set(facet) > set(simplex)], self.quotient_gens)

    # Gives the link of simplex in the complex.
    def link(self, simplex: tuple):
        return Simplicial_Complex(list(set([tuple(set(gen) - set(simplex))
                                            for gen in self.star(simplex).generators])),
                                  self.quotient_gens)

    # Gives the cone of the complex. The default value for
    # the cone vertex is 1 + the largest vertex in the complex.
    def cone(self):
        cone_vertex = 1 + max(max(facet) for facet in self.generators)
        new_generators: list[tuple[int]] = [tuple(list(gen) + [cone_vertex])
                                            for gen in self.generators]
        new_quotient_gens: list[tuple[int]] = [tuple(list(gen) + [cone_vertex])
                                               for gen in self.quotient_gens]
        return Simplicial_Complex(new_generators, new_quotient_gens)


RP2 = Simplicial_Complex([(1, 3, 4), (1, 2, 6), (4, 5, 6), (1, 4, 6),
                         (2, 3, 6), (3, 5, 6), (1, 3, 5), (1, 2, 5), (2, 4, 5), (2, 3, 4)])

rooted_RP2 = Simplicial_Complex(RP2.generators,
                                [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)])


def tutorial() -> str:
    print("\nWe'll consider the 6-pointed triangulation of the real projective plane as a running example:\n\n"

        "RP2 = Simplicial_Complex([(1, 3, 4), (1, 2, 6), (4, 5, 6), (1, 4, 6), (2, 3, 6), "
        "(3, 5, 6), (1, 3, 5), (1, 2, 5), (2, 4, 5), (2, 3, 4)])\n\n"

        "New simplicial complexes can be instantiated as above by calling Simplicial_Complex with a list of faces (positive integer tuples) "
        "from which the complex will be generated. This list of faces doesn't necessarily have to be an antichain with respect to inclusion. "
        "A proper list of facets for the generated complex can be obtained by calling the function\n\n"

        f"RP2.facets() \n-> {RP2.facets()}\n\n"

        "A sorted version of the originally input list can be obtained by calling the attribute\n\n"

        f"RP2.generators \n-> {RP2.generators}\n\n"

        "Since ther runtime of RP2.facets() scales quadratically with the number of generators, it's recommended to simply input only facets as generators "
        "whenever possible and to call RP2.generators, which gets called and stored automatically upon instantiation anyway, instead of using RP2.facets().\n\n"

        "We can also get a slightly more visually accessible representation of our complex by printing it:\n"

        f"print(RP2) \n->\n{RP2}\n"

        "Here are the other attribute and functions that can be called on our simplicial complex. "
        "What each thing does should be clear based on its name and displayed output.\n\n"

        f"RP2.dimension \n-> {RP2.dimension}\n\n"

        f"RP2.faces() \n-> {RP2.faces()}\n\n"

        f"RP2.f_vector() \n-> {RP2.f_vector()}\n\n"
        f"RP2.h_vector() \n-> {RP2.h_vector()}\n\n"

        f"RP2.dfaces(-1 - any_positive_int) \n-> {RP2.dfaces(-2)}\n\n"
        f"RP2.dfaces(-1) \n-> {RP2.dfaces(-1)}\n\n"
        f"RP2.dfaces(0) \n-> {RP2.dfaces(0)}\n\n"
        f"RP2.dfaces(1) \n-> {RP2.dfaces(1)}\n\n"
        f"RP2.dfaces(2) \n-> {RP2.dfaces(2)}\n\n"
        f"RP2.dfaces(RP2.dimension + any_positive_int) \n-> {RP2.dfaces(3)}\n\n"

        f"print(RP2.link(())) \n-> \n{RP2.link(())}\n"
        f"print(RP2.star(())) \n-> \n{RP2.star(())}\n"
        f"print(RP2.link((6,))) \n-> \n{RP2.link((6,))}\n"
        f"print(RP2.star((6,))) \n-> \n{RP2.star((6,))}\n"
        f"print(RP2.link((5, 6))) \n-> \n{RP2.link((5, 6))}\n"
        f"print(RP2.star((5, 6))) \n-> \n{RP2.star((5, 6))}\n"

        f"print(RP2.cone()) \n-> \n{RP2.cone()}\n"

        "We can whip up the boundary matrices and Laplacians with labels on rows and columns.\n\n"

        f"RP2.boundary_matrix(0) \n-> \n{RP2.boundary_matrix(0)}\n\n"
        f"RP2.boundary_matrix(1) \n-> \n{RP2.boundary_matrix(1)}\n\n"
        f"RP2.boundary_matrix(2) \n-> \n{RP2.boundary_matrix(2)}\n\n"

        f"RP2.UD_Laplacian(0) \n-> \n{RP2.UD_Laplacian(0)}\n\n"
        f"RP2.UD_Laplacian(1) \n-> \n{RP2.UD_Laplacian(1)}\n\n"
        f"RP2.UD_Laplacian(2) \n-> \n{RP2.UD_Laplacian(2)}\n\n"

        f"RP2.DU_Laplacian(0) \n-> \n{RP2.DU_Laplacian(0)}\n\n"
        f"RP2.DU_Laplacian(1) \n-> \n{RP2.DU_Laplacian(1)}\n\n"
        f"RP2.DU_Laplacian(2) \n-> \n{RP2.DU_Laplacian(2)}\n\n"

        f"RP2.total_Laplacian(0) \n-> \n{RP2.total_Laplacian(0)}\n\n"
        f"RP2.total_Laplacian(1) \n-> \n{RP2.total_Laplacian(1)}\n\n"
        f"RP2.total_Laplacian(2) \n-> \n{RP2.total_Laplacian(2)}\n\n"

        "We can apply any function available in the numpy.linalg library to these matrices.\n\n"

        "We can also access entries using pairs of simplices.\n\n"

        f"RP2.total_Laplacian(2)[(1, 4, 6), (4, 5, 6)] "
        f"\n-> {RP2.total_Laplacian(2)[(1, 4, 6), (4, 5, 6)]}\n\n"

        "For the purpose of working with 'rooted' complexes, we can also consider pairs of simplicial complexes.\n\n"
        "Let's consider RP2 rooted at the 1-tree generated by the edges (1, 6), (2, 6), (3, 6), (4, 6), (5, 6).\n\n"

        "rooted_RP2 = Simplicial_Complex([RP2.generators, "
        "[(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]])\n\n"

        "All the above functions can be called on relative complexes such as rooted_RP2 as well. \n"

        f"print(rooted_RP2) \n->\n{rooted_RP2}\n"

        f"rooted_RP2.dimension \n-> {rooted_RP2.dimension}\n\n"

        f"rooted_RP2.faces() \n-> {rooted_RP2.faces()}\n\n"

        f"rooted_RP2.f_vector() \n-> {rooted_RP2.f_vector()}\n\n"
        f"rooted_RP2.h_vector() \n-> {rooted_RP2.h_vector()}\n\n"

        f"rooted_RP2.dfaces(-1) \n-> {rooted_RP2.dfaces(-1)}\n\n"
        f"rooted_RP2.dfaces(0) \n-> {rooted_RP2.dfaces(0)}\n\n"
        f"rooted_RP2.dfaces(1) \n-> {rooted_RP2.dfaces(1)}\n\n"
        f"rooted_RP2.dfaces(2) \n-> {rooted_RP2.dfaces(2)}\n\n"

        f"print(rooted_RP2.link(())) \n-> \n{rooted_RP2.link(())}\n"
        f"print(rooted_RP2.star(())) \n-> \n{rooted_RP2.star(())}\n"
        f"print(rooted_RP2.link((6,))) \n-> \n{rooted_RP2.link((6,))}\n"
        f"print(rooted_RP2.star((6,))) \n-> \n{rooted_RP2.star((6,))}\n"
        f"print(rooted_RP2.link((5, 6))) \n-> \n{rooted_RP2.link((5, 6))}\n"
        f"print(rooted_RP2.star((5, 6))) \n-> \n{rooted_RP2.star((5, 6))}\n"

        f"print(rooted_RP2.cone()) \n-> \n{rooted_RP2.cone()}\n"

        f"rooted_RP2.boundary_matrix(0) \n-> \n{rooted_RP2.boundary_matrix(0)}\n\n"
        f"rooted_RP2.boundary_matrix(1) \n-> \n{rooted_RP2.boundary_matrix(1)}\n\n"
        f"rooted_RP2.boundary_matrix(2) \n-> \n{rooted_RP2.boundary_matrix(2)}\n\n"

        "As we would hope, the determinant of the above matrix reflects the size of RP2's torsion subgroup. \n\n"
        f"np.linalg.det(rooted_RP2.boundary_matrix(2)) \n-> \n{np.linalg.det(rooted_RP2.boundary_matrix(2))}\n\n"

        f"rooted_RP2.UD_Laplacian(0) \n-> \n{rooted_RP2.UD_Laplacian(0)}\n\n"
        f"rooted_RP2.UD_Laplacian(1) \n-> \n{rooted_RP2.UD_Laplacian(1)}\n\n"
        f"rooted_RP2.UD_Laplacian(2) \n-> \n{rooted_RP2.UD_Laplacian(2)}\n\n"

        f"rooted_RP2.DU_Laplacian(0) \n-> \n{rooted_RP2.DU_Laplacian(0)}\n\n"
        f"rooted_RP2.DU_Laplacian(1) \n-> \n{rooted_RP2.DU_Laplacian(1)}\n\n"
        f"rooted_RP2.DU_Laplacian(2) \n-> \n{rooted_RP2.DU_Laplacian(2)}\n\n"

        f"rooted_RP2.total_Laplacian(0) \n-> \n{rooted_RP2.total_Laplacian(0)}\n\n"
        f"rooted_RP2.total_Laplacian(1) \n-> \n{rooted_RP2.total_Laplacian(1)}\n\n"
        f"rooted_RP2.total_Laplacian(2) \n-> \n{rooted_RP2.total_Laplacian(2)}\n\n")


def Wilson_alg(graph):
    # graph is expected to be a connected one-dimensional Simplicial_Complex
    vertices = [face[0] for face in graph.dfaces(0)]
    edges = [set(face) for face in graph.dfaces(1)]
    next = {vertex: None for vertex in vertices}
    in_tree = {vertex: False for vertex in vertices}
    root = vertices[random.choice(range(len(vertices)))]
    in_tree[root] = True
    for vertex in vertices:
        head = vertex
        while not in_tree[head]:
            next[head] = random.choice(
                [u for u in vertices if {head, u} in edges])
            head = next[head]
        head = vertex
        while not in_tree[head]:
            in_tree[head] = True
            head = next[head]
    return Simplicial_Complex([tuple(sorted([v, next[v]])) for v in vertices if v != root])

# These two functions just have to do with permutations of general well-ordered sets. Such
# permutations are represented as ordered lists of elements which have a prescribed "correct" order.


def generate_all_permutations_that(elements, condition=(lambda x, y: True), targets=None):
    # generates all permutations recursively according to the provided condition
    # if no condition is entered, no restriction will be put on the permutations
    if targets == None:
        targets = elements[:]
    # this first if statement is triggered on the top level of recursion
    # only to achieve the effect of setting targets=elements by default.
    # if, instead of None or nothing, a list of length len(elements) is
    # passed as a third argument targets, the result will be the set of
    # bijections from elements to targets satisfying the given condition
    if targets != []:
        perms = []
        for i in range(len(targets)):
            if condition(elements[-1], targets[i]):
                new_elements = elements[:-1]
                new_targets = targets[:i] + targets[i+1:]
                smaller_perms = generate_all_permutations_that(
                    new_elements, condition, new_targets)
                perms_to_add = [smaller_perm + [targets[i]]
                                for smaller_perm in smaller_perms]
                # by inductive hypothesis, new_perms contains all permutations
                # of elements that put targets[i] at the end of the reordering
                perms += perms_to_add
        return perms
    return [[]]


def disjoint_cycle_decomp(permutation, identity_perm):
    # Produces a disjoint cycle representation of a permuatation which itself is represented by
    # a list of elements that have an implied total ordering e.g., [2, 1, 4, 3] -> [[1, 2], [3, 4]]
    num_rep = [identity_perm.index(elem) for elem in permutation]
    # the above line reduces the code to the case for permutations of integers
    cycles_list = []
    unprocessed_indices = list(reversed(range(len(num_rep))))
    while unprocessed_indices != []:
        tail = unprocessed_indices.pop()
        next_cycle = [identity_perm[tail]]
        head = num_rep[tail]
        while head != tail:
            unprocessed_indices.remove(head)
            next_cycle.append(identity_perm[head])
            head = num_rep[head]
        cycles_list.append(next_cycle)
    cycles_list.sort(key=len)
    return cycles_list
