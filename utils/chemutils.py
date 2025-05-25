import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import BRICS, Recap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

MST_MAX_WEIGHT = 0.01

MAX_NCAND = 2000



def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    # get rings
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]

        if len(bonds) > 3 or (len(bonds) == 3 and len(
                cnei) > 3):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 3:  # Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    if len(edges) == 0:
        for node in list(set(range(n_atoms))-set(sum(cliques, []))):
            cliques.append([node])
        return cliques, edges

    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]
    for node in list(set(range(n_atoms)) - set(sum(cliques, []))):
        cliques.append([node])

    cliques = [sorted(list(set(c))) for c in cliques] 
    cliques = list(map(list, set(map(tuple, cliques))))  

    return (cliques, edges)



def tree_decomp_no_overlap_clean(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []
    cliques = []
    atom_assigned = [False] * n_atoms

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    ssr = sorted(ssr, key=lambda ring: len(ring)) 

    for ring in ssr:

        filtered = [atom for atom in ring if not atom_assigned[atom]]
        if len(filtered) >= 1:  
            cliques.append(filtered)
            for atom in filtered:
                atom_assigned[atom] = True


    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            if not atom_assigned[a1] and not atom_assigned[a2]:
                cliques.append([a1, a2])
                atom_assigned[a1] = True
                atom_assigned[a2] = True


    for atom in range(n_atoms):
        if not atom_assigned[atom]:
            cliques.append([atom])
            atom_assigned[atom] = True


    edges_dict = defaultdict(int)
    atom_to_clique = {}
    for idx, clique in enumerate(cliques):
        for atom in clique:
            atom_to_clique[atom] = idx

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        c1 = atom_to_clique[a1]
        c2 = atom_to_clique[a2]
        if c1 != c2:
            edges_dict[(min(c1, c2), max(c1, c2))] = MST_MAX_WEIGHT - 1

    edges = [u + (v,) for u, v in edges_dict.items()]


    if edges:
        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        edges = [(row[i], col[i]) for i in range(len(row))]
    else:
        edges = []

    return cliques, edges


MST_MAX_WEIGHT = 100
