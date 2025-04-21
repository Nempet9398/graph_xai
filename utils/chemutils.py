import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import BRICS, Recap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

MST_MAX_WEIGHT = 0.01

MAX_NCAND = 2000


# map atoms to clique
def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if
               int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    # get the fragment of clique
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol


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
        ### 여기 숫자 3을 싹다 2에서 바꾼거임 - 완화한거
        if len(bonds) > 3 or (len(bonds) == 3 and len(
                cnei) > 3):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        ### 이것도 바꾼거임 2에서 3으로
        elif len(rings) > 3:  # Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            ## 그냥 클리크들기리 최대한 연결
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

    # ===== 중복 제거 =====
    cliques = [sorted(list(set(c))) for c in cliques]  # 원자 중복 제거 및 정렬
    cliques = list(map(list, set(map(tuple, cliques))))  # 완전 중복된 클리크 제거


    return (cliques, edges)


# def tree_decomp_no_overlap(mol):
#     n_atoms = mol.GetNumAtoms()
#     if n_atoms == 1:
#         return [[0]], []

#     cliques = []
#     atom_assigned = [False] * n_atoms  # 원자가 이미 클리크에 들어갔는지 표시

#     ### Step 1: 고리 먼저 처리 ###
#     ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
#     for ring in ssr:
#         cliques.append(ring)
#         for atom in ring:
#             atom_assigned[atom] = True

#     ### Step 2: 비고리 결합 처리 (안 겹치게!) ###
#     for bond in mol.GetBonds():
#         a1 = bond.GetBeginAtom().GetIdx()
#         a2 = bond.GetEndAtom().GetIdx()
#         if not bond.IsInRing():
#             if not atom_assigned[a1] and not atom_assigned[a2]:
#                 cliques.append([a1, a2])
#                 atom_assigned[a1] = True
#                 atom_assigned[a2] = True

#     ### Step 3: 남은 원자들은 싱글턴 ###
#     for atom in range(n_atoms):
#         if not atom_assigned[atom]:
#             cliques.append([atom])

#     ### Step 4: 클리크 간 edge 설정 ###
#     # 원자 하나가 여러 클리크에 걸쳐있지 않기 때문에,
#     # edge는 '공통 원자가 존재'하는 경우만으로 설정

#     edges_dict = defaultdict(int)
#     atom_to_clique = {}

#     for idx, clique in enumerate(cliques):
#         for atom in clique:
#             atom_to_clique[atom] = idx

#     for bond in mol.GetBonds():
#         a1 = bond.GetBeginAtom().GetIdx()
#         a2 = bond.GetEndAtom().GetIdx()
#         c1 = atom_to_clique[a1]
#         c2 = atom_to_clique[a2]
#         if c1 != c2:
#             edges_dict[(min(c1, c2), max(c1, c2))] = MST_MAX_WEIGHT - 1  # 기본 weight

#     edges = [u + (v,) for u, v in edges_dict.items()]

#     # MST 계산
#     if len(edges) > 0:
#         row, col, data = zip(*edges)
#         n_clique = len(cliques)
#         clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
#         junc_tree = minimum_spanning_tree(clique_graph)
#         row, col = junc_tree.nonzero()
#         edges = [(row[i], col[i]) for i in range(len(row))]
#     else:
#         edges = []

#     return cliques, edges

def tree_decomp_no_overlap_clean(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    atom_assigned = [False] * n_atoms

    ### Step 1: 고리 처리 (작은 고리부터, 중복 제거)
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    ssr = sorted(ssr, key=lambda ring: len(ring))  # 작은 고리 우선

    for ring in ssr:
        # 아직 안 할당된 원자만 남기기
        filtered = [atom for atom in ring if not atom_assigned[atom]]
        if len(filtered) >= 2:  # 2개 이상 남아 있어야 클리크 의미 있음
            cliques.append(filtered)
            for atom in filtered:
                atom_assigned[atom] = True

    ### Step 2: 비고리 결합 처리
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            if not atom_assigned[a1] and not atom_assigned[a2]:
                cliques.append([a1, a2])
                atom_assigned[a1] = True
                atom_assigned[a2] = True

    ### Step 3: 남은 애들 싱글턴으로
    for atom in range(n_atoms):
        if not atom_assigned[atom]:
            cliques.append([atom])
            atom_assigned[atom] = True

    ### Step 4: 클리크 간 edge 연결
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

    # MST 계산
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
# MST_MAX_WEIGHT는 에지 가중치 기준값 (적절한 상수로 정의)
MST_MAX_WEIGHT = 100

def tree_decomp_no_overlap(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    atom_assigned = [False] * n_atoms  # 각 원자가 할당되었는지 표시

    ### Step 1: 고리 처리 (작은 고리부터 할당)
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    ssr = sorted(ssr, key=lambda ring: len(ring))  # 작은 고리부터
    for ring in ssr:
        # 만약 이미 대부분의 원자가 할당되어 있다면, 새 고리로 추가할지 선택적으로 결정할 수 있음
        new_ring = [atom for atom in ring if not atom_assigned[atom]]
        if new_ring:  # 새로 할당할 원소가 있다면
            cliques.append(new_ring)
            for atom in new_ring:
                atom_assigned[atom] = True

    ### Step 2: 비고리 결합 처리 (겹치지 않게)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            # 두 원자가 모두 아직 할당되지 않았다면 새로운 클리크로 추가
            if not atom_assigned[a1] and not atom_assigned[a2]:
                cliques.append([a1, a2])
                atom_assigned[a1] = True
                atom_assigned[a2] = True

    ### Step 3: 남은 원자들은 싱글턴 클리크
    for atom in range(n_atoms):
        if not atom_assigned[atom]:
            cliques.append([atom])
            atom_assigned[atom] = True

    ### Step 4: 클리크 간 에지 생성
    # 원자가 한 클리크에만 할당되어 있으므로, 두 클리크가 연결되는 경우는 해당 원자가 
    # 두 클리크를 잇는 결합에 의해 결정됩니다.
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
            # 중복되지 않도록 최소-최대 순서로 키를 생성하고, 기본 가중치 할당
            edges_dict[(min(c1, c2), max(c1, c2))] = MST_MAX_WEIGHT - 1

    edges = [u + (v,) for u, v in edges_dict.items()]

    # MST 계산
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



def absorb_singletons(mol, cliques):
    atom_to_clique = {}
    for idx, c in enumerate(cliques):
        for atom in c:
            atom_to_clique[atom] = idx

    for atom in range(mol.GetNumAtoms()):
        if atom not in atom_to_clique:
            neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(atom).GetNeighbors()]
            neighbor_cliques = set()
            for n in neighbors:
                if n in atom_to_clique:
                    neighbor_cliques.add(atom_to_clique[n])

            if len(neighbor_cliques) > 0:
                # 인접 클리크 하나에 포함시키기
                target_clique = neighbor_cliques.pop()
                cliques[target_clique].append(atom)
            else:
                # 고립된 원자면 그대로 싱글턴
                cliques.append([atom])
    
    return cliques




def recap_decomp(mol):
    recap_tree = Recap.RecapDecompose(mol)
    leaf_frags = list(recap_tree.GetLeaves().keys())  # SMILES 리스트

    # 클리크: 각 fragment의 원자 인덱스 리스트
    cliques = []
    used_atoms = set()

    for frag_smiles in leaf_frags:
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        match = mol.GetSubstructMatch(frag_mol)
        if match:
            atom_ids = list(match)
            cliques.append(atom_ids)
            used_atoms.update(atom_ids)

    # 누락된 원자들 (할당 안된 것들)은 싱글턴으로
    all_atoms = set(range(mol.GetNumAtoms()))
    for atom in all_atoms - used_atoms:
        cliques.append([atom])

    # 간단한 연결 edge: 공통 원자 기반
    edges = []
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            if set(cliques[i]) & set(cliques[j]):
                edges.append((i, j))

    return cliques, edges

def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for node in list(set(range(n_atoms))-set(sum(cliques, []))):
        cliques.append([node])
    return cliques, edges


def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()


# Bond type not considered because all aromatic (so SINGLE matches DOUBLE)
def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])


def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):
    prev_nids = [node.nid for node in prev_nodes]
    for nei_node in prev_nodes + neighbors:
        nei_id, nei_mol = nei_node.nid, nei_node.mol
        amap = nei_amap[nei_id]
        for atom in nei_mol.GetAtoms():
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        if nei_mol.GetNumBonds() == 0:
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtom().GetIdx()]
                a2 = amap[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
                elif nei_id in prev_nids:  # father node overrides
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
    return ctr_mol


def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei.nid: {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom

    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()


# This version records idx mapping between ctr_mol and nei_mol
def enum_attach(ctr_mol, nei_node, amap, singletons):
    nei_mol, nei_idx = nei_node.mol, nei_node.nid
    att_confs = []
    black_list = [atom_idx for nei_id, atom_idx, _ in amap if nei_id in singletons]
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]
    ctr_bonds = [bond for bond in ctr_mol.GetBonds()]

    if nei_mol.GetNumBonds() == 0:  # neighbor singleton
        nei_atom = nei_mol.GetAtomWithIdx(0)
        used_list = [atom_idx for _, atom_idx, _ in amap]
        for atom in ctr_atoms:
            if atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                new_amap = amap + [(nei_idx, atom.GetIdx(), 0)]
                att_confs.append(new_amap)

    elif nei_mol.GetNumBonds() == 1:  # neighbor is a bond
        bond = nei_mol.GetBondWithIdx(0)
        bond_val = int(bond.GetBondTypeAsDouble())
        b1, b2 = bond.GetBeginAtom(), bond.GetEndAtom()

        for atom in ctr_atoms:
            # Optimize if atom is carbon (other atoms may change valence)
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                continue
            if atom_equal(atom, b1):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())]
                att_confs.append(new_amap)
            elif atom_equal(atom, b2):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())]
                att_confs.append(new_amap)
    else:
        # intersection is an atom
        for a1 in ctr_atoms:
            for a2 in nei_mol.GetAtoms():
                if atom_equal(a1, a2):
                    # Optimize if atom is carbon (other atoms may change valence)
                    if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                        continue
                    new_amap = amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())]
                    att_confs.append(new_amap)

        # intersection is an bond
        if ctr_mol.GetNumBonds() > 1:
            for b1 in ctr_bonds:
                for b2 in nei_mol.GetBonds():
                    if ring_bond_equal(b1, b2):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetBeginAtom().GetIdx()),
                                           (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetEndAtom().GetIdx())]
                        att_confs.append(new_amap)

                    if ring_bond_equal(b1, b2, reverse=True):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetEndAtom().GetIdx()),
                                           (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetBeginAtom().GetIdx())]
                        att_confs.append(new_amap)
    return att_confs


# Try rings first: Speed-Up
def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[]):
    all_attach_confs = []
    singletons = [nei_node.nid for nei_node in neighbors + prev_nodes if nei_node.mol.GetNumAtoms() == 1]

    def search(cur_amap, depth):
        if len(all_attach_confs) > MAX_NCAND:
            return
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        nei_node = neighbors[depth]
        cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)
        cand_smiles = set()
        candidates = []
        for amap in cand_amap:
            cand_mol = local_attach(node.mol, neighbors[:depth + 1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue
            smiles = get_smiles(cand_mol)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            candidates.append(amap)

        if len(candidates) == 0:
            return

        for new_amap in candidates:
            search(new_amap, depth + 1)

    search(prev_amap, 0)
    cand_smiles = set()
    candidates = []
    for amap in all_attach_confs:
        cand_mol = local_attach(node.mol, neighbors, prev_nodes, amap)
        cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
        smiles = Chem.MolToSmiles(cand_mol)
        if smiles in cand_smiles:
            continue
        cand_smiles.add(smiles)
        Chem.Kekulize(cand_mol)
        candidates.append((smiles, cand_mol, amap))

    return candidates


# Only used for debugging purpose
def dfs_assemble(cur_mol, global_amap, fa_amap, cur_node, fa_node):
    fa_nid = fa_node.nid if fa_node is not None else -1
    prev_nodes = [fa_node] if fa_node is not None else []

    children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
    neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors

    cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
    cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)

    cand_smiles, cand_amap = zip(*cands)
    label_idx = cand_smiles.index(cur_node.label)
    label_amap = cand_amap[label_idx]

    for nei_id, ctr_atom, nei_atom in label_amap:
        if nei_id == fa_nid:
            continue
        global_amap[nei_id][nei_atom] = global_amap[cur_node.nid][ctr_atom]

    cur_mol = attach_mols(cur_mol, children, [], global_amap)  # father is already attached
    for nei_node in children:
        if not nei_node.is_leaf:
            dfs_assemble(cur_mol, global_amap, label_amap, nei_node, cur_node)




