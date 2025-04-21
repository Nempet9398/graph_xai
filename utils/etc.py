
 #%%
    from torch_geometric.utils import to_networkx
    import networkx as nx
    import matplotlib.pyplot as plt

    # Visualize the dataset
    def visualize_data(data):
        G = to_networkx(data, to_undirected=True)
        plt.figure(figsize=(10, 10))
        pos = nx.kamada_kawai_layout(G)
        # pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("Graph Visualization")
        plt.show()

    visualize_data(data)

    #%%

    from rdkit import Chem
    from rdkit.Chem import BRICS, Draw
    import networkx as nx
    import matplotlib.pyplot as plt

    # 아스피린(Aspirin) 분자의 SMILES 표현
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    # 분자 객체 생성
    aspirin_mol = Chem.MolFromSmiles(aspirin_smiles)

    # 원래 분자 시각화
    img_original = Draw.MolToImage(aspirin_mol, size=(200, 200))
    from IPython.display import display
    from torch_geometric.nn import GCNConv, global_mean_pool
    import random

    display(img_original)

    # BRICS를 사용하여 서브구조 분할
    fragments = list(BRICS.BRICSDecompose(aspirin_mol))

    # 서브구조 출력
    print("Aspirin Motifs (BRICS fragments):")
    for idx, frag in enumerate(fragments):
        print(f"{idx+1}: {frag}")

    # 서브구조 시각화
    aspirin_frags = [Chem.MolFromSmiles(f) for f in fragments]
    img_fragments = Draw.MolsToGridImage(aspirin_frags, molsPerRow=3, subImgSize=(200, 200))
    display(img_fragments)
