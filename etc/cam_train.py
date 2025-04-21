import os
import argparse
import importlib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from graph.utils.utils import set_seed  # 시드 고정 함수

# dataset, model, train, inference 모듈은 이미 작성된 것으로 가정합니다.
dataset_module = importlib.import_module('dataset')
importlib.reload(dataset_module)
model_module = importlib.import_module('model')
importlib.reload(model_module)
train_module = importlib.import_module('train')
importlib.reload(train_module)
inference = importlib.import_module('inference')
importlib.reload(inference)

# argparse를 통한 하이퍼파라미터 설정
parser = argparse.ArgumentParser(description='PyTorch GCN + Grad-CAM on MUTAG')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN, GAT 등")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MUTAG', help='데이터셋 이름 (현재는 MUTAG만 지원)')
parser.add_argument('--eval_metric', type=str, default='auc')
parser.add_argument('--target_col', type=int, default=14)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

# -------------------------
# 학습 및 Grad-CAM 적용 함수
# -------------------------
def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        # 분류 문제로 cross entropy loss 사용 (batch.y는 label)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def grad_cam(model, sample, target_class=None):
    """
    model의 마지막 합성곱 계층에 대해 Grad-CAM을 계산합니다.
    sample: 하나의 배치 (여러 그래프일 수 있음)
    target_class: 만약 None이면, 모델의 예측 class를 사용합니다.
    """
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        # grad_output는 튜플로 나오므로 첫번째를 사용합니다.
        gradients.append(grad_output[0])
    
    # 모델이 GCN 모듈 내에 conv_layers 리스트를 갖고 있다고 가정합니다.
    # 마지막 GCNConv layer에 hook을 등록합니다.
    hook_forward = model.conv_layers[-1].register_forward_hook(forward_hook)
    hook_backward = model.conv_layers[-1].register_backward_hook(backward_hook)
    
    model.eval()
    sample = sample.to(next(model.parameters()).device)
    output = model(sample)
    
    if target_class is None:
        # 각 그래프마다 예측된 class 사용
        target_class = output.argmax(dim=1)
    
    one_hot = torch.zeros_like(output)
    one_hot.scatter_(1, target_class.unsqueeze(1), 1)
    
    model.zero_grad()
    output.backward(gradient=one_hot, retain_graph=True)
    
    # 저장된 활성화와 기울기를 이용해 Grad-CAM 가중치 계산
    # (기울기를 채널별로 global average pooling)
    grad = gradients[0]  # shape: [num_nodes, channels]
    activation = activations[0]  # shape: [num_nodes, channels]
    weights = grad.mean(dim=0, keepdim=True)  # 각 채널에 대한 평균 기울기, shape: [1, channels]
    
    # 채널별 가중합을 수행하여 각 노드에 대한 중요도 계산, ReLU 적용
    cam = torch.relu((activation * weights).sum(dim=1))  # shape: [num_nodes]
    
    # hook 제거
    hook_forward.remove()
    hook_backward.remove()
    
    return cam

def main(args):
    set_seed(args.seed)
    
    # MUTAG 데이터셋 로드
    if args.dataset.upper() == 'MUTAG':
        dataset, split_idx = dataset_module.get_MUTAG()
    
    # train / test split 생성
    train_dataset = [dataset[i] for i in split_idx['train']]
    test_dataset = [dataset[i] for i in split_idx['test']]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 초기화: model_module에 정의된 GCN 모델을 사용 (입력 피처 수, hidden 크기, 클래스 수 필요)
    model = model_module.GCN(args.layers, dataset.num_features, args.hidden, dataset.num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 학습 루프
    for epoch in range(args.epochs):
        loss = train(model, optimizer, train_loader, args.device)
        print(f'Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}')
    
    # Grad-CAM 적용: test 데이터셋에서 첫 번째 배치의 첫번째 그래프 선택
    sample_batch = next(iter(test_loader))
    # 여러 그래프가 배치로 들어올 수 있으므로, 여기서는 첫번째 그래프만 시각화
    cam = grad_cam(model, sample_batch)
    
    # sample_batch의 그래프 정보를 networkx로 변환
    # (batch의 경우, to_networkx는 하나의 그래프를 표현하므로 주의)
    nx_graph = to_networkx(sample_batch, to_undirected=True)
    
    # 첫번째 그래프에 해당하는 노드의 Grad-CAM 값 추출
    # 여기서는 배치 내 첫번째 그래프의 노드 인덱스에 해당하는 cam 값을 사용합니다.
    cam_values = cam.detach().cpu().numpy()
    
    # NetworkX를 이용해 그래프 시각화 (노드 색은 Grad-CAM 중요도에 따라 결정)
    pos = nx.spring_layout(nx_graph)
    plt.figure(figsize=(8, 6))
    nx.draw(nx_graph, pos, node_color=cam_values, with_labels=True, cmap=plt.cm.jet)
    plt.title("Grad-CAM Visualization on MUTAG Sample")
    plt.show()

if __name__ == '__main__':
    main(args)
