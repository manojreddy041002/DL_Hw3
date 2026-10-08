import time, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR = 5, 64, 1e-3

# Dataset: CIFAR-10 restricted to 2 classes (cat=3, dog=5), resized for ResNet18
tf = T.Compose([T.Resize(112), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def two_class(ds, per_class):
    idx, cnt = [], {3:0, 5:0}
    for i, y in enumerate(ds.targets):
        if y in cnt and cnt[y] < per_class:
            idx.append(i); cnt[y] += 1
    return Subset(ds, idx)
def remap(y): return (y == 5).long()          # cat -> 0, dog -> 1

train_ds = two_class(torchvision.datasets.CIFAR10("data", True,  download=True, transform=tf), 1000)
test_ds  = two_class(torchvision.datasets.CIFAR10("data", False, download=True, transform=tf), 500)
train_dl = DataLoader(train_ds, BATCH, shuffle=True)
test_dl  = DataLoader(test_ds, BATCH)

def build(mode):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False       # freeze everything
    m.fc = nn.Linear(m.fc.in_features, 2)                   # new classifier (trainable)
    if mode == "finetune":
        for p in m.layer4.parameters(): p.requires_grad = True   # unfreeze last conv block
    return m.to(device)

def run(mode):
    m = build(mode)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
    loss_fn, losses = nn.CrossEntropyLoss(), []
    t0 = time.time()
    for ep in range(EPOCHS):
        m.train(); tot = 0
        for x, y in train_dl:
            x, y = x.to(device), remap(y).to(device)
            opt.zero_grad(); loss = loss_fn(m(x), y); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        losses.append(tot / len(train_ds))
        print(f"{mode} epoch {ep+1}: loss {losses[-1]:.4f}")
    t = time.time() - t0
    m.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (m(x.to(device)).argmax(1).cpu() == remap(y)).sum().item()
    return n_train, t, correct / len(test_ds), losses

res = {k: run(k) for k in ["frozen", "finetune"]}
print(f"{'Method':22}{'Trainable Params':>18}{'Time (s)':>10}{'Accuracy':>10}")
for k, name in [("frozen","Frozen Feature Extr."),("finetune","Fine-Tuned Network")]:
    n, t, a, _ = res[k]; print(f"{name:22}{n:>18,}{t:>10.1f}{a*100:>9.2f}%")

for k, lab in [("frozen","Frozen"),("finetune","Fine-tuned")]:
    plt.plot(range(1, EPOCHS+1), res[k][3], marker="o", label=lab)
plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.legend(); plt.savefig("loss.png")
