# %%
import torch
from torch.nn import functional as F
import torch.nn as nn
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
    print("Running on MPS Device")
else:
    print("MPS device not found")

# %%
max_iters = 10000
eval_iters = 200
eval_interval = 300
learning_rate = 1e-2
n_embed = 32

# %%
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# %%
print("length of text file = ", len(text))

# %%
print(text[:1000])

# %%
chars = sorted(list(set(text)))

# %%
vocab_size = len(chars)

# %%
print(''.join(chars))
print(vocab_size)

# %%
# Character level language model
# Encode characters to numbers

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# take a string output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


# %%
print(encode("hii there"))

# %%
print(decode(encode("hii there")))

# %%
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# %%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(len(train_data), len(val_data))

# %%
block_size = 8
train_data[: block_size+1]


# %%
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context}, output is {target}")

# %%
print(x, y)

# %%
torch.randint(10, (5,))

# %%
torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batches(split):
    # generate a small batch of data of inputs and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(mps_device), y.to(mps_device)
    return x, y


xb, yb = get_batches('train')
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

print("---")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()}, the target is: {target}")

# %%


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


# %%
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # number of embedding dimensions n_embed
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)  # Language modeling head

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)  # (batch, time, channel)
        logits = self.lm_head(tok_emb)  # (B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T)  array of indices in the current context

        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(mps_device)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
context = torch.zeros((1, 1), dtype=torch.long, device=mps_device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

# %%
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# %%
# batch_size = 32
# for steps in range(10000):


# print(loss.item())

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")

    xb, yb = get_batches('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
context = torch.zeros((1, 1), dtype=torch.long, device=mps_device)
print(decode(m.generate(idx=context, max_new_tokens=500)[0].tolist()))

# %%
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape

# %% [markdown]
# ### We want $x[b,t] = mean_{(i≤t)}\ x[b,i]$

# %%

xbow = torch.zeros((B, T, C))  # bow = bag of words
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]  # (t,C)
        xbow[b, t] = torch.mean(xprev, 0)

# %% [markdown]
# ### Same thing using Matrix multiplication

# %%
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
# a = torch.ones(3,3)
a = a / torch.sum(a, dim=1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print("a = ")
print(a)
print("--")
print("b = ")
print(b)
print("--")
print("c = ")
print(c)

# %%
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (B,T,T)  @ (B,T,C) --> (B,T,C) pytorch adds this B dimension
torch.allclose(xbow, xbow2)  # same thing


# %% [markdown]
# #### Remember softmax: $σ(z)_i= \frac{e^{z_i}}{Σ^{K}_{j=1}e^{z_j}}$ for $i=1,...,K$ and $z=(z_1,...,z_K)\in\R^{K}$

# %%
tril = torch.tril((torch.ones(T, T)))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

# %%
