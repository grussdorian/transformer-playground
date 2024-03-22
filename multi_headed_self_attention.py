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
batch_size = 64
block_size = 256

max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

# %%
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# %% [markdown]
# ### Character level language model
# ### Encode characters to numbers

# %%
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# take a string output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])

# %%
# print(encode("hii there"))
# print(decode(encode("hii there")))
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

# %%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# print(len(train_data), len(val_data))

# %%
# block_size = 8
# train_data[: block_size+1]
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"When input is {context}, output is {target}")

# %%
torch.manual_seed(1337)

def get_batches(split):
    # generate a small batch of data of inputs and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(mps_device), y.to(mps_device)
    return x, y



xb, yb = get_batches('train')
# print("inputs:")
# print(xb.shape)
# print(xb)
# print("targets:")
# print(yb.shape)
# print(yb)

# print("---")

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, : t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()}, the target is: {target}")

# %%
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# %%
class Head(nn.Module):
  """ One head of self-attention"""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) #(B,T,C)
    q = self.query(x) #(B,T,C)
    # compute self-attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * C**-0.5 # shouldn't it be head_size instead of C?
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1) # (B,T,T)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out

# %%
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

# %%
class FeedForward(nn.Module):
  """ a simple linear layer followed by a noon-linearity """

  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embed, 4*n_embed),
        nn.ReLU(),
        nn.Linear(4*n_embed, n_embed),
        nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.net(x)

# %%
class Block(nn.Module):
  """ Transformer block: communication followed by computation """
  def __init__(self, n_embed, n_head):
    # n_embed: embedding dimention, n_head: the number of heads we'd like
    super().__init__() 
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  

# %%
class LayerNorm1d:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

    def __call__(self, x):
      # calculate the forward pass
      xmean = x.mean(1, keepdim=True) # batch mean
      xvar = x.var(1, keepdim=True) # batch variance
      xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
      self.out = self.gamma * xhat + self.beta
      return self.out
    
    def parameters(self):
      return [self.gamma, self.beta]

# %%
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # number of embedding dimensions n_embed
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        #self.sa_heads = MultiHeadAttention(4, n_embed//4) # i.e 4 heads of 8-dimentional self-attention
        #self.ffwd = FeedForward(n_embed)
        # self.blocks = nn.Sequential(
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     nn.LayerNorm(n_embed),
        # )
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)  # Language modeling head

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (batch, time, channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=mps_device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) , broadcasting along the Batch dimension, for pos_emb, a new batch dim is added
        #x = self.sa_heads(x)
        #x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size) 

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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            print(decode(idx_next.tolist()[0]), end='')
            idx = torch.cat((idx, idx_next), dim=1)
        print("\n-----\n")
        return idx
    
    
    def inference(self, n_tokens=10, device='cpu'):
      context = torch.zeros((1, 1), dtype=torch.long, device=device)
      print("\n-----\n")
      print(decode(self.generate(idx=context, max_new_tokens=n_tokens)[0].tolist()))

# %%
model = BigramLanguageModel()
m = model.to(mps_device)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
context = torch.zeros((1, 1), dtype=torch.long, device=mps_device)
# print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

# %%
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# %%
def train():
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
def save_model():
  torch.save(m.state_dict(), './saved_models/multiheaded_attention.pth')

def load_model():
  m.load_state_dict(torch.load('./saved_models/multiheaded_attention.pth'))

def inference(model=None, n_tokens=50):
    context = torch.zeros((1, 1), dtype=torch.long, device=mps_device)
    print(decode(model.generate(idx=context, max_new_tokens=n_tokens)[0].tolist()))

# %%
if __name__ == '__main__':
    # train()
    # m.inference()
    # save_model()

    load_model()
    model2 = BigramLanguageModel()
    model2.load_state_dict(torch.load('./saved_models/multiheaded_attention.pth'))
    inference(model2)



print(f'Running inference')
# %%
