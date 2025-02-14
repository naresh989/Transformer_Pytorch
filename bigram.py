import torch
import torch.nn as nn
from torch.nn import functional as F
import os

#hyperparameters
batch_size=32
block_size=8
max_iters=1000
eval_interval=300
learning_rate=1e-2
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200


# torch.manual_seed(1337)


with open('input.txt','r',encoding='utf-8') as f:
  text=f.read()

chars=''.join(sorted(list(set(text))))
vocab_size=len(chars)
# Create a mapping from characters to integers and vice versa
stoi={s:i for i,s in enumerate(chars)}
itos={i:s for i,s in enumerate(chars)}
encode=lambda s:[stoi[i] for i in s]
decode=lambda s:''.join([itos[i] for i in s])


data=torch.tensor(encode(text),dtype=torch.long)

#split into train and test 90% train and 10% test
n=int(0.9*len(text))
train_data=data[:n]
val_data=data[n:]


#data loading
def batch(split):
  data=train_data if split=='train' else val_data
  ix=torch.randint(len(data)-block_size,(batch_size,))
  x=torch.stack([data[i:i+block_size] for i in ix])
  y=torch.stack([data[i+1:block_size+i+1] for i in ix])
  x=x.to(device)
  y=y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
  out={'train':0,'val':0}
  model.eval()
  for split in ['train','val']:
    for _ in range(eval_iters):
      x,y=batch(split)
      logits,loss=model(x,y)
      out[split]+=loss.item()
  out['train']/=eval_iters
  out['val']/=eval_iters
  model.train()
  return out

#bigram language model
class BigramLanguageModel(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.embedding_table=nn.Embedding(vocab_size,vocab_size) #vocab_size=65

  def forward(self,idx,targets=None):
    #idx dimension is (B,T)
    logits=self.embedding_table(idx) #logits become (B,T,C) batch time and channel
    if targets==None:
      loss=None
    else:
      B,T,C=logits.shape
      logits=logits.view(B*T,C) # simplifying into a smaller dimension shape as per cross entropy docs, or should convert to (B,C,T)
      targets=targets.view(B*T)
      loss=F.cross_entropy(logits,targets)
    return logits,loss

  def generator(self,idx,max_tokens):
    #idx=(B,T)
    for _ in range(max_tokens):
      logits,loss=self(idx)
      last=logits[:,-1,:] #this gives last row of each example in batch B*C
      probs=F.softmax(last,dim=1) #this gives softmax only along the row B*C

      sample=torch.multinomial(probs,num_samples=1) #(B,1)
      idx=torch.cat((idx,sample),dim=1) #(B,T+1)
    return idx




model=BigramLanguageModel(vocab_size)
m=model.to(device)
# m.generator(torch.zeros((1,1),dtype=torch.long),100)[0]


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

  if iter % eval_interval == 0:
    losses=estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb,yb=batch('train')
  logits,loss=model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context= torch.zeros((1, 1), dtype=torch.long,device=device)

# print(decode(m.generator(context, max_tokens=500)[0].tolist()))

