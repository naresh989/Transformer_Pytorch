import torch
import torch.nn as nn
from torch.nn import functional as F
import os

#hyperparameters
batch_size=4
block_size=8
max_iters=5000
eval_interval=300
learning_rate=1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embed=32
n_head=4
n_layer=3
dropout=0.2


torch.manual_seed(1337)


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

##Attention with one head and adding number of heads in other class
class Head(nn.Module):

  def __init__(self,head_size):
    super().__init__()
    self.key=nn.Linear(n_embed,head_size,bias=False)
    self.query=nn.Linear(n_embed,head_size,bias=False)
    self.value=nn.Linear(n_embed,head_size,bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout=nn.Dropout(dropout)

  def forward(self,x):
    B,T,C=x.shape
    k=self.key(x) #(B,T,C) C=head_size  (B,T,head_size)
    q=self.query(x) #(B,T,C)
    v=self.value(x) #(B,T,C)
    #compute attention scores
    wei= q@k.transpose(-2,-1) * k.shape[-1]**-0.5 #(B,T,C)@(B,C,T) = (B,T,T)
    wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) #(B,T,T)
    wei=F.softmax(wei,dim=-1)
    wei=self.dropout(wei)
    out=wei @ v #(B,T,T)@(B,T,C) = (B,T,C) 
    return out
#Multi headed self attention
class MultiHeadAttention(nn.Module):
  def __init__(self,n_heads,head_size):
    super().__init__()
    self.heads=nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.proj=nn.Linear(n_embed,n_embed)
    self.dropout=nn.Dropout(dropout)
    
  def forward(self,x):
    out=torch.cat([h(x) for h in self.heads],dim=-1) #This becomes (B,T,n_embed)=because n_embed=head_size*n_heads                        (B,T,n_heads,head_size)
    out=self.dropout(self.proj(out))
    return out

##Optimized entire Multi head in a single class and more efficient computing all heads at a time
class CausualSelfAttention(nn.Module):
  def __init__(self,n_heads,head_size):
    super().__init__()
    self.c_attn=nn.Linear(n_embed,3*n_embed)
    self.proj=nn.Linear(n_embed,n_embed)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))
                            .view(1,1,block_size,block_size)) ##we need to reshape to 4d so that gets broadcasted
    self.dropout=nn.Dropout(dropout)
    self.res_dropout=nn.Dropout(dropout)
    self.nh=n_heads
    self.hs=head_size

  def forward(self,x):
    q,k,v=self.c_attn(x).split(n_embed,dim=2) #q,k,v === (B,T,n_embed)
    B,T,C=q.shape
    q=q.view(B,T,self.nh,self.hs).transpose(1,2) # (B,T,nh,hs) ---> (B,nh,T,hs)
    k=k.view(B,T,self.nh,self.hs).transpose(1,2) # (B,T,nh,hs) ---> (B,nh,T,hs)
    v=v.view(B,T,self.nh,self.hs).transpose(1,2) # (B,T,nh,hs) ---> (B,nh,T,hs)

    wei= q@k.transpose(-2,-1) * k.shape[-1]**-0.5 #(B,nh,T,hs)@(B,nh,hs,T) = (B,nh,T,T)
    wei=wei.masked_fill(self.tril[:,:,:T,:T]==0,float('-inf')) #(B,nh,T,T)
    wei=F.softmax(wei,dim=-1)
    wei=self.dropout(wei)
    out=wei @ v #(B,nh,T,T)@(B,nh,T,hs) = (B,nh,T,hs) 
    out=out.transpose(1,2).contiguous().view(B,T,C)
    out=self.res_dropout(out)
    return out



#Feed forward class
class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net=nn.Sequential(
      nn.Linear(n_embed,4*n_embed),
      nn.ReLU(),
      nn.Linear(4*n_embed,n_embed),
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
    return self.net(x)
  
#Block class
class Block(nn.Module):
  def __init__(self,n_embed,n_heads):
    super().__init__()
    head_size=n_embed//n_heads
    self.sa=CausualSelfAttention(n_heads,head_size)
    # self.sa=MultiHeadAttention(n_heads,head_size)
    self.ffwd=FeedForward(n_embed)
    self.ln=nn.LayerNorm(n_embed)  

  def forward(self,x):
    x=x+self.sa(self.ln(x))
    x=x+self.ffwd(self.ln(x))
    return x

#bigram language model
class Gpt(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding_table=nn.Embedding(vocab_size,n_embed) #vocab_size=65
    self.positional_embedding_table=nn.Embedding(block_size,n_embed) #block_size=8
    self.blocks=nn.Sequential(*[Block(n_embed,n_heads=n_head) for _ in range(n_layer)])
    self.ln_f=nn.Linear(n_embed,n_embed)
    # self.sa_heads=MultiHeadAttention(4,n_embed//4) #4 heads, each head has size n_embed/4 # before we have taken size of 32 , now how many heads we want that we divide with n_embed which is 32 , so after concatenating again we get 32 only
    # self.ff=FeedForward(n_embed) #feed forward layer
    self.lm_head=nn.Linear(n_embed,vocab_size)

  def forward(self,idx,targets=None):
    #idx dimension is (B,T)
    B,T=idx.shape
    token_emb=self.embedding_table(idx) #logits become (B,T,C) batch time and channel C is embed_size, n_embed
    pos_emb=self.positional_embedding_table(torch.arange(T,device=device)) #positional embedding (T,C)
    x=token_emb+pos_emb #this is the positional embedding (T,C) is broadcasted along the batch size
    # x=self.sa_heads(x) #(B,T,C) this is the self attention head
    # x=self.ff(x) #(B,T,C) this is the feed forward layer
    x=self.blocks(x)
    x=self.ln_f(x) #final layer norm  
    logits=self.lm_head(x) #logits become (B,T,vocab_size) 
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
      idx_cond=idx[:,-block_size:] #take the last block_size tokens
      logits,loss=self(idx_cond)
      last=logits[:,-1,:] #this gives last row of each example in batch B*C
      probs=F.softmax(last,dim=1) #this gives softmax only along the row B*C

      sample=torch.multinomial(probs,num_samples=1) #(B,1)
      idx=torch.cat((idx,sample),dim=1) #(B,T+1)
    return idx
  




model=Gpt()
m=model.to(device)

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

print(decode(m.generator(context, max_tokens=500)[0].tolist()))



# Single token input: [[0]]
# Token + Positional Embeddings: [[0.5, -1.2, 0.8, ..., 0.3]]
# Multi-Head Attention:
# Compute Q, K, V → Shape (1,1,64)
# Compute attention weights (softmax over T=1) → [[1.0]]
# Compute weighted sum → Output shape (1,1,64) for each head
# Concatenate all 6 heads → Final output (1,1,384)
# Feedforward network: Adds non-linearity and transformations.
# Final Linear Layer: Projects to vocabulary size (1,1,65).
# Softmax + Sampling: Picks the next token.
# Loop continues for max_tokens.