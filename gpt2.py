from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F


##Optimized entire Multi head in a single class and more efficient computing all heads at a time #added flash attention
class CausalSelfAttention(nn.Module):
  def __init__(self,config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
    self.c_proj=nn.Linear(config.n_embd,config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT=1
    self.n_head=config.n_head
    self.n_embd=config.n_embd

  def forward(self,x):
    B,T,C=x.shape
    qkv=self.c_attn(x)#q,k,v === (B,T,n_embd)
    q,k,v=qkv.split(self.n_embd,dim=2) 
    q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,T,nh,hs) ---> (B,nh,T,hs)
    k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,T,nh,hs) ---> (B,nh,T,hs)
    v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,T,nh,hs) ---> (B,nh,T,hs)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True) #flash atention  FlashAttention fuses key operations into fewer, more optimized CUDA kernels.
    out=out.transpose(1,2).contiguous().view(B,T,C)
    out=self.c_proj(out)
    return out



class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc =nn.Linear(config.n_embd , 4 * config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj= nn.Linear(4 * config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
    
    def forward(self,x): # just takes the x and does a MLP c -> 4*c and 4*c -> c just to give it some time to think about the embeddings from self attention
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+self.attn(self.ln_1(x)) # this is about the communication between words ,Reduce
        x=x+self.mlp(self.ln_2(x)) # this is infereing the gathered attention embeddings independently , Map 
        return x  #(B,T,C) C is n_embd   

@dataclass
class GPTConfig:
    block_size:int = 1024  #sequence length 
    vocab_size:int = 50257  # total vocab size 
    n_layer:int = 12  # no of layers
    n_head:int = 12 # no of heads 
    n_embd: int = 768  # embedding size of each word


class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config=config

        self.transformer = nn.ModuleDict(dict(

            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'): # for residual pathway we are giving std as 1/sqrt(2*l)
                std *=(2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
                    
                
    def forward(self,idx,targets=None):
        B,T=idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence length{T},block size is only {self.config.block_size}"
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb=self.transformer.wpe(pos) # (T,n_embed)
        tok_emb=self.transformer.wte(idx) 
        x=pos_emb+tok_emb
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) #B,T,Vocabsize , softmax away from predictions
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1) # (B*T,Vocabsize) y= (B*T)
        return logits,loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type): # weight decay
            # start with all of the candidate parameters (that require grad)
            param_dict = {pn: p for pn, p in self.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            # use_fused = fused_available and device_type == "cuda"
            
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
            return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    
        

#DataLoader------------------------------------------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self,B,T):
        self.B=B
        self.T=T

        with open('input.txt','r') as f:
            text=f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"Total tokens: {len(self.tokens)}")
        print(f"Total batches: {len(self.tokens)//(B*T)}")

        #state
        self.current_position=0
    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position:self.current_position+B*T+1]
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        self.current_position+=B*T
        if self.current_position>=len(self.tokens):
            self.current_position=0
        return x,y
#-----------------------------------------------------------------------------------------------------------------------------------
    
#autodetect device
device='cpu'
if torch.cuda.is_available():
    device='cuda'
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): #mps is for mac 
    device='mps'
print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)



#get a data batch


total_batch_size=32768 # we will do a update once thse many are done
B=8
T=128
assert total_batch_size % (B*T) == 0, "total_batch_size must be divisible by B*T"
grad_accum_steps = total_batch_size // (B*T)
print(f"total_batch_size: {total_batch_size}, grad_accum_steps: {grad_accum_steps}")


train_loader=DataLoaderLite(B=B,T=T) # (B,T)
torch.set_float32_matmul_precision('high') ## can set this to low for faster training means low precision matmul
# model=GPT.from_pretrained('gpt2')
model=GPT(GPTConfig(vocab_size=50304))
model.to(device)
torch.compile(model) # sees the entire code at same time and optimizes it, it will compile entirely before going to interpreter, this makes the input to complete all the operations
#when the data is in chip all the operations are done and a single round trip is required, when used torch compile
# logits ,loss=model(x,y)
# print(loss) # (B,T,Vocab Size)

max_lr=6e-4
min_lr=max_lr/10
warmup_steps=10
max_steps=50
def get_lr(step):
    if step<warmup_steps:
        return max_lr *(step+1)
    if step>max_steps:
        return min_lr
    decay_ratio= (step-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    coeff=0.5 *(1.0+math.cos(math.pi*decay_ratio))
    return min_lr + (max_lr-min_lr)*coeff



import time
# optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=1e-1,eps=1e-8)
optimizer=model.configure_optimizers(weight_decay=1e-1,learning_rate=6e-4,device_type=device) # here we are weight decaying only more than 2 dim parameters
for i in range(10):
    t0=time.time()
    
    optimizer.zero_grad()
    loss_accum=0.0
    for micro_step in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x=x.to(device)
        y=y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits,loss=model(x,y)
        ##her grad accumulation is done by just adding the loss to the loss variable
        #but we need to divide the loss by grad_accum_steps so instead of sum of losses we take the average
        loss=loss/grad_accum_steps ## normalizer made sure all the losses are averaged to grad_accum_steps
        loss_accum+=loss.detach()
        loss.backward()
    
    norm= torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr=get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    optimizer.step()
    torch.cuda.synchronize()
    t1=time.time()
    dt=(t1-t0)*1000
    tokens_processed=train_loader.B*train_loader.T*grad_accum_steps
    tokens_per_second= tokens_processed/dt
    print(f"step {i},  loss:{loss_accum.item()}, lr:{lr:.4f} norm :{norm:.4f} time:{dt:.2f}ms tokens/s:{tokens_per_second:.2f}")
import sys; sys.exit(0)

num_return_sequences=5
max_length=30
model.eval()

import tiktoken
enc=tiktoken.get_encoding('gpt2')
tokens=enc.encode("Hello Iam a Language Model")
tokens=torch.tensor(tokens,dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x=tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1)<max_length:
    logits=model(x)
    logits=logits[:,-1,:]  #(B,Vocab Size)

    probs=F.softmax(logits,dim=-1) # -1 in the sense of last dimension

    topk_probs, topk_indices= torch.topk(probs,50,dim=-1)
    ix=torch.multinomial(topk_probs,1) #(B,1)

    xcol = torch.gather(topk_indices,-1,ix) #(B,1)

    x=torch.cat((x,xcol),dim=1)

for  i in range(num_return_sequences):
    tokens=x[i,:max_length].tolist()
    decoded=enc.decode(tokens)
    print(">",decoded)



