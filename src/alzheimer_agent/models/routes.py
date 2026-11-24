import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
 def __init__(self,c):
  super().__init__();self.net=nn.Sequential(nn.Conv3d(c,c,5,padding=2,groups=c),nn.GroupNorm(8,c),nn.Conv3d(c,3*c,1),nn.GELU(),nn.Conv3d(3*c,c,1))
  self.scale=nn.Parameter(torch.ones(c)*1e-3)
 def forward(self,x):return x+self.net(x)*self.scale[None,:,None,None,None]

class BidirectionalMamba(nn.Module):
 def __init__(self,dim):
  super().__init__()
  from mamba_ssm import Mamba
  self.norm=nn.LayerNorm(dim);self.ssm=Mamba(d_model=dim,d_state=16,d_conv=4,expand=2,use_fast_path=False)
  self.scale=nn.Parameter(torch.ones(dim)*1e-2)
 def forward(self,x):
  z=self.norm(x)
  a=self.ssm(z);b=self.ssm(z.flip(1)).flip(1)
  return x+.5*(a+b)*self.scale

class ImageEncoder(nn.Module):
 def __init__(self,kind='cnn',width=24,dim=192):
  super().__init__();channels=[24,48,96,192];self.stages=nn.ModuleList();self.projections=nn.ModuleList();c=1
  for i,co in enumerate(channels):
   self.stages.append(nn.Sequential(nn.Conv3d(c,co,2,stride=2),nn.GroupNorm(8,co),nn.GELU(),ConvBlock(co)));c=co
   if i>0:self.projections.append(nn.Conv3d(co,dim,1))
  self.pos=nn.Parameter(torch.randn(1,81,dim)*.01)
  self.global_net=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,6,dim*3,dropout=.2,activation='gelu',batch_first=True,norm_first=True),2,enable_nested_tensor=False) if kind=='transformer' else (nn.Sequential(BidirectionalMamba(dim),BidirectionalMamba(dim)) if kind=='mamba' else nn.Identity())
  self.norm=nn.LayerNorm(dim)
 def forward(self,x):
  tokens=[]
  for i,stage in enumerate(self.stages):
   x=stage(x)
   if i>0:tokens.append(self.projections[i-1](F.adaptive_avg_pool3d(x,3)).flatten(2).transpose(1,2))
  z=torch.cat(tokens,1)+self.pos
  return self.norm(self.global_net(z))

class RouteNet(nn.Module):
 def __init__(self,route,kind='cnn',dim=192):
  super().__init__();assert route in ['MRI_ONLY','MRI_PET','MRI_TABLE','FULL'];self.route=route
  self.mri=ImageEncoder(kind,dim=dim);self.attention_pool=nn.Linear(dim,1)
  if route in ['MRI_PET','FULL']:
   self.reliability=nn.Linear(dim*2,2);self.fusion_scale=nn.Parameter(torch.tensor(.1));self.pet=ImageEncoder(kind,dim=dim);self.cross_mp=nn.MultiheadAttention(dim,6,dropout=.1,batch_first=True);self.cross_pm=nn.MultiheadAttention(dim,6,dropout=.1,batch_first=True)
   self.fuse=nn.Sequential(nn.Linear(dim*4,dim),nn.LayerNorm(dim),nn.GELU(),nn.Dropout(.25));self.aux_pet=nn.Linear(dim,3)
  if route in ['MRI_TABLE','FULL']:
   self.table=nn.Sequential(nn.Linear(10,dim),nn.LayerNorm(dim),nn.GELU(),nn.Dropout(.2),nn.Linear(dim,dim),nn.GELU())
   self.film=nn.Linear(dim,dim*2);nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
   setattr(self,'table_fuse' if route=='FULL' else 'fuse',nn.Sequential(nn.Linear(dim*2,dim),nn.LayerNorm(dim),nn.GELU(),nn.Dropout(.25)));self.aux_table=nn.Linear(dim,3)
  self.aux_mri=nn.Linear(dim,3);self.head=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(.3),nn.Linear(dim,3))
 def pool_tokens(self,x):return (self.attention_pool(x).softmax(1)*x).sum(1)
 def forward(self,mri,pet,table):
  mt=self.mri(mri);mv=self.pool_tokens(mt);aux={'mri':self.aux_mri(mv)}
  if self.route in ['MRI_PET','FULL']:
   pt=self.pet(pet);pv=self.pool_tokens(pt)
   mz=mt+self.cross_mp(mt,pt,pt,need_weights=False)[0];pz=pt+self.cross_pm(pt,mt,mt,need_weights=False)[0]
   z=self.fuse(torch.cat([self.pool_tokens(mz),self.pool_tokens(pz),torch.abs(mv-pv),mv*pv],-1));aux['pet']=self.aux_pet(pv)
   weights=self.reliability(torch.cat([mv,pv],-1)).softmax(-1);z=weights[:,0:1]*mv+weights[:,1:2]*pv+self.fusion_scale.tanh()*z
  if self.route in ['MRI_TABLE','FULL']:
   tv=self.table(table);scale,shift=self.film(tv).chunk(2,-1)
   base=mt if self.route=='MRI_TABLE' else z[:,None]
   conditioned=base*(1+.1*torch.tanh(scale[:,None]))+.1*shift[:,None]
   z=getattr(self,'table_fuse' if self.route=='FULL' else 'fuse')(torch.cat([self.pool_tokens(conditioned),tv],-1));aux['table']=self.aux_table(tv)
  if self.route=='MRI_ONLY':z=mv
  return self.head(z),aux
