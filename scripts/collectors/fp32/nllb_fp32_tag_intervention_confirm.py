# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.
from IPython import get_ipython

# %% original code cell 0; notebook cell 1
# Colab setup. Reconnect after package installation if Colab requests it.
get_ipython().run_line_magic('pip', "-q install 'transformers==4.51.3' 'tokenizers==0.21.1' 'huggingface_hub==0.30.2' 'datasets==4.0.0' 'sentencepiece==0.2.2' 'sacrebleu==2.3.1' 'indictranstoolkit==1.1.1' 'scipy'")
import os, sys, subprocess
from pathlib import Path
# Match the original IndicTrans2 collector's pinned normalization stack.
# The old collector set INDIC_RESOURCES_PATH before importing IndicProcessor.
dependency_root=Path('/content/fp32_tag_intervention_dependencies')
def pinned_checkout(name,url,revision):
    path=dependency_root/name
    if not (path/'.git').is_dir():
        path.mkdir(parents=True,exist_ok=True)
        subprocess.check_call(['git','init',str(path)],stdout=subprocess.DEVNULL)
        subprocess.check_call(['git','-C',str(path),'remote','add','origin',url])
        subprocess.check_call(['git','-C',str(path),'fetch','--depth','1','origin',revision])
        subprocess.check_call(['git','-C',str(path),'checkout','--detach',revision])
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    assert actual==revision,(name,actual,revision)
    return path
indic_nlp_library=pinned_checkout('indic_nlp_library',
    'https://github.com/anoopkunchukuttan/indic_nlp_library.git',
    '4cead0ae6c78fe9a19a51ef679f586206df9c476')
indic_nlp_resources=pinned_checkout('indic_nlp_resources',
    'https://github.com/anoopkunchukuttan/indic_nlp_resources.git',
    '5b4ff17c080db4bc4cfc068deaa794aa87562eed')
os.environ['INDIC_RESOURCES_PATH']=str(indic_nlp_resources)
subprocess.check_call([sys.executable,'-m','pip','install','-q','sacremoses',str(indic_nlp_library)])
print('Pinned Indic NLP library and resources match the original collector.')
fasttext_package = 'fasttext-numpy2' if sys.version_info >= (3,13) else 'fasttext-numpy2-wheel'
subprocess.check_call([sys.executable,'-m','pip','install','-q',fasttext_package])
from google.colab import drive
drive.mount('/content/drive')


# %% original code cell 1; notebook cell 2
ROLE='confirm'


# %% original code cell 2; notebook cell 3
import gc, hashlib, json, math, os, time, unicodedata, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from datasets import load_dataset
from huggingface_hub import model_info, get_token, HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

assert torch.cuda.is_available(), 'An NVIDIA GPU is required for fp32 collection.'
assert ROLE in {'pilot','holdout','confirm'}
SEED=240901
SPLIT_SEED=240902
BOOT_SEED=240903
TARGETS=['pan_Guru','mal_Mlym','tam_Taml','guj_Gujr','hin_Deva','ben_Beng']
SOURCES=['kan_Knda','tel_Telu','ory_Orya','mar_Deva']
LANGS=sorted(set(TARGETS+SOURCES+['asm_Beng']))
CONTROL='urd_Arab'
DATASET='openlanguagedata/flores_plus'
DATASET_REV='5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06'
EXPECTED_SAMPLE_HASH='64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c'
MODEL_SPEC={
 'it2':('ai4bharat/indictrans2-indic-indic-dist-320M','ffb7582b6d43791f1fb26b2153fc065f2e9ea575'),
 'nllb':('facebook/nllb-200-distilled-600M','f8d333a098d19b4fd9a8b18f94170487ad3f821d')}
OLD_BETA={
 'it2':{'pan_Guru':-.033612,'mal_Mlym':-.019487,'tam_Taml':-.002669,'guj_Gujr':.011901,'hin_Deva':.013689,'ben_Beng':.025658},
 'nllb':{'mal_Mlym':-.016568,'tam_Taml':-.009276,'pan_Guru':-.006260,'guj_Gujr':.010203,'ben_Beng':.011996,'hin_Deva':.025458}}
MAX_LENGTH=256
BATCH_SIZE=int(os.environ.get('INTERVENTION_BATCH_SIZE','4'))
N_BOOT=int(os.environ.get('INTERVENTION_N_BOOT','10000'))
N_GENERATION_MEANINGS=int(os.environ.get('INTERVENTION_GENERATION_MEANINGS','150'))
assert N_GENERATION_MEANINGS in {100,150},'Choose 100 or 150 generation meanings before either full run.'
ROOT=Path(os.environ.get('INTERVENTION_ROOT','/content/drive/MyDrive/fp32_tag_intervention'))
ROOT.mkdir(parents=True,exist_ok=True)
OLD_ROOTS={
 'it2':Path(os.environ.get('OLD_IT2_ROOT','/content/drive/MyDrive/indic_40_backups/indic_40_balanced_holdout_750_rows250_999_locked_v4')),
 'nllb':Path(os.environ.get('OLD_NLLB_ROOT','/content/drive/MyDrive/indic_40_backups/nllb_40_confirmatory_holdout_750_final_v6'))}
PAIRS=[('asm_Beng','ben_Beng'),('hin_Deva','ory_Orya'),('mar_Deva','guj_Gujr'),('guj_Gujr','pan_Guru'),('ben_Beng','mar_Deva'),('kan_Knda','tam_Taml'),('tam_Taml','mal_Mlym'),('tam_Taml','tel_Telu'),('tel_Telu','mal_Mlym'),('kan_Knda','mal_Mlym'),('ben_Beng','kan_Knda'),('hin_Deva','kan_Knda'),('hin_Deva','mal_Mlym'),('hin_Deva','tam_Taml'),('ory_Orya','tel_Telu'),('asm_Beng','kan_Knda'),('ben_Beng','tam_Taml'),('guj_Gujr','mal_Mlym'),('mar_Deva','kan_Knda'),('pan_Guru','tam_Taml')]
DIRECTIONS=PAIRS+[(b,a) for a,b in PAIRS]
assert len(DIRECTIONS)==40 and len(set(DIRECTIONS))==40
assert len({x.split('_')[1] for x in TARGETS})==6

def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)
def atomic_csv(path,df):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');df.to_csv(temp,index=False);temp.replace(path)
def digest(s):return hashlib.sha256(str(s).encode('utf-8')).hexdigest()
def progress(msg):print(time.strftime('%H:%M:%S'),msg,flush=True)
def resolve_hf_token():
 token=os.environ.get('HF_TOKEN')
 try:
  from google.colab import userdata
  token=userdata.get('HF_TOKEN') or token
 except Exception:
  pass
 token=token or get_token()
 if not token:
  raise RuntimeError('FLORES+ is gated. Add HF_TOKEN in Colab Secrets, grant this notebook access, then rerun from the imports cell. Use the Hugging Face account approved for openlanguagedata/flores_plus.')
 try:
  account=HfApi(token=token).whoami(token=token).get('name','unknown')
 except Exception as exc:
  raise RuntimeError('The Hugging Face token could not be authenticated. Check the HF_TOKEN Colab secret and notebook access. The token is not printed.') from exc
 progress(f'Hugging Face authenticated as {account}; token value hidden')
 return token,account
HF_TOKEN,HF_ACCOUNT=resolve_hf_token()
def punctuation(piece):
 s=str(piece).replace('▁','').replace('Ġ','').strip()
 return not s or all(unicodedata.category(ch)[0] in {'P','S','Z','C'} for ch in s)
def active(mask):return [i for i,v in enumerate(mask) if int(v)==1]


# %% original code cell 3; notebook cell 4
def load_meanings():
 try:
  ds=load_dataset(DATASET,split='devtest',revision=DATASET_REV,token=HF_TOKEN)
 except Exception as exc:
  if any(word in str(exc).lower() for word in ['gated','403','401','access denied','unauthorized']):
   raise RuntimeError(f'Authenticated as {HF_ACCOUNT}, but this token cannot read {DATASET}. Check that this Hugging Face account has accepted the dataset gate and that Colab uses its HF_TOKEN secret.') from exc
  raise
 long=ds.to_pandas()
 long['lang_code']=long.iso_639_3.astype(str)+'_'+long.iso_15924.astype(str)
 required_langs=LANGS+[CONTROL]
 sub=long[long.lang_code.isin(required_langs)]
 assert not sub.duplicated(['id','lang_code']).any()
 wide=sub.pivot(index='id',columns='lang_code',values='text').reset_index().sort_values('id').reset_index(drop=True)
 assert len(wide)>=1000 and set(required_langs)<=set(wide.columns)
 wide=wide.iloc[250:1000].reset_index(drop=True)
 assert len(wide)==750 and not wide[required_langs].isna().any().any()
 wide.insert(0,'sentence_id',np.arange(750))
 # Reconstruct the original 40-direction sample manifest byte-for-byte.
 original_directions=[]
 for a,b in PAIRS[:10]:original_directions.extend([(a,b),(b,a)])
 original_directions.extend(PAIRS[10:])
 original_directions.extend([(b,a) for a,b in PAIRS[10:]])
 identity=[{'pair':f'{s}__{t}','sentence_id':int(i),'corpus_sentence_id':str(row.id),
            'source_sha256':digest(row[s]),'target_sha256':digest(row[t]),'in_free_subset':1}
           for s,t in original_directions for i,row in wide.iterrows()]
 sample_csv=pd.DataFrame(identity).to_csv(index=False)
 sample_hash=hashlib.sha256(sample_csv.encode('utf-8')).hexdigest()
 assert sample_hash==EXPECTED_SAMPLE_HASH,('Original holdout sample identity differs',sample_hash)
 manifest=ROOT/'sample_identity.csv'
 if manifest.exists():assert manifest.read_text(encoding='utf-8')==sample_csv
 else:manifest.write_text(sample_csv,encoding='utf-8',newline='')
 progress(f'original sample identity verified: {sample_hash}')
 # Verify exact original corpus rows against the old collector manifest, if mounted.
 for kind,old_root in OLD_ROOTS.items():
  p=old_root/'manifests'/'sample_identity.csv'
  if not p.exists():
   progress(f'{kind} original sample manifest unavailable at {p}; pinned corpus revision and row positions still enforced')
   continue
  old=pd.read_csv(p)
  for row in old.itertuples(index=False):
   src,tgt=str(row.pair).split('__')
   k=int(row.sentence_id)
   assert str(wide.loc[k,'id'])==str(row.corpus_sentence_id)
   assert digest(wide.loc[k,src])==row.source_sha256
   assert digest(wide.loc[k,tgt])==row.target_sha256
  progress(f'{kind} original 40-direction sample manifest verified: {len(old)} rows')
 return wide

def frozen_split(wide):
 p=ROOT/'frozen_meaning_split.json'
 ids=[str(x) for x in wide.id]
 sample_hash=EXPECTED_SAMPLE_HASH
 if p.exists():
  obj=json.loads(p.read_text(encoding='utf-8'))
  assert obj['sample_hash']==sample_hash and obj['dataset_revision']==DATASET_REV
 else:
  rng=np.random.default_rng(SPLIT_SEED)
  perm=rng.permutation(750)
  baseline=sorted(map(int,perm[:450]));intervention=sorted(map(int,perm[450:]))
  smoke=sorted(map(int,rng.choice(baseline,120,replace=False)))
  greedy=sorted(map(int,rng.choice(intervention,100,replace=False)))
  obj={'split_seed':SPLIT_SEED,'dataset_revision':DATASET_REV,'sample_hash':sample_hash,
       'baseline450':baseline,'intervention300':intervention,'smoke120':smoke,'greedy100':greedy}
  atomic_json(p,obj)
 if 'generation150' not in obj:
  # Preserve an existing 100-meaning frozen subset, then add 50 meanings.
  hundred=list(map(int,obj['greedy100']))
  remaining=sorted(set(obj['intervention300'])-set(hundred))
  extra=np.random.default_rng(SPLIT_SEED+150).choice(remaining,50,replace=False)
  obj['generation100']=sorted(hundred)
  obj['generation150']=sorted(hundred+list(map(int,extra)))
  atomic_json(p,obj)
 assert len(obj['baseline450'])==450 and len(obj['intervention300'])==300
 assert set(obj['baseline450']).isdisjoint(obj['intervention300'])
 assert set(obj['smoke120'])<=set(obj['baseline450'])
 assert set(obj['greedy100'])<=set(obj['intervention300'])
 assert len(obj['generation100'])==100 and len(obj['generation150'])==150
 assert set(obj['generation100'])<=set(obj['generation150'])<=set(obj['intervention300'])
 progress(f'frozen split {p}; sample hash {sample_hash}')
 return obj


# %% original code cell 4; notebook cell 5
class Adapter:
 def __init__(self,kind):
  self.kind=kind;name,rev=MODEL_SPEC[kind]
  kw={'revision':rev,'trust_remote_code':kind=='it2'}
  if kind=='nllb':kw.update(use_fast=False,legacy_behaviour=False)
  self.tok=AutoTokenizer.from_pretrained(name,token=HF_TOKEN,**kw)
  self.model=AutoModelForSeq2SeqLM.from_pretrained(name,revision=rev,token=HF_TOKEN,trust_remote_code=kind=='it2',torch_dtype=torch.float32,attn_implementation='eager').eval().cuda()
  assert next(self.model.parameters()).dtype==torch.float32
  assert getattr(self.model.config,'_commit_hash',rev)==rev
  self.model_rev=rev;self.name=name
  if kind=='it2':
   assert os.environ.get('INDIC_RESOURCES_PATH') and Path(os.environ['INDIC_RESOURCES_PATH']).is_dir(),(
    'Set the pinned INDIC_RESOURCES_PATH in the setup cell before importing IndicProcessor.')
   from IndicTransToolkit.processor import IndicProcessor
   self.IndicProcessor=IndicProcessor
   assert hasattr(self.tok,'src_decoder') and hasattr(self.tok,'tgt_decoder')
  else:
   assert self.tok.legacy_behaviour is False
  self.language_id(CONTROL)
  for l in LANGS:self.language_id(l)
  progress(f'loaded {kind} {rev} fp32 eager; control {CONTROL} valid')
 def language_id(self,lang):
  if self.kind=='nllb':
   n=int(self.tok.convert_tokens_to_ids(lang))
   assert n!=int(self.tok.unk_token_id) and self.tok.convert_ids_to_tokens(n)==lang,lang
  else:
   # IT2 source and target-language tags are encoder-prefix tokens. The
   # decoder vocabulary holds target text pieces, not these prefix tags.
   assert lang in set(self.tok.src_decoder.values()),f'{lang} not in IT2 source vocab'
   n=next(k for k,v in self.tok.src_decoder.items() if v==lang)
  return n
 def pieces(self,ids,side='source'):
  ids=[int(x) for x in ids]
  if self.kind=='nllb':return self.tok.convert_ids_to_tokens(ids)
  dec=self.tok.src_decoder if side=='source' else self.tok.tgt_decoder
  return [dec[i] for i in ids]
 def encode(self,src,tag,text,src_texts,tgt_texts):
  t=self.tok
  if self.kind=='it2':
   ip=self.IndicProcessor(inference=True)
   st=ip.preprocess_batch(list(map(str,src_texts)),src_lang=src,tgt_lang=tag)
   x=t(st,return_tensors='pt',padding=True,truncation=True,max_length=MAX_LENGTH,return_attention_mask=True)
   tp=self.IndicProcessor(inference=False)
   tt=tp.preprocess_batch(list(map(str,tgt_texts)),src_lang=text,tgt_lang=src,is_target=True)
   y=t(text_target=tt,return_tensors='pt',padding=True,truncation=True,max_length=MAX_LENGTH,add_special_tokens=True,return_attention_mask=True)
  else:
   t.src_lang=src;t.tgt_lang=text;t.set_src_lang_special_tokens(src)
   x=t(list(map(str,src_texts)),return_tensors='pt',padding=True,truncation=True,max_length=MAX_LENGTH,return_attention_mask=True)
   y=t(text_target=list(map(str,tgt_texts)),return_tensors='pt',padding=True,truncation=True,max_length=MAX_LENGTH,add_special_tokens=True,return_attention_mask=True)
   for b in range(len(tgt_texts)):
    a=active(y['attention_mask'][b].tolist())
    assert int(y['input_ids'][b,a[0]])==self.language_id(text)
    y['input_ids'][b,a[0]]=self.language_id(tag)
  return x,y
 def categories(self,ids,mask,src,tag):
  ids=list(map(int,ids));a=active(mask);assert len(a)>=3
  p=self.pieces(ids,'source')
  if self.kind=='it2':
   assert p[a[0]]==src and p[a[1]]==tag,(p[a[0]],p[a[1]],src,tag)
   assert ids[a[-1]]==int(self.tok.eos_token_id),('IT2 EOS',p[a[-1]])
   base={'source_tag':[a[0]],'target_tag':[a[1]],'eos':[a[-1]]};body=a[2:-1]
  else:
   assert ids[a[0]]==self.language_id(src) and ids[a[-1]]==int(self.tok.eos_token_id)
   base={'source_code':[a[0]],'eos':[a[-1]]};body=a[1:-1]
  other_special=set(map(int,self.tok.all_special_ids))-{int(self.tok.unk_token_id)}
  assert not any(ids[i] in other_special for i in body),[(i,p[i]) for i in body if ids[i] in other_special]
  # The original IT2 collector removes every all_special_ids token, including
  # <unk>, before forming its lexical map. NLLB's collector retains <unk>.
  # Keep IT2 unknown mass as a separate component so full attention closes.
  if self.kind=='it2':base['unknown']=[i for i in body if ids[i]==int(self.tok.unk_token_id)]
  candidate=[i for i in body if self.kind!='it2' or i not in base['unknown']]
  base['punctuation']=[i for i in candidate if punctuation(p[i])]
  base['lexical']=[i for i in candidate if i not in base['punctuation']]
  assert sorted(sum(base.values(),[]))==a
  return base
 def target_rows(self,ids,mask,tag):
  a=active(mask);ids=list(map(int,ids))
  if self.kind=='nllb':
   assert ids[a[0]]==self.language_id(tag) and ids[a[-1]]==int(self.tok.eos_token_id)
   a=a[1:-1]
  else:
   a=[i for i in a if ids[i] not in set(map(int,self.tok.all_special_ids))]
  p=self.pieces(ids,'target')
  return [i for i in a if not punctuation(p[i])]
 def audit(self,src,tag,text,src_texts,tgt_texts):
  x,y=self.encode(src,tag,text,src_texts,tgt_texts)
  for b in range(len(src_texts)):
   ids=x['input_ids'][b].tolist();mask=x['attention_mask'][b].tolist()
   cats=self.categories(ids,mask,src,tag)
   print(json.dumps({'model':self.kind,'source':src,'tag':tag,'text':text,'attention_mask':mask,
                     'source_tokens':list(enumerate(self.pieces(ids))),
                     'source_positions':cats,'target_tokens':list(enumerate(self.pieces(y['input_ids'][b].tolist(),'target'))),
                     'target_mask':y['attention_mask'][b].tolist()},ensure_ascii=False,indent=2))
  return x,y


# %% original code cell 5; notebook cell 6
def shares_from_attention(att,cats,rows,mask):
 # att = layers x heads x target x source for one batch member
 assert rows and cats['lexical'],'No lexical source/target tokens'
 target=att[:,:,rows,:]
 out={k:target[:,:,:,pos].sum(axis=-1).mean(axis=-1) for k,pos in cats.items()}
 total=sum(out.values())
 err=float(np.max(np.abs(total-1)))
 assert err<1e-5,('per-head closure',err)
 return out,err

def teacher_batches(ad,wide,src,tag,text,ids,batch_size=BATCH_SIZE,head_dir=None):
 all_rows=[];all_heads=[];tic=time.perf_counter()
 for off in range(0,len(ids),batch_size):
  batch=ids[off:off+batch_size]
  x,y=ad.encode(src,tag,text,[wide.loc[i,src] for i in batch],[wide.loc[i,text] for i in batch])
  labels=y['input_ids'].clone();labels[y['attention_mask']==0]=-100
  with torch.inference_mode():
   out=ad.model(input_ids=x['input_ids'].cuda(),attention_mask=x['attention_mask'].cuda(),labels=labels.cuda(),output_attentions=True,return_dict=True)
   # First dimension is layer. No fp16/autocast in this notebook.
   att=torch.stack([v.detach().float().cpu() for v in out.cross_attentions],dim=0).numpy()
   loss=F.cross_entropy(out.logits.float().reshape(-1,out.logits.shape[-1]),labels.cuda().reshape(-1),reduction='none',ignore_index=-100).reshape(labels.shape).detach().cpu().numpy()
  del out
  for b,i in enumerate(batch):
   src_ids=x['input_ids'][b].tolist();src_mask=x['attention_mask'][b].tolist()
   tgt_ids=y['input_ids'][b].tolist();tgt_mask=y['attention_mask'][b].tolist()
   cats=ad.categories(src_ids,src_mask,src,tag);rows=ad.target_rows(tgt_ids,tgt_mask,tag)
   if not rows or not cats['lexical']:
    all_rows.append({'sentence_id':i,'corpus_sentence_id':str(wide.loc[i,'id']),'source':src,'tag':tag,'text':text,'eligible':False,'reason':'no lexical source or target'})
    continue
   h,err=shares_from_attention(att[:,b],cats,rows,src_mask)
   mean={k:float(v.mean()) for k,v in h.items()}
   # NLL excludes forced language prediction; IT2 includes lexical and EOS labels.
   nll_pos=[j for j in active(tgt_mask) if j not in ([active(tgt_mask)[0]] if ad.kind=='nllb' else [])]
   row={'sentence_id':i,'corpus_sentence_id':str(wide.loc[i,'id']),'source':src,'tag':tag,'text':text,
        'eligible':True,'reason':'','lm':mean['lexical'],'nll':float(np.mean(loss[b,nll_pos])),
        'nll_sum':float(np.sum(loss[b,nll_pos])),'nll_tokens':len(nll_pos),'closure_maxerr':err,
        'source_len':len(cats['lexical']),'target_len':len(rows),
        'source_unknown_tokens':len(cats.get('unknown',[])),**mean}
   assert np.isclose(sum(mean.values()),1,atol=1e-5)
   all_rows.append(row)
   if head_dir is not None:all_heads.append(np.stack([h[k] for k in h],axis=-1).astype(np.float32))
  del att,loss;x=None;y=None
  if off==0 or off+batch_size>=len(ids):
   progress(f'{ad.kind} TF {src} / tag {tag} / text {text}: {min(off+batch_size,len(ids))}/{len(ids)}; {time.perf_counter()-tic:.1f}s')
 if head_dir is not None:
  arr=np.stack(all_heads)
  head_dir=Path(head_dir);head_dir.parent.mkdir(parents=True,exist_ok=True)
  temp=head_dir.with_suffix('.tmp.npz')
  np.savez_compressed(temp,shares=arr,columns=np.array(list(h)),sentence_id=np.array([r['sentence_id'] for r in all_rows if r['eligible']]))
  temp.replace(head_dir)
 return pd.DataFrame(all_rows)

def collect_tf(ad,wide,cells,ids,folder,heads=True):
 out=[];folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 for k,(src,tag,text) in enumerate(cells,1):
  stem=f'{src}__{tag}__{text}'
  path=folder/(stem+'.csv');hp=folder/(stem+'_heads.npz')
  if path.exists() and (not heads or hp.exists()):
   frame=pd.read_csv(path);assert set(map(int,frame.sentence_id))==set(ids)
  else:
   frame=teacher_batches(ad,wide,src,tag,text,ids,head_dir=hp if heads else None)
   atomic_csv(path,frame)
  out.append(frame)
  progress(f'{ad.kind} completed TF cell {k}/{len(cells)}: {stem}')
 return pd.concat(out,ignore_index=True)


# %% original code cell 6; notebook cell 7
def generation_flags(hypothesis):
 import re
 from collections import Counter
 s=str(hypothesis).strip();words=s.split()
 empty=not s
 repetitive=bool(len(words)>=8 and Counter(words).most_common(1)[0][1]/len(words)>.5)
 long_run=bool(re.search(r'(.)\1{7,}',s))
 reasons=[name for name,flag in [('empty',empty),('repetitive_word',repetitive),('long_character_run',long_run)] if flag]
 return {'degenerate':bool(reasons),'degenerate_reason':','.join(reasons)}

def generate_one(ad,wide,src,tag,i):
 # Output-only generation: no attention tensors are requested or saved.
 t=ad.tok;text=str(wide.loc[i,src])
 if ad.kind=='it2':
  ip=ad.IndicProcessor(inference=True)
  st=ip.preprocess_batch([text],src_lang=src,tgt_lang=tag)
  x=t(st,return_tensors='pt',padding=True,truncation=True,max_length=MAX_LENGTH,return_attention_mask=True)
  args={}
 else:
  t.src_lang=src;t.tgt_lang=tag;t.set_src_lang_special_tokens(src)
  x=t([text],return_tensors='pt',padding=True,truncation=True,max_length=MAX_LENGTH,return_attention_mask=True)
  args={'forced_bos_token_id':ad.language_id(tag)}
 with torch.inference_mode():
  out=ad.model.generate(input_ids=x['input_ids'].cuda(),attention_mask=x['attention_mask'].cuda(),max_length=MAX_LENGTH,
      min_length=0,num_beams=1,do_sample=False,use_cache=True,output_attentions=False,return_dict_in_generate=True,**args)
 seq=out.sequences[0].tolist()
 if ad.kind=='nllb':assert len(seq)>1 and seq[1]==ad.language_id(tag)
 decoded=t.batch_decode(out.sequences,skip_special_tokens=True,clean_up_tokenization_spaces=(ad.kind=='it2'))
 if ad.kind=='it2':decoded=ip.postprocess_batch(decoded,lang=tag)
 ended_eos=bool(seq and seq[-1]==int(t.eos_token_id))
 record={'sentence_id':i,'corpus_sentence_id':str(wide.loc[i,'id']),'source':src,'tag':tag,'text':decoded[0],
         'reference':str(wide.loc[i,tag]),'generated_length':len(seq),
         'ended_with_eos':ended_eos,'generation_hit_cap':bool(not ended_eos and len(seq)>=MAX_LENGTH),
         **generation_flags(decoded[0])}
 del out
 return record

def collect_gen(ad,wide,cells,ids,folder):
 out=[];folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 for ci,(src,tag) in enumerate(cells,1):
  stem=f'{src}__{tag}';path=folder/(stem+'.csv')
  records=pd.read_csv(path,keep_default_na=False).to_dict('records') if path.exists() else []
  assert len({int(r['sentence_id']) for r in records})==len(records)
  assert set(int(r['sentence_id']) for r in records)<=set(ids)
  if records:assert {'generation_hit_cap','degenerate','text'}<=set(records[0]),'Old attention cache is not an output-only generation cache.'
  done={int(r['sentence_id']) for r in records}
  tic=time.perf_counter()
  for j,i in enumerate(ids,1):
   if i in done:continue
   r=generate_one(ad,wide,src,tag,i);records.append(r);done.add(i)
   if j==1 or j%25==0 or j==len(ids):
    atomic_csv(path,pd.DataFrame(records))
    progress(f'{ad.kind} generation {ci}/{len(cells)} {src}->{tag}: {j}/{len(ids)} {time.perf_counter()-tic:.1f}s')
  frame=pd.DataFrame(records).sort_values('sentence_id').reset_index(drop=True)
  assert set(map(int,frame.sentence_id))==set(ids)
  for field in ['ended_with_eos','generation_hit_cap','degenerate']:
   frame[field]=frame[field].map(lambda v:str(v).strip().lower() in {'true','1'})
  out.append(frame)
  progress(f'{ad.kind} completed generation direction {ci}/{len(cells)} {stem}')
 return pd.concat(out,ignore_index=True)


# %% original code cell 7; notebook cell 8
def design_matrix(df,levels=LANGS):
 # Sum coding via centered one-hot gives identifiable centered source/target effects.
 src=np.column_stack([(df.source.to_numpy()==l).astype(float) for l in levels])
 tgt=np.column_stack([(df.tag.to_numpy()==l).astype(float) for l in levels])
 return np.column_stack([np.ones(len(df)),src[:,:-1]-src[:,-1,None],tgt[:,:-1]-tgt[:,-1,None]])
def fit_additive(df,levels=LANGS,outcome='lm'):
 means=df[df.eligible].groupby(['source','tag'],as_index=False)[outcome].mean()
 assert len(means)==40,(outcome,len(means))
 x=design_matrix(means,levels);y=means[outcome].to_numpy(float)
 coef=np.linalg.lstsq(x,y,rcond=None)[0]
 # Under our sum coding, final level has negative sum of first n-1 coefficients.
 n=len(levels);alpha=np.r_[coef[1:n],-coef[1:n].sum()];beta=np.r_[coef[n:],-coef[n:].sum()]
 return dict(zip(levels,beta)),dict(zip(levels,alpha)),float(coef[0]),float(1-np.sum((y-x@coef)**2)/np.sum((y-y.mean())**2))

def graph_field(df):
 pivot=df[df.eligible].pivot_table(index='sentence_id',columns=['source','tag'],values='lm',aggfunc='mean')
 nodes=LANGS;edges=PAIRS
 d=np.array([(pivot[a,b]-pivot[b,a]).mean(skipna=True) for a,b in edges])
 b=np.array([[int(l==a)-int(l==c) for l in nodes] for a,c in edges],float)
 theta=np.linalg.lstsq(b,d,rcond=None)[0];r=d-b@theta
 return {'gradient_energy':float(1-np.dot(r,r)/np.dot(d,d)),'residual_l2':float(np.linalg.norm(r)),
         'theta':dict(zip(nodes,theta)),'edge_d':d.tolist(),'edge_residual':r.tolist()}

def meaning_bootstrap_field(df,reps=N_BOOT):
 ids=np.array(sorted(df.sentence_id.unique()),int)
 rng=np.random.default_rng(BOOT_SEED)
 grouped=df[df.eligible].groupby(['sentence_id','source','tag']).lm.mean().unstack(['source','tag']).reindex(ids)
 assert len(grouped)==len(ids)
 a=grouped.to_numpy();cols=list(grouped.columns)
 ix={pair:cols.index(pair) for pair in DIRECTIONS}
 e=np.column_stack([a[:,ix[(x,y)]]-a[:,ix[(y,x)]] for x,y in PAIRS])
 b=np.array([[int(l==x)-int(l==y) for l in LANGS] for x,y in PAIRS],float)
 pinv=np.linalg.pinv(b);p=b@pinv
 obs=np.nanmean(e,axis=0);res=(np.eye(len(PAIRS))-p)@obs
 energies=[];residuals=[];noise=[]
 centered=e-np.nanmean(e,axis=0)
 for _ in range(reps):
  draw=rng.integers(0,len(e),len(e))
  v=np.nanmean(e[draw],axis=0);rv=(np.eye(len(PAIRS))-p)@v
  energies.append(1-float(rv@rv/(v@v)));residuals.append(float(np.linalg.norm(rv)))
  # Conservative noise null preserves per-meaning edge covariance.
  nv=np.nanmean(centered[draw],axis=0);noise.append(float(np.linalg.norm((np.eye(len(PAIRS))-p)@nv)))
 null_p=1.0 if np.linalg.norm(res)<1e-12 else float((1+np.sum(np.array(noise)>=np.linalg.norm(res)))/(reps+1))
 return {'gradient_energy_ci':np.quantile(energies,[.025,.975]).tolist(),
         'residual_l2_ci':np.quantile(residuals,[.025,.975]).tolist(),
         'noise_null_p':null_p,
         'noise_null_q95':float(np.quantile(noise,.95)),'n_boot':reps}


# %% original code cell 8; notebook cell 9
def run_pilot(ad,wide,split,folder):
 ids=split['smoke120'];p=Path(folder)
 # Audit two rows with different lengths to expose left padding and every position.
 short=min(ids,key=lambda i:len(str(wide.loc[i,'kan_Knda'])))
 long=max(ids,key=lambda i:len(str(wide.loc[i,'kan_Knda'])))
 audit_x,audit_y=ad.audit('kan_Knda','mal_Mlym','mal_Mlym',[wide.loc[short,'kan_Knda'],wide.loc[long,'kan_Knda']],
          [wide.loc[short,'mal_Mlym'],wide.loc[long,'mal_Mlym']])
 audit_rows=[]
 for b,meaning_id in enumerate([short,long]):
  source_ids=audit_x['input_ids'][b].tolist()
  source_mask=audit_x['attention_mask'][b].tolist()
  target_ids=audit_y['input_ids'][b].tolist()
  target_mask=audit_y['attention_mask'][b].tolist()
  audit_rows.append({'sentence_id':meaning_id,'source':'kan_Knda','tag':'mal_Mlym',
                     'source_ids':source_ids,'source_mask':source_mask,
                     'source_tokens':ad.pieces(source_ids),
                     'source_positions':ad.categories(source_ids,source_mask,'kan_Knda','mal_Mlym'),
                     'target_ids':target_ids,'target_mask':target_mask,
                     'target_tokens':ad.pieces(target_ids,'target'),
                     'target_lexical_rows':ad.target_rows(target_ids,target_mask,'mal_Mlym')})
 atomic_json(p/'token_position_audit.json',audit_rows)
 mask=audit_x['attention_mask'][0].tolist()
 assert 0 in mask,'Select a more length-contrasting audit pair to verify padding.'
 padding_side_detected='left' if mask[0]==0 else 'right'
 progress(f'{ad.kind} observed {padding_side_detected} padding in printed audit')
 # Validate the extra control on the model's actual inference paths before
 # collecting 4,800 smoke rows. This meaning belongs to baseline450 only.
 control_id=ids[0]
 progress(f'{ad.kind} checking {CONTROL} control tag on baseline meaning {control_id}')
 control_probe=teacher_batches(ad,wide,'kan_Knda',CONTROL,'mal_Mlym',[control_id],batch_size=1)
 assert len(control_probe)==1 and bool(control_probe.eligible.iloc[0])
 assert float(control_probe.closure_maxerr.iloc[0])<1e-5
 control_generation=generate_one(ad,wide,'kan_Knda',CONTROL,control_id)
 assert control_generation['tag']==CONTROL and control_generation['sentence_id']==control_id
 atomic_json(p/'control_preflight.json',{
  'model':ad.kind,'control':CONTROL,'sentence_id':int(control_id),
  'teacher_forced_eligible':True,
  'teacher_forced_closure_maxerr':float(control_probe.closure_maxerr.iloc[0]),
  'generation_length':int(control_generation['generated_length']),
  'generation_hit_cap':bool(control_generation['generation_hit_cap'])})
 progress(f'{ad.kind} {CONTROL} teacher-forced and generation preflight passed')
 torch.cuda.reset_peak_memory_stats();batch_t0=time.perf_counter()
 probe=teacher_batches(ad,wide,'kan_Knda','mal_Mlym','mal_Mlym',ids[:BATCH_SIZE])
 torch.cuda.synchronize()
 one_batch_seconds=time.perf_counter()-batch_t0
 assert probe.eligible.all()
 old=OLD_ROOTS[ad.kind]/'teacher_forced_all_stats.csv'
 assert old.exists(),f'Smoke comparison requires original fp16 CSV: {old}'
 old_df=pd.read_csv(old)
 assert {'pair','sentence_id','mean','src_len','analysis_view'}<=set(old_df.columns)
 assert old_df.analysis_view.eq('lexical_no_punctuation').all()
 old_df['lm']=pd.to_numeric(old_df['mean'])*pd.to_numeric(old_df['src_len'])
 old_df[['source','tag']]=old_df.pair.str.split('__',expand=True)
 if ad.kind=='it2':
  # Exact token IDs are a stronger reproduction check than lexical lengths.
  # These two frozen baseline rows exposed the largest script-specific drift.
  assert {'source_token_ids_json','target_token_ids_json'}<=set(old_df.columns)
  parity=[];special=set(map(int,ad.tok.all_special_ids))
  for src,tgt,i in [('kan_Knda','ben_Beng',604),('ben_Beng','asm_Beng',677)]:
   assert i in ids,(src,tgt,i,'not in frozen smoke120')
   source=old_df[(old_df.source==src)&(old_df.tag==tgt)&(old_df.sentence_id==i)]
   assert len(source)==1,(src,tgt,i,len(source))
   old_row=source.iloc[0]
   x,y=ad.encode(src,tgt,tgt,[wide.loc[i,src]],[wide.loc[i,tgt]])
   sx=x['input_ids'][0].tolist();sm=x['attention_mask'][0].tolist()
   ty=y['input_ids'][0].tolist();tm=y['attention_mask'][0].tolist()
   ad.categories(sx,sm,src,tgt)
   new_source=[int(sx[j]) for j in active(sm)[2:] if int(sx[j]) not in special]
   new_target=[int(ty[j]) for j in active(tm) if int(ty[j]) not in special]
   old_source=list(map(int,json.loads(old_row.source_token_ids_json)))
   old_target=list(map(int,json.loads(old_row.target_token_ids_json)))
   parity.append({'source':src,'target':tgt,'sentence_id':int(i),
     'source_ids_match':new_source==old_source,'target_ids_match':new_target==old_target,
     'source_length_new':len(new_source),'source_length_old':len(old_source),
     'target_length_new':len(new_target),'target_length_old':len(old_target),
     'source_ids_new':new_source,'source_ids_old':old_source,
     'target_ids_new':new_target,'target_ids_old':old_target})
  atomic_json(p/'it2_token_id_parity.json',parity)
  assert all(r['source_ids_match'] and r['target_ids_match'] for r in parity),(
   'IndicTrans2 preprocessing differs from the old collector; inspect it2_token_id_parity.json',parity)
  progress('it2 old/new source and target token IDs match on Bengali and Assamese probes')
 beta_old,_,_,_=fit_additive(old_df.assign(eligible=True))
 cells=[(a,b,b) for a,b in DIRECTIONS]
 tic=time.perf_counter()
 # v3 restores the old pinned Indic NLP resources; do not reuse older CSVs.
 frame=collect_tf(ad,wide,cells,ids,p/'baseline_smoke_tf_v3')
 torch.cuda.synchronize()
 baseline_tf_seconds=time.perf_counter()-tic
 valid=frame[frame.eligible]
 assert len(valid)==40*120
 old_df=old_df[old_df.sentence_id.isin(ids)].copy()
 old_df=old_df[old_df[['source','tag']].apply(tuple,axis=1).isin(DIRECTIONS)]
 pair_new=valid.groupby(['source','tag']).lm.mean();pair_old=old_df.groupby(['source','tag']).lm.mean()
 assert set(pair_new.index)==set(pair_old.index) and len(pair_new)==40
 max_delta=float((pair_new-pair_old).abs().max())
 paired=valid.merge(old_df[['sentence_id','source','tag','lm','src_len','tgt_len']],
                    on=['sentence_id','source','tag'],suffixes=('_fp32','_fp16'),validate='one_to_one')
 paired['lm_abs_delta']=(paired.lm_fp32-paired.lm_fp16).abs()
 paired['source_len_delta']=paired.source_len-paired.src_len
 paired['target_len_delta']=paired.target_len-paired.tgt_len
 diagnostic_columns=['sentence_id','source','tag','lm_fp32','lm_fp16','lm_abs_delta',
                     'source_len','src_len','source_len_delta','target_len','tgt_len',
                     'target_len_delta','source_unknown_tokens']
 atomic_csv(p/'fp16_fp32_row_diagnostic.csv',paired[diagnostic_columns].sort_values('lm_abs_delta',ascending=False))
 progress(f'{ad.kind} fp16/fp32 max direction delta {max_delta:.6g}; row diagnostics saved')
 beta,_,_,_=fit_additive(valid)
 x=np.array([beta[l] for l in LANGS]);y=np.array([beta_old[l] for l in LANGS])
 rho=float(spearmanr(x,y).statistic)
 boot=[];rng=np.random.default_rng(BOOT_SEED)
 for _ in range(min(N_BOOT,1000)):
  draw=rng.choice(ids,len(ids),replace=True)
  new_b=valid.set_index('sentence_id').loc[draw].reset_index()
  xb=fit_additive(new_b)[0]
  boot.append(float(spearmanr([xb[l] for l in LANGS],[beta_old[l] for l in LANGS]).statistic))
 ci=np.quantile(boot,[.025,.975]).tolist()
 lm_tolerance_passed=bool(max_delta<=.002)
 beta_rank_tolerance_passed=bool(ci[1]>=.99)
 if not lm_tolerance_passed or not beta_rank_tolerance_passed:
  progress(f'{ad.kind} fp16 comparison needs review: max LM delta {max_delta:.6g}, beta rho {rho:.4f}, bootstrap CI {ci}; continuing smoke diagnostics only')
 # Same meaning, source, text: diagonal and tag-swapped. Other intervention meanings untouched.
 dry=ids[:4];diag=teacher_batches(ad,wide,'kan_Knda','mal_Mlym','mal_Mlym',dry,head_dir=p/'dry_diag_heads.npz')
 swap=teacher_batches(ad,wide,'kan_Knda','tam_Taml','mal_Mlym',dry,head_dir=p/'dry_swap_heads.npz')
 ctrl=teacher_batches(ad,wide,'kan_Knda',CONTROL,'mal_Mlym',dry,head_dir=p/'dry_control_heads.npz')
 ref=valid[(valid.source=='kan_Knda')&(valid.tag=='mal_Mlym')].set_index('sentence_id').loc[dry]
 assert np.allclose(ref.lm.to_numpy(),diag.lm.to_numpy(),atol=1e-4)
 assert set(swap.sentence_id)==set(dry) and set(ctrl.sentence_id)==set(dry)
 # IT2 encoder tag differs while source lexical IDs stay fixed; NLLB encoder is identical.
 x0,y0=ad.encode('kan_Knda','mal_Mlym','mal_Mlym',[wide.loc[dry[0],'kan_Knda']],[wide.loc[dry[0],'mal_Mlym']])
 x1,y1=ad.encode('kan_Knda','tam_Taml','mal_Mlym',[wide.loc[dry[0],'kan_Knda']],[wide.loc[dry[0],'mal_Mlym']])
 encoder_suffix_equal=None
 if ad.kind=='nllb':
  assert torch.equal(x0['input_ids'],x1['input_ids'])
  assert not torch.equal(y0['input_ids'],y1['input_ids'])
 else:
  assert not torch.equal(x0['input_ids'],x1['input_ids'])
  assert torch.equal(y0['input_ids'],y1['input_ids'])
  encoder_suffix_equal=bool(torch.equal(x0['input_ids'][:,2:],x1['input_ids'][:,2:]))
  progress(f'IT2 tokenized encoder suffix unchanged by tag swap: {encoder_suffix_equal}')
 gen=[];generation_t0=time.perf_counter()
 for tag in ['mal_Mlym','tam_Taml',CONTROL]:
  for i in ids[:12]:gen.append(generate_one(ad,wide,'kan_Knda',tag,i))
 torch.cuda.synchronize()
 smoke_generation_seconds=time.perf_counter()-generation_t0
 atomic_csv(p/'smoke_generation.csv',pd.DataFrame(gen))
 report={'kind':ad.kind,'role':'pilot','n_smoke':120,'fp16_fp32_direction_max_delta':max_delta,
         'fp16_fp32_tolerance_passed':lm_tolerance_passed,
         'beta_rank_tolerance_passed':beta_rank_tolerance_passed,
         'beta_rank_spearman':rho,'beta_rank_ci':ci,'diagonal_reproduction_max_delta':float(np.max(np.abs(ref.lm.to_numpy()-diag.lm.to_numpy()))),
         'control':CONTROL,'baseline_tf_seconds':baseline_tf_seconds,
         'smoke_generation_seconds':smoke_generation_seconds,
         'smoke_generation_seconds_per_sentence':smoke_generation_seconds/len(gen),
         'rough_150_generation_hours_per_model':smoke_generation_seconds/len(gen)*4200/3600,
         'rough_100_generation_hours_per_model':smoke_generation_seconds/len(gen)*2800/3600,
         'rough_full_tf_hours_per_model':one_batch_seconds*math.ceil((40*450+168*300)/BATCH_SIZE)/3600,
         'rough_total_150_hours_per_model':(
           smoke_generation_seconds/len(gen)*4200+
           one_batch_seconds*math.ceil((40*450+168*300)/BATCH_SIZE))/3600,
         'one_batch_seconds':one_batch_seconds,'one_batch_size':len(probe),
         'padding_side_detected':padding_side_detected,
         'it2_encoder_suffix_equal_on_dry_row':encoder_suffix_equal,
         'cuda_peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,'closure_maxerr':float(valid.closure_maxerr.max())}
 atomic_json(p/'smoke_report.json',report);print(json.dumps(report,indent=2))
 return report


# %% original code cell 9; notebook cell 10
def load_lid(role):
 import fasttext
 from huggingface_hub import hf_hub_download
 lock=ROOT/'lid_model_lock.json'
 if lock.exists():rev=json.loads(lock.read_text(encoding='utf-8'))['revision']
 else:
  assert role=='pilot','Run and freeze the pilot LID audit first.'
  rev=model_info('cis-lmu/glotlid').sha
  atomic_json(lock,{'repo':'cis-lmu/glotlid','filename':'model_v3.bin','revision':rev})
 path=hf_hub_download(repo_id='cis-lmu/glotlid',filename='model_v3.bin',revision=rev)
 model=fasttext.load_model(path)
 progress(f'GlotLID model_v3.bin revision {rev}')
 return model,rev

def attach_lid_and_chrf(frame,lid,wide):
 import sacrebleu
 frame=frame.copy();labels=[];probs=[];chrf=[]
 for r in frame.itertuples(index=False):
  s='' if pd.isna(r.text) else str(r.text).replace('\n',' ').strip()
  if s:
   label,prob=lid.predict(s,k=1)
   labels.append(label[0].removeprefix('__label__'));probs.append(float(prob[0]))
   chrf.append(float(sacrebleu.sentence_chrf(s,[str(wide.loc[int(r.sentence_id),r.tag])],word_order=2).score))
  else:labels.append('EMPTY');probs.append(np.nan);chrf.append(np.nan)
 frame['lid_label']=labels;frame['lid_probability']=probs;frame['off_target']=(frame.lid_label!=frame.tag)
 frame['chrfpp']=chrf
 return frame

def run_pilot_lid(ad,wide,folder):
 p=Path(folder)/'smoke_generation.csv'
 frame=pd.read_csv(p)
 lid,rev=load_lid('pilot')
 frame=attach_lid_and_chrf(frame,lid,wide)
 atomic_csv(Path(folder)/'smoke_generation_lid.csv',frame)
 assert frame.lid_label.notna().all()
 summary=frame.groupby('tag').agg(n=('off_target','size'),off_target_rate=('off_target','mean'),mean_chrfpp=('chrfpp','mean')).reset_index()
 print(summary.to_string(index=False))
 labels=frame.pivot(index='sentence_id',columns='tag',values='lid_label')
 tag_changes=float((labels['mal_Mlym']!=labels['tam_Taml']).mean())
 assert tag_changes>0,'Swapped tag did not change any detected output language in smoke.'
 report=json.loads((Path(folder)/'smoke_report.json').read_text(encoding='utf-8'))
 report['lid_revision']=rev;report['lid_works']=True;report['tag_changes_lid_fraction']=tag_changes
 report['status']=('passed_requires_human_token_position_review'
                  if report['fp16_fp32_tolerance_passed'] and report['beta_rank_tolerance_passed']
                  else 'fp16_comparison_requires_review')
 atomic_json(Path(folder)/'smoke_report.json',report)
 return report


# %% original code cell 10; notebook cell 11
def baseline_analysis(kind,tf,split,folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 assert len(tf)==40*450 and tf.eligible.all()
 assert set(tf.sentence_id)==set(split['baseline450'])
 beta,alpha,mu,fit_r2=fit_additive(tf)
 mass_cols=['source_tag','target_tag','eos','unknown','punctuation','lexical'] if kind=='it2' else ['source_code','eos','punctuation','lexical']
 component=[]
 for c in mass_cols:
  b,a,m,r2=fit_additive(tf,outcome=c)
  component.extend({'component':c,'language':l,'beta':b[l],'alpha':a[l],'r2':r2} for l in LANGS)
 comp=pd.DataFrame(component);atomic_csv(folder/'baseline_component_additive.csv',comp)
 for l in LANGS:
  assert abs(comp.loc[comp.language==l,'beta'].sum())<1e-5
 tf_field=graph_field(tf);tf_boot=meaning_bootstrap_field(tf)
 result={'kind':kind,'beta_hat_baseline450':beta,'alpha_baseline450':alpha,'mu_baseline450':mu,
   'additive_r2_baseline450':fit_r2,
   'teacher_forced_baseline450':{**tf_field,**tf_boot},
   'F11_nonconservative_residual_status':('retain_for_investigation' if tf_boot['noise_null_p']<.05 else 'drop_noise_compatible'),
   'tf_component_means':{c:float(tf[c].mean()) for c in mass_cols},
   'old_fp16_used_as_predictor':False}
 atomic_json(folder/'baseline_analysis.json',result)
 progress(f'{kind} baseline beta_hat computed from 450 fp32 meanings; additive R2 {fit_r2:.5f}')
 return result

def indexed_cube(df,ids,cols):
 index=pd.MultiIndex.from_product([ids,SOURCES,TARGETS,TARGETS],names=['sentence_id','source','tag','text'])
 sub=df[df.tag.isin(TARGETS)].set_index(['sentence_id','source','tag','text']).reindex(index)
 assert len(sub)==len(ids)*4*6*6
 assert sub.eligible.fillna(False).all(),f'Missing or ineligible intervention cell: {sub[~sub.eligible.fillna(False)].head()}'
 return {c:sub[c].to_numpy(float).reshape(len(ids),4,6,6) for c in cols}

def pair_arrays(cube,beta,mask_mar=False,keep=None):
 # Orientation: [meaning, source, proposed B, baseline A].
 if keep is None:keep=list(range(6))
 a=cube[:,:,keep,:][:,:,:,keep]
 n=len(keep);diag=np.diagonal(a,axis1=2,axis2=3)
 tag_delta=a-diag[:,:,None,:]
 text_delta=np.swapaxes(a,2,3)-diag[:,:,None,:]
 bv=np.array([beta[TARGETS[k]] for k in keep]);x=bv[:,None]-bv[None,:]
 mask=~np.eye(n,dtype=bool)
 if mask_mar and TARGETS.index('hin_Deva') in keep:
  hin=keep.index(TARGETS.index('hin_Deva'))
  mar_mask=mask.copy();mar_mask[hin,:]=False;mar_mask[:,hin]=False
  mask4=np.broadcast_to(mask,(len(a),4,n,n)).copy();mask4[:,SOURCES.index('mar_Deva')]=mar_mask
 else:mask4=np.broadcast_to(mask,(len(a),4,n,n))
 return x,tag_delta,text_delta,mask4

def slope_by_meaning(cube,beta,kind='tag',mask_mar=False,keep=None):
 x,dt,dx,mask=pair_arrays(cube,beta,mask_mar,keep)
 y=dt if kind=='tag' else dx
 num=np.where(mask,y*x[None,None,:,:],0).sum(axis=(1,2,3))
 den=np.where(mask,x[None,None,:,:]**2,0).sum(axis=(1,2,3))
 return num,den

def slope_ci(num,den,draws):
 point=float(num.sum()/den.sum())
 b=num[draws].sum(axis=1)/den[draws].sum(axis=1)
 return point,np.quantile(b,[.025,.975]).tolist(),b

def adjusted_slope(cube_y,cube_nll,beta,which,draws):
 x,dt,dx,mask=pair_arrays(cube_y,beta)
 _,nt,nx,_=pair_arrays(cube_nll,beta)
 y=dt if which=='tag' else dx;z=nt if which=='tag' else nx
 xv=np.broadcast_to(x[None,None,:,:],y.shape)
 # Accumulate normal-equation sufficient statistics per meaning.
 stat=np.column_stack([np.where(mask,xv*xv,0).sum((1,2,3)),np.where(mask,xv*z,0).sum((1,2,3)),
                       np.where(mask,z*z,0).sum((1,2,3)),np.where(mask,xv*y,0).sum((1,2,3)),
                       np.where(mask,z*y,0).sum((1,2,3))])
 def fit(s):
  xx,xz,zz,xy,zy=s
  return float(np.linalg.lstsq(np.array([[xx,xz],[xz,zz]]),np.array([xy,zy]),rcond=None)[0][0])
 point=fit(stat.sum(axis=0));vals=[fit(stat[d].sum(axis=0)) for d in draws]
 return point,np.quantile(vals,[.025,.975]).tolist()

def fe_effects(cube,beta,draws,nll_cube=None):
 # Balanced cell means for each meaning, then cluster bootstrap by meaning.
 grid=pd.MultiIndex.from_product([SOURCES,TARGETS,TARGETS],names=['source','tag','text']).to_frame(index=False)
 def code(col,levels):
  o=np.column_stack([(grid[col].to_numpy()==v).astype(float) for v in levels])
  return o[:,:-1]-o[:,-1,None]
 x=np.column_stack([np.ones(len(grid)),code('source',SOURCES),code('tag',TARGETS),code('text',TARGETS)])
 y=cube.reshape(len(cube),-1)
 if nll_cube is not None:x=np.column_stack([x,nll_cube.mean(axis=0).reshape(-1)])
 pinv=np.linalg.pinv(x)
 fit=pinv@y.mean(axis=0)
 # For NLL-adjusted FE, refit each bootstrap against its resampled mean NLL.
 def unpack(coef):
  tag=np.r_[coef[4:9],-coef[4:9].sum()]
  text=np.r_[coef[9:14],-coef[9:14].sum()]
  b=np.array([beta[l] for l in TARGETS]);b=b-b.mean()
  return {'tag_effect':dict(zip(TARGETS,tag)),'text_effect':dict(zip(TARGETS,text)),
          'tag_slope_on_beta':float(tag@b/(b@b)),'text_slope_on_beta':float(text@b/(b@b)),
          'tag_spearman_beta':float(spearmanr(tag,b).statistic),'text_spearman_beta':float(spearmanr(text,b).statistic)}
 result=unpack(fit)
 if nll_cube is None:
  coef_by_meaning=(pinv@y.T).T
  boot=coef_by_meaning[draws].mean(axis=1)
  result['tag_slope_ci']=np.quantile([unpack(c)['tag_slope_on_beta'] for c in boot],[.025,.975]).tolist()
  result['text_slope_ci']=np.quantile([unpack(c)['text_slope_on_beta'] for c in boot],[.025,.975]).tolist()
 else:
  result['nll_coefficient']=float(fit[-1])
 return result

def analyze_intervention(kind,inter,gen,wide,split,beta,folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 ids=split['intervention300'];rng=np.random.default_rng(BOOT_SEED)
 draws=rng.integers(0,len(ids),size=(N_BOOT,len(ids)))
 cols=['source_tag','target_tag','eos','unknown','punctuation','lexical'] if kind=='it2' else ['source_code','eos','punctuation','lexical']
 cube=indexed_cube(inter,ids,cols+['lm','nll'])
 slopes={};boot_sums={}
 for c in cols:
  slopes[c]={}
  for which in ['tag','text']:
   num,den=slope_by_meaning(cube[c],beta,which)
   point,ci,boot=slope_ci(num,den,draws)
   slopes[c][which]={'slope':point,'ci':ci};boot_sums[(c,which)]=boot
 for which in ['tag','text']:
  assert abs(sum(slopes[c][which]['slope'] for c in cols))<1e-5,'Component slope closure'
 btag=slopes['lexical']['tag']['slope'];btext=slopes['lexical']['text']['slope']
 sum_boot=boot_sums['lexical','tag']+boot_sums['lexical','text']
 sum_ci=np.quantile(sum_boot,[.025,.975]).tolist()
 share=None;share_ci=None
 if btag>0 and btext>0 and (sum_ci[0]>0 or sum_ci[1]<0):
  share=btag/(btag+btext)
  share_ci=np.quantile(boot_sums['lexical','tag']/sum_boot,[.025,.975]).tolist()
 adjusted={w:adjusted_slope(cube['lm'],cube['nll'],beta,w,draws) for w in ['tag','text']}
 fe=fe_effects(cube['lm'],beta,draws)
 fe_nll=fe_effects(cube['lm'],beta,draws,nll_cube=cube['nll'])
 jackknife={}
 for omitted in range(6):
  keep=[i for i in range(6) if i!=omitted]
  jackknife[TARGETS[omitted]]={w:float(sum(slope_by_meaning(cube['lm'],beta,w,keep=keep)[0])/sum(slope_by_meaning(cube['lm'],beta,w,keep=keep)[1])) for w in ['tag','text']}
 mar={w:float(sum(slope_by_meaning(cube['lm'],beta,w,mask_mar=True)[0])/sum(slope_by_meaning(cube['lm'],beta,w,mask_mar=True)[1])) for w in ['tag','text']}
 # Compare the Urdu control tag against within-set tag swaps on matched cells.
 control=inter[inter.tag==CONTROL].set_index(['sentence_id','source','text']).lm
 diagonal=inter[(inter.tag==inter.text)&(inter.tag.isin(TARGETS))].set_index(['sentence_id','source','text']).lm
 assert len(control)==len(diagonal)==len(ids)*4*6
 control_delta=(control-diagonal).abs()
 x,tag_delta,_,mask=pair_arrays(cube['lm'],beta)
 in_set_abs=float(np.abs(tag_delta[mask]).mean())
 control_abs=float(control_delta.mean())
 # Every generated output, including cap-hit and degenerate rows, enters LID/chrF.
 lid,rev=load_lid('full')
 gen=attach_lid_and_chrf(gen,lid,wide)
 atomic_csv(folder/'generation_lid_chrf.csv',gen)
 def wilson(k,n):
  z=1.959963984540054;p=k/n;den=1+z*z/n
  center=(p+z*z/(2*n))/den
  half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
  return center-half,center+half
 rates=gen.groupby(['source','tag'],as_index=False).agg(n=('off_target','size'),off_target_count=('off_target','sum'),
   off_target_rate=('off_target','mean'),mean_chrfpp=('chrfpp','mean'),cap_hit_count=('generation_hit_cap','sum'),
   degenerate_count=('degenerate','sum'))
 intervals=[wilson(int(r.off_target_count),int(r.n)) for r in rates.itertuples(index=False)]
 rates['off_target_ci_low']=[x[0] for x in intervals]
 rates['off_target_ci_high']=[x[1] for x in intervals]
 atomic_csv(folder/'generation_cell_summary.csv',rates)
 by_tag=gen.groupby('tag',as_index=False).agg(n=('off_target','size'),off_target_count=('off_target','sum'),
   off_target_rate=('off_target','mean'),mean_chrfpp=('chrfpp','mean'),cap_hit_count=('generation_hit_cap','sum'),
   degenerate_count=('degenerate','sum'))
 intervals=[wilson(int(r.off_target_count),int(r.n)) for r in by_tag.itertuples(index=False)]
 by_tag['off_target_ci_low']=[x[0] for x in intervals]
 by_tag['off_target_ci_high']=[x[1] for x in intervals]
 atomic_csv(folder/'generation_tag_summary.csv',by_tag)
 mar_hin=rates[(rates.source=='mar_Deva')&(rates.tag=='hin_Deva')]
 result={'kind':kind,'n_intervention_meanings':len(ids),'n_intervention_cells':168,'n_rows':len(inter),
  'component_paired_slopes':slopes,'b_tag':btag,'b_text':btext,'b_sum':btag+btext,'b_sum_ci':sum_ci,
  'tag_share':share,'tag_share_ci':share_ci,'classification':None if share is None else ('tag-driven' if share>.7 else 'text-driven' if share<.3 else 'mixed'),
  'nll_adjusted_slopes':{w:{'point':adjusted[w][0],'ci':adjusted[w][1]} for w in adjusted},
  'fixed_effects':fe,'fixed_effects_with_nll':fe_nll,
  'leave_one_tag_language_out':jackknife,'exclude_mar_hin_sensitivity':mar,
  'control_urd_mean_abs_shift':control_abs,'within_set_mean_abs_shift':in_set_abs,
  'control_shift_smaller_than_within_set':bool(control_abs<in_set_abs),
  'n_generation_meanings':int(gen.sentence_id.nunique()),'n_generation_rows':len(gen),
  'generation_off_target_overall':float(gen.off_target.mean()),'generation_mean_chrfpp':float(gen.chrfpp.mean()),
  'generation_cap_hit_count':int(gen.generation_hit_cap.sum()),'generation_degenerate_count':int(gen.degenerate.sum()),
  'mar_hin_generation_cell':mar_hin.to_dict(orient='records'),'lid_revision':rev}
 atomic_json(folder/'intervention_analysis.json',result)
 print(json.dumps({k:result[k] for k in ['kind','b_tag','b_text','b_sum','b_sum_ci','tag_share','tag_share_ci','classification','nll_adjusted_slopes','control_urd_mean_abs_shift','within_set_mean_abs_shift','generation_off_target_overall']},indent=2))
 return result


# %% original code cell 11; notebook cell 12
def run_full(ad,wide,split,folder):
 p=Path(folder);p.mkdir(parents=True,exist_ok=True)
 smoke=ROOT/'pilot'/ad.kind/'smoke_report.json'
 assert smoke.exists(),f'Run pilot smoke for {ad.kind} and review token positions first: {smoke}'
 smoke_report=json.loads(smoke.read_text(encoding='utf-8'))
 assert smoke_report.get('lid_works') and smoke_report['fp16_fp32_direction_max_delta']<=.002
 # Human verification is recorded explicitly after inspecting the audit printout.
 review=ROOT/'pilot'/ad.kind/'token_positions_reviewed.json'
 assert review.exists(),f'Inspect the printed token/mask audit and save {review} with {{"verified":true}}.'
 assert json.loads(review.read_text(encoding='utf-8')).get('verified') is True
 generation_ids=split[f'generation{N_GENERATION_MEANINGS}']
 other=ROOT/('confirm' if ad.kind=='it2' else 'holdout')/'run_lock.json'
 if other.exists():
  other_lock=json.loads(other.read_text(encoding='utf-8'))
  assert other_lock['generation_meanings']==N_GENERATION_MEANINGS,'Both models must use the same generation subset size.'
 lock={'model':ad.name,'model_revision':ad.model_rev,'dataset':DATASET,'dataset_revision':DATASET_REV,
          'split_hash':split['sample_hash'],'split_seed':SPLIT_SEED,'bootstrap_seed':BOOT_SEED,'n_boot':N_BOOT,
          'targets':TARGETS,'sources':SOURCES,'control':CONTROL,'generation_meanings':N_GENERATION_MEANINGS,
          'generation_ids_sha256':digest('|'.join(map(str,generation_ids))),
          'torch_version':torch.__version__,'transformers_version':__import__('transformers').__version__}
 run_lock=p/'run_lock.json'
 if run_lock.exists():
  previous=json.loads(run_lock.read_text(encoding='utf-8'))
  assert previous==lock,'Run settings changed after this model began; use the frozen settings or a new output root.'
 else:atomic_json(run_lock,lock)
 # Step 1: 40 original directions × baseline450, fp32 full attention.
 tf=collect_tf(ad,wide,[(a,b,b) for a,b in DIRECTIONS],split['baseline450'],p/'baseline_tf')
 # The baseline and intervention meanings are disjoint. Check diagonal
 # reproduction with duplicate forwards on the same intervention meanings.
 overlap_directions=[(s,t) for s in SOURCES for t in TARGETS if (s,t) in set(DIRECTIONS)]
 assert len(overlap_directions)==9,overlap_directions
 preflight_ids=split['intervention300'][:4]
 preflight_deltas=[];preflight_reference={}
 for s,t in overlap_directions:
  batched=teacher_batches(ad,wide,s,t,t,preflight_ids)
  single=teacher_batches(ad,wide,s,t,t,preflight_ids,batch_size=1)
  by_id=single.set_index('sentence_id')
  for row in batched.itertuples(index=False):
   key=(row.sentence_id,s,t)
   preflight_reference[key]=float(row.lm)
   preflight_deltas.append(abs(row.lm-float(by_id.loc[row.sentence_id,'lm'])))
 preflight_max=float(max(preflight_deltas))
 atomic_json(p/'analysis'/'diagonal_reproduction.json',
             {'preflight_max_delta':preflight_max,'preflight_rows':len(preflight_deltas),
              'comparison':'same intervention meanings, batch of four versus batch of one','tolerance':1e-4})
 progress(f'{ad.kind} early diagonal reproduction max {preflight_max:.8g} on {len(preflight_deltas)} rows')
 assert preflight_max<1e-4,('Early diagonal reproduction failed',preflight_max)
 baseline=baseline_analysis(ad.kind,tf,split,p/'analysis')
 beta=baseline['beta_hat_baseline450']
 # Step 2: 4 sources × (6 set tags + Urdu) × 6 forced-text languages × 300.
 cells=[(s,t,x) for s in SOURCES for t in TARGETS+[CONTROL] for x in TARGETS]
 assert len(cells)==168
 intervention=collect_tf(ad,wide,cells,split['intervention300'],p/'intervention_tf')
 assert len(intervention)==168*300 and intervention.eligible.all()
 diag=intervention[(intervention.tag==intervention.text)&(intervention.tag.isin(TARGETS))]
 overlap=[]
 for row in diag.itertuples(index=False):
  key=(row.sentence_id,row.source,row.tag)
  if key in preflight_reference:overlap.append(abs(row.lm-preflight_reference[key]))
 full_diag_max=float(max(overlap)) if overlap else None
 atomic_json(p/'analysis'/'diagonal_reproduction.json',
             {'preflight_max_delta':preflight_max,'preflight_rows':len(preflight_deltas),
              'full_max_delta':full_diag_max,'reproduced_rows':len(overlap),
              'diagonal_intervention_rows':len(diag),
              'comparison':'early duplicate forward versus matching intervention-grid row','tolerance':1e-4})
 assert len(overlap)==len(preflight_reference) and full_diag_max<1e-4,('diagonal reproduction failed',full_diag_max)
 # Step 3: 150 frozen intervention meanings (or prespecified 100 fallback), 4×7 outputs only.
 gen=collect_gen(ad,wide,[(s,t) for s in SOURCES for t in TARGETS+[CONTROL]],generation_ids,p/'intervention_free_outputs')
 assert len(gen)==N_GENERATION_MEANINGS*4*7
 result=analyze_intervention(ad.kind,intervention,gen,wide,split,beta,p/'analysis')
 result['diagonal_reproduction_max_delta']=full_diag_max
 atomic_json(p/'analysis'/'intervention_analysis.json',result)
 return result


# %% original code cell 12; notebook cell 13
wide=load_meanings()
split=frozen_split(wide)
ad=Adapter('nllb')
result=run_full(ad,wide,split,ROOT/'confirm')
print('Final analysis saved to',ROOT/'confirm'/'analysis')
# Side-by-side accounting once both full baselines are available.
it2_path=ROOT/'holdout'/'analysis'/'baseline_analysis.json'
nllb_path=ROOT/'confirm'/'analysis'/'baseline_analysis.json'
if it2_path.exists() and nllb_path.exists():
    it2=json.loads(it2_path.read_text(encoding='utf-8'))
    nllb=json.loads(nllb_path.read_text(encoding='utf-8'))
    comparison={model:{'tf_component_means':report['tf_component_means']}
                for model,report in [('it2',it2),('nllb',nllb)]}
    atomic_json(ROOT/'component_comparison.json',comparison)
    print('IT2/NLLB teacher-forced tag, EOS, punctuation and lexical shares:')
    print(json.dumps(comparison,indent=2))

