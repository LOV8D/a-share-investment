from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from io import BytesIO
import hashlib,json
import pandas as pd
import requests
from pypdf import PdfReader
p=Path('data/reconstruction/20260915-20261116');d=pd.read_parquet(p/'selected_documents.parquet')
cache=p/'pdf_text';cache.mkdir(exist_ok=True)
# Bounded disclosure excerpts; incomplete documents are explicitly labeled.
def one(row):
 dest=cache/(row['art_code']+'.json')
 if dest.exists():return row|json.loads(dest.read_text())
 try:
  s=requests.Session();s.trust_env=False
  r=s.get(f"https://pdf.dfcfw.com/pdf/H2_{row['art_code']}_1.pdf",timeout=(4,12));r.raise_for_status()
  if len(r.content)>15_000_000:raise ValueError('PDF size budget')
  reader=PdfReader(BytesIO(r.content));chunks=[]
  for page in reader.pages[:8]:
   chunks.append(page.extract_text() or '')
   if sum(map(len,chunks))>22000:break
  body='\n'.join(chunks)[:22000]
  if len(body)<30:raise ValueError('No text layer')
  result={'text':body,'text_scope':'pdf_excerpt','pdf_pages':len(reader.pages),'pdf_sha256':hashlib.sha256(r.content).hexdigest()}
  dest.write_text(json.dumps(result,ensure_ascii=False));return row|result
 except Exception as e:return row|{'content_error':type(e).__name__}
rows=[]
with ThreadPoolExecutor(max_workers=6) as pool:
 fs=[pool.submit(one,r) for r in d.to_dict('records')]
 for i,f in enumerate(as_completed(fs)):
  rows.append(f.result())
  if (i+1)%100==0:print('公告PDF',i+1,'/',len(d),'成功',sum(r['text_scope']=='pdf_excerpt' for r in rows),flush=True)
result=pd.DataFrame(rows);result.to_parquet(p/'selected_documents.parquet',index=False);print(result.text_scope.value_counts().to_dict(),flush=True)
