"""Generate reference prompt rows with the canonical HF tokenizer, without inference.
JSON fixtures cover plain, directed, speaker-switched, reference and prior-take inputs.
"""
import argparse,hashlib,json,pathlib
from tokenizers import Tokenizer

p=argparse.ArgumentParser();p.add_argument('--tokenizer',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
tokenizer=Tokenizer.from_file(str(a.tokenizer));special=lambda t:tokenizer.token_to_id(t)
def tagged(text):return text if text.lstrip().startswith('<|speaker:') else '<|speaker:0|> '+text
def rows(text,references,history):
 segments=['<|im_start|>system\n','convert the provided text to speech reference to the following:' if references else 'convert the provided text to speech']
 for i,(transcript,codes)in enumerate(references,1):
  tag=tagged(transcript);body=tag.lstrip();end=body.index('|>')+2
  transcript=body[:end]+tag[:len(tag)-len(body)]+body[end:]
  segments += [f'\n\n[Reference {i}]\nText: {transcript}\nSpeech: ',codes]
 segments+=['<|im_end|>\n']
 for turn,codes in history:segments+=['<|im_start|>user\n',tagged(turn),'<|im_end|>\n','<|im_start|>assistant\n<|voice|>',codes,'<|im_end|>\n']
 segments+=['<|im_start|>user\n',tagged(text),'<|im_end|>\n','<|im_start|>assistant\n<|voice|>']
 out=[]
 for segment in segments:
  if isinstance(segment,str):out += [[i]+[0]*10 for i in tokenizer.encode(segment,add_special_tokens=False).ids]
  else:out += [[f[0]+special('<|semantic:0|>')]+f for f in segment]
 return out
codes=[[i+j for j in range(10)]for i in range(2)]
cases=[('plain','The evidence remains clear.',[],[]),('directed','<whisper>Do not change this.</whisper> [sighs] I understand.',[],[]),('speakers','<|speaker:0|>Hello. <|speaker:15|>Guten Tag!',[],[]),('reference','The records are unchanged.',[('  <|speaker:1|>The reference.',codes)],[]),('history','The report is due on Thursday.',[],[('The report is due on Wednesday.',codes)])]
data={'scope':'canonical tokenizer and prompt rows','model_revision':'e0a9218aa48888270f06ef1ed9ea67622c0e17d2','tokenizer_sha256':hashlib.file_digest(a.tokenizer.open('rb'),'sha256').hexdigest(),
 'cases':[{'name':name,'text':text,'references':[{'transcript':t,'frames':c}for t,c in refs],'history':[{'text':t,'frames':c}for t,c in history],'expected':rows(text,refs,history)}for name,text,refs,history in cases]}
a.out.write_text(json.dumps(data,indent=2)+'\n');print('Generated five canonical prompt-row fixtures')
