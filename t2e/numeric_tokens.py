"""Exact numeric answer tokenization cache for the audited InternLM2 BPE.

The ONSEP prompt ends with a comma and answers start with ASCII digits. If
no vocabulary token crosses that boundary, BPE cannot merge across it. The
normalizer and special-token template are checked before enabling this path.
Other tokenizers or answer formats retain the original implementation.
"""
import hashlib
import json
import re
from functools import lru_cache
from types import MethodType


def install(scorer,expected_tokenizer_sha256=None):
    tokenizer=scorer.tokenizer
    if not getattr(tokenizer,'is_fast',False):return {'enabled':False,'reason':'slow tokenizer'}
    serialized=tokenizer.backend_tokenizer.to_str()
    digest=hashlib.sha256(serialized.encode()).hexdigest()
    if expected_tokenizer_sha256 is not None and digest!=expected_tokenizer_sha256:
        raise ValueError('Tokenizer differs from the tokenization equivalence audit')
    backend=json.loads(serialized);model=backend['model']
    normalizer={'type':'Sequence','normalizers':[{'type':'Replace','pattern':{'String':' '},'content':'\u2581'}]}
    post=backend.get('post_processor',{})
    template=[{'SpecialToken':{'id':'<s>','type_id':0}},{'Sequence':{'id':'A','type_id':0}}]
    if (model.get('type')!='BPE' or model.get('dropout') is not None or
        backend.get('normalizer')!=normalizer or backend.get('pre_tokenizer') is not None or
        post.get('type')!='TemplateProcessing' or post.get('single')!=template):
        return {'enabled':False,'reason':'unsupported tokenizer structure','tokenizer_sha256':digest}
    tokens=list(model['vocab'])+[t['content'] for t in backend.get('added_tokens',[])]
    if any(re.search(r',[0-9]',token) for token in tokens):
        return {'enabled':False,'reason':'vocabulary can cross numeric boundary','tokenizer_sha256':digest}
    original=scorer._answer_tokens
    scorer._encode=lru_cache(maxsize=256)(scorer._encode)

    @lru_cache(maxsize=512)
    def suffix_tokens(answer):
        return tuple(tokenizer.encode(answer,add_special_tokens=False))

    def answer_tokens(self,prompt,answer,prefix):
        if prompt.endswith(',') and re.fullmatch(r'[0-9]+\n?',answer):
            suffix=list(suffix_tokens(answer))
            return prefix+suffix,suffix
        return original(prompt,answer,prefix)

    scorer._answer_tokens=MethodType(answer_tokens,scorer)
    return {'enabled':True,'tokenizer_sha256':digest,'boundary':'comma followed by ASCII digit',
            'vocabulary_boundary_crossings':0,'prefix_cache_entries':256,'suffix_cache_entries':512}
