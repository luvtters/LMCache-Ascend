"""Sequential official vLLM bench serve fill/revisit, without cache intervention."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--base-url',required=True)
    p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--service-port',type=int,default=31685)
    p.add_argument('--p-port',type=int,default=31920)
    p.add_argument('--d-port',type=int,default=31930)
    p.add_argument('--stop-new-at',default='2026-11-01T07:30:00+08:00')
    p.add_argument('--stop-at',default='2026-11-01T07:45:00+08:00')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.data/'manifest.json').read_text(encoding='utf-8'))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def snap(out,tag):
        for role,port,endpoint in [('service',a.service_port,'/status'),('p',a.p_port,'/metrics'),('d',a.d_port,'/metrics')]:
            try:
                with opener.open(f'http://127.0.0.1:{port}{endpoint}',timeout=3) as r:
                    (out/(tag+'_'+role+'.txt')).write_bytes(r.read())
            except Exception as exc:
                (out/(tag+'_'+role+'.error.txt')).write_text(repr(exc),encoding='utf-8')
    for phase in ['fill','revisit']:
        if time.time()>=datetime.fromisoformat(a.stop_new_at).timestamp():
            raise TimeoutError('Stop-new deadline; phase was not started')
        out=a.output/phase;out.mkdir()
        cmd=[sys.executable,'-m','vllm.entrypoints.cli.main','bench','serve','--backend','openai',
             '--base-url',a.base_url,'--endpoint','/v1/completions','--model',a.model,'--served-model-name',a.model,
             '--tokenizer',a.tokenizer,'--dataset-name','custom','--dataset-path',str(a.data/(phase+'.jsonl')),
             '--custom-output-len',str(manifest['output_tokens']),'--num-prompts',str(manifest['sessions']),
             '--max-concurrency',str(a.concurrency),'--request-rate','inf','--temperature','0',
             '--num-warmups','0','--ready-check-timeout-sec','0','--ignore-eos','--disable-shuffle',
             '--no-oversample','--seed','0','--skip-chat-template','--save-result','--save-detailed',
             '--percentile-metrics','ttft,tpot,itl,e2el',
             '--result-dir',str(out),'--result-filename','benchmark.json']
        (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n',encoding='utf-8')
        snap(out,'before');started=time.time()
        remaining=datetime.fromisoformat(a.stop_at).timestamp()-time.time()
        if remaining<=0:raise TimeoutError('Release deadline')
        with (out/'benchmark.log').open('wb') as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        (out/'wait.json').write_text(json.dumps({'started':started,'ended':time.time(),
            'returncode':result.returncode,'real_wait':True})+'\n',encoding='utf-8')
        snap(out,'after');result.check_returncode()
        raw=json.loads((out/'benchmark.json').read_text(encoding='utf-8'))
        assert raw['completed']==manifest['sessions'] and raw['failed']==0
        assert all(n==manifest['input_tokens'] for n in raw['input_lens'])
        assert all(n==manifest['output_tokens'] for n in raw['output_lens'])
        assert len(raw['input_lens'])==len(raw['output_lens'])==manifest['sessions']
        assert len(raw['errors'])==len(raw['generated_texts'])==manifest['sessions']
        assert all(not error for error in raw['errors'])


if __name__=='__main__':main()
