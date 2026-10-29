"""Small real HTTP/child fault checks; no accelerator modules or services."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import urllib.request


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


c=load('cross_campaign'); h=load('host_control')
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary); supervisors={}; servers={}; threads=[]
    for host in c.HOSTS:
        supervisor=h.Supervisor(root/host,time.time()+60)
        server=h.ThreadingHTTPServer(('127.0.0.1',0),h.handler(supervisor))
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        supervisors[host]=supervisor; servers[host]=server; threads.append(thread)
    def api(host,path,body=None,timeout=145):
        data=None if body is None else json.dumps(body).encode()
        request=urllib.request.Request('http://127.0.0.1:'+str(servers[host].server_port)+path,data=data,
                                       headers={'Content-Type':'application/json'})
        with c.OPENER.open(request,timeout=min(timeout,5)) as response:
            return json.load(response)
    c.api=api
    def spawn(host,ident,code,persistent=False):
        return api(host,'/start',dict(id=ident,argv=[sys.executable,'-c',code],persistent=persistent))
    try:
        # HTTP start succeeds but its response is lost: recover exact task once.
        submitted=[]
        def lost_response(host,path,body=None,timeout=145):
            if path=='/start':
                submitted.append(body['id'])
                body=dict(body,argv=[sys.executable,'-c','pass'],cwd=str(supervisors[host].root))
                api(host,path,body,timeout)
                raise TimeoutError('injected response loss after real start')
            return api(host,path,body,timeout)
        c.api=lost_response; active=[]; evidence=root/'recovery'
        result=c.start('137','response-lost',[sys.executable,'-c','pass'],active=active,evidence=evidence)
        assert submitted==['response-lost'] and active==[('137','response-lost')]
        c.api=api
        assert c.wait('137','response-lost',5,evidence)['returncode']==0
        c.cleanup(active,evidence,lambda _:0)
        # Cleanup must continue after an unknown task and reject natural 130/0.
        spawn('12','natural130','raise SystemExit(130)')
        spawn('137','persistent-zero','pass',True)
        c.wait('12','natural130',5,root/'natural')
        c.wait('137','persistent-zero',5,root/'natural')
        try:
            c.cleanup([('137','missing'),('12','natural130'),('137','persistent-zero')],root/'cleanup',lambda _:0,direct=True)
        except c.CleanupUnknown:
            pass
        else:
            raise AssertionError('uncertain stop accepted')
        report=json.loads((root/'cleanup/cleanup.json').read_text())
        assert len(report['errors'])==3 and report['status']=='CLEANUP_UNKNOWN'
        assert (root/'cleanup/natural130/stopped_task.raw.json').exists()
        assert (root/'cleanup/persistent-zero/stopped_task.raw.json').exists()
        # A real failed raw L2 child blocks model stages but allows direct raw.
        called=[]
        def l2(ident,kind,reverse,tp,evidence):
            called.append(ident)
            assert kind=='smoke'
            spawn('137',ident,'raise SystemExit(7)')
            assert c.wait('137',ident,5,evidence)['returncode']==7
            c.cleanup([('137',ident)],evidence,lambda _:0)
        def direct(ident,kind,evidence,tp):
            called.append(ident)
            assert kind=='primitive'
            spawn('12',ident,'pass')
            assert c.wait('12',ident,5,evidence)['returncode']==0
            c.cleanup([('12',ident)],evidence,lambda _:0,direct=True)
            return {'function':'PASSED'}
        c.l2_case=l2; c.direct_case=direct
        selected=[q for q in c.QUEUE if q[0] in ('l2-smoke-f','l2-model-tp1','direct-primitive','direct-inference-tp1','l2-model-tp2')]
        state=c.run_queue(selected,root/'queue','a1')
        assert state['status']=='NEEDS_REPAIR' and len(called)==2
        assert [r['status'] for r in state['attempts']]==['FAILED','BLOCKED_PREREQUISITE','FUNCTION_COMPLETE','BLOCKED_PREREQUISITE','BLOCKED_PREREQUISITE']
        # A peer halt before 06:00 blocks both start and readiness immediately.
        api('12','/halt',{'reason':'cpu_check'})
        for operation in (lambda:c.start('137','forbidden',['unused']),
                          lambda:c.ready('137','http://127.0.0.1:1',1,[])):
            try: operation()
            except c.CampaignHalted: pass
            else: raise AssertionError('global halt ignored')
        assert 'forbidden' not in supervisors['137'].tasks
        assert c.run_queue([c.QUEUE[0]],root/'halted-queue','a1')['status']=='HALTED'
    finally:
        for supervisor in supervisors.values():
            for ident in list(supervisor.tasks):
                supervisor.stop(ident,0,'cpu_check_cleanup')
        for server in servers.values():
            server.shutdown(); server.server_close()
print('CPU_ONLY PASS: real HTTP start-response recovery, all-role cleanup, raw wait, lifecycle rejection, prerequisites, global halt')
