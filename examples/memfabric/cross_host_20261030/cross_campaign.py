"""Persistent two-host acceptance queue. Raw child evidence lives on each host."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import time
import traceback
import urllib.request
import urllib.error

ROOT = Path('/workspace/d00896497/crosshost_20261030')
PY = '/workspace/d00896497/venv/bin/python'
HOSTS = {'137':'192.168.0.223', '12':'192.168.0.50'}
CONTROL_PORT = 29870
L2 = ROOT/'src/ascend_l2/examples/memfabric/cross_host_20261030'
EXCLUSIVE_UNTIL = datetime.fromisoformat('2026-10-30T06:00:00+08:00').timestamp()
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class CampaignHalted(RuntimeError):
    pass


class CleanupUnknown(RuntimeError):
    pass


def check_policies():
    # Both hosts must be reachable before every spawn and during readiness.
    for host in HOSTS:
        try:
            policy = api(host, '/policy', timeout=5)
        except Exception as exc:
            raise CampaignHalted('Supervisor unreachable on '+host) from exc
        if policy['halted']:
            raise CampaignHalted('Supervisor halt on '+host)


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(body, indent=2), encoding='utf-8')
    temporary.replace(path)


def api(host, path, body=None, timeout=145):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f'http://{HOSTS[host]}:{CONTROL_PORT}{path}', data=data,
                                    headers={'Content-Type':'application/json'})
    with OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def start(host, ident, argv, env=None, persistent=False, active=None, evidence=None):
    check_policies()
    command=['bash','-c','source /workspace/d00896497/env_crosshost.sh; exec "$@"','cross-env',*map(str,argv)]
    request=dict(id=ident,argv=command,cwd=str(ROOT),env=env or {},persistent=persistent)
    # Record the intent before the network request: a lost response may still
    # have launched a real process, which must remain in the cleanup set.
    if active is not None:
        active.append((host,ident))
    if evidence is not None:
        save(evidence/ident/'start_intent.json',dict(host=host,request=request,time=time.time()))
    try:
        result=api(host,'/start',request)
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            # A definitive rejection did not launch our request. In particular,
            # never adopt or stop an older task with the rejected duplicate ID.
            if active is not None:
                active.remove((host,ident))
            if evidence is not None:
                save(evidence/ident/'start_rejected.json',dict(status=exc.code,error=str(exc)))
            raise
        raise CleanupUnknown('Start HTTP failure; do not resubmit '+ident) from exc
    except Exception as exc:
        try:
            result=api(host,'/tasks/'+ident,timeout=5)
        except Exception as lookup:
            raise CleanupUnknown('Start uncertain; never resubmit '+ident+': '+repr(lookup)) from exc
        if evidence is not None:
            save(evidence/ident/'start_response_recovered.json',dict(task=result,error=repr(exc)))
    if evidence is not None:
        save(evidence/ident/'started_task.raw.json',result)
    return result


def wait(host, ident, timeout, evidence):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        check_policies()
        status=api(host,'/tasks/'+ident)
        save(evidence/status['id']/'latest_task.raw.json',status)
        if status['state']=='EXITED':
            return status
        time.sleep(1)
    raise TimeoutError(ident)


def stop(host, ident, evidence, grace=120):
    try:
        result=api(host,'/stop',dict(id=ident,grace=grace,reason='controlled_case_shutdown'))
    except Exception as exc:
        # A response loss is not a second stop/start request. Read the exact task.
        try:
            result=api(host,'/tasks/'+ident,timeout=5)
        except Exception as lookup:
            save(evidence/ident/'cleanup_unknown.json',dict(error=repr(exc),lookup_error=repr(lookup)))
            raise CleanupUnknown(ident+' stop uncertain') from exc
    save(evidence/ident/'stopped_task.raw.json',result)
    if result.get('state')!='EXITED' or not result.get('real_wait') or 'returncode' not in result:
        raise CleanupUnknown(ident+' missing EXITED/real wait')
    return result


def cleanup(active, evidence, grace_for, direct=False):
    errors=[]; unknown=False
    for host,ident in active:
        try:
            result=stop(host,ident,evidence,grace_for(ident))
            rc=result['returncode']
            controlled=(result.get('planned_stop') and
                        result.get('stop_reason')=='controlled_case_shutdown' and
                        any(s.get('delivered') and s.get('reason')=='controlled_case_shutdown'
                            for s in result.get('stop_signals',[])))
            valid=(rc==0 and (not result.get('persistent') or controlled))
            if direct and rc in (130,-2) and controlled:
                valid=True
            if not valid:
                errors.append(dict(id=ident,host=host,status='LIFECYCLE_FAILED',task=result))
        except Exception as exc:
            unknown=True
            errors.append(dict(id=ident,host=host,status='CLEANUP_UNKNOWN',error=repr(exc)))
    save(evidence/'cleanup.json',dict(status='CLEANUP_UNKNOWN' if unknown else 'FAILED' if errors else 'PASSED',errors=errors))
    if unknown:
        raise CleanupUnknown('Cleanup uncertainty: '+str(errors))
    if errors:
        raise RuntimeError('Lifecycle failures: '+str(errors))


def ready(host, url, timeout, roles):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        check_policies()
        for h,i in roles:
            if api(h,'/tasks/'+i)['state']=='EXITED':
                raise RuntimeError('Role exited before readiness: '+i)
        try:
            with OPENER.open(url,timeout=2) as response:
                if response.status==200: return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(url)


def l2_case(ident, kind, reverse, tp, evidence, member_host=None, member_device=8, core_path=None, config_overrides=None):
    local,remote=('12','137') if reverse else ('137','12')
    remote=member_host or remote
    cfg=dict(run_root=str(ROOT/'runs'),run_id=ident,core_path=str(core_path or ROOT/'src/core_compat'),
             ascend_path=str(ROOT/'src/ascend_l2'),python=PY,env_file='/workspace/d00896497/env_crosshost.sh',
             communication_ip=HOSTS[local],store_port=28580,pool_gb=1 if tp<4 else 2,
             slot_size_mb=32,tp_size=tp,driver_devices=[8],driver_bm_device=0,
             service_bm_device=2,local_devices=dict(p=[8],d=[10],service=[8,10,12]),
             model_path='/workspace/d00896497/models/Qwen3-8B',stop_timeout=18,shutdown_timeout=90)
    if tp==2:
        cfg.update(local_devices=dict(p=[8,9],d=[10,11],service=[8,9,10,11,12]),service_bm_device=4)
    if tp==4:
        cfg.update(local_devices=dict(p=[2,3,4,5],d=[8,9,10,11],service=[2,3,4,5,8,9,10,11,12]),service_bm_device=8)
    cfg.update(config_overrides or {})
    config=api(local,'/config',dict(path='configs/'+ident+'.json',data=cfg))['path']
    member=ident+'-member'; driver=ident+'-driver'
    common={'PYTHONPATH':cfg['ascend_path']+':'+cfg['core_path']}
    member_env=dict(common,ASCEND_RT_VISIBLE_DEVICES=str(member_device),TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    member_argv=[PY,'-u',Path(cfg['ascend_path'])/'examples/memfabric/pool_member.py','--device','0','--rank','1','--world-size','2','--pool-gb',str(cfg['pool_gb']),'--store-url',f"tcp://{HOSTS[local]}:{cfg['store_port']}"]
    if cfg.get('pool_memory_type'):
        member_argv += ['--pool-memory-type',cfg['pool_memory_type']]
    started=[]
    try:
        start(remote,member,member_argv,member_env,active=started,evidence=evidence)
        argv=[PY,'-u',Path(cfg['ascend_path'])/'examples/memfabric/cross_host_20261030'/('l2_driver.py' if kind=='smoke' else 'l2_model.py'),'--config',config]
        env=dict(common)
        if kind=='smoke': env.update(ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,cfg['driver_devices'])),TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        start(local,driver,argv,env,active=started,evidence=evidence)
        result=wait(local,driver,240 if kind=='smoke' else 900,evidence)
        if result['returncode']!=0: raise RuntimeError('L2 role failed '+str(result['returncode']))
        member_result=wait(remote,member,35,evidence)
        if member_result['returncode']!=0: raise RuntimeError('L2 member failed '+str(member_result['returncode']))
        return dict(function_and_exit='PASSED',config=cfg)
    finally:
        cleanup(reversed(started),evidence,lambda i:120 if i==driver else 20)


def direct_case(ident, stage, evidence, tp=1, topology='T3', role_root=None, p_devices=None, d_devices=None, pool_device=None, pool_memory_type='host', skip_blocks=0, followup_argv=None):
    role_root=Path(role_root) if role_root else ROOT if tp==1 else ROOT/'tp_candidate'
    runner=role_root/'src/ascend_direct/examples/memfabric/cross_host_role.py'
    spec=importlib.util.spec_from_file_location('role_'+ident,runner)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    parser=argparse.Namespace(root=str(role_root),python=PY,topology=topology,stage=stage,
          session_id=ident,model='/workspace/d00896497/models/Qwen3-8B',pool_gb=1 if tp==1 else 2,
          skip_blocks=skip_blocks,tp_size=tp,p_devices=p_devices,d_devices=d_devices,pool_device=pool_device,pool_memory_type=pool_memory_type)
    mapping=module.TOPOLOGIES[topology]
    roles=['coordinator'] if stage!='inference' else ['trace'+h for h in sorted({v[0] for v in mapping.values()})]
    roles+=['service','p','d']
    if stage=='inference': roles+=['proxy','gate']
    active=[]; plans={}
    try:
        for role in roles:
            parser.role=role
            plan=module.build_plan(parser); plans[role]=plan
            host=plan['host']; task=ident+'-'+role
            argv=[PY,'-u',runner,'--root',role_root,'--topology',topology,'--stage',stage,'--session-id',ident,'--role',role,'--run']
            if tp>1: argv += ['--tp-size',str(tp),'--pool-gb','2']
            if p_devices is not None: argv += ['--p-devices',','.join(map(str,p_devices))]
            if d_devices is not None: argv += ['--d-devices',','.join(map(str,d_devices))]
            if pool_device is not None: argv += ['--pool-device',str(pool_device)]
            if pool_memory_type!='host': argv += ['--pool-memory-type',pool_memory_type]
            if skip_blocks: argv += ['--skip-blocks',str(skip_blocks)]
            if plan['physical_device'] is not None: argv+=['--allow-npu']
            if role=='proxy':
                for engine in ('p','d'):
                    h=plans[engine]['host']
                    ready(h,f'http://{HOSTS[h]}:{28401 if engine=="p" else 28402}/health',600,active)
                h=plans['service']['host']
                ready(h,f'http://{HOSTS[h]}:28195/healthcheck',60,active)
            persistent=(role=='coordinator' or (role=='service' and stage=='context') or
                        (role in ('service','p','d','proxy','trace137','trace12') and stage=='inference'))
            start(host,task,argv,persistent=persistent,active=active,evidence=evidence)
            if role in ('coordinator','trace137','trace12'):
                time.sleep(.7)
            if role=='proxy': ready('137','http://192.168.0.223:28400/v1/models',30,active)
        save(evidence/'role_plans.json',plans)
        if stage=='inference':
            result=wait('137',ident+'-gate',600,evidence)
            if result['returncode']!=0: raise RuntimeError('Direct gate failed')
            if followup_argv:
                start('137',ident+'-followups',followup_argv,active=active,evidence=evidence)
                followup=wait('137',ident+'-followups',600,evidence)
                if followup['returncode']!=0:raise RuntimeError('Direct follow-up cases failed')
        else:
            for role in ('p','d'):
                result=wait(plans[role]['host'],ident+'-'+role,420,evidence)
                if result['returncode']!=0: raise RuntimeError('Direct '+role+' failed')
        function_result=dict(function='PASSED',lifecycle='CONTROLLED_WAIT_CODES_RECORDED')
    finally:
        priority={'gate':0,'followups':0,'proxy':1,'d':2,'p':3,'service':4,'trace137':5,'trace12':5,'coordinator':5}
        cleanup(sorted(active,key=lambda pair:priority[pair[1].removeprefix(ident+'-')]),evidence,lambda i:60,direct=True)
    return function_result


QUEUE=[('l2-smoke-f','l2','smoke',False,1),('l2-smoke-r','l2','smoke',True,1),
           ('l2-model-tp1','l2','model',False,1),('direct-primitive','direct','primitive',False,1),
           ('direct-context','direct','context',False,1),('direct-inference-tp1','direct','inference',False,1),
           ('l2-model-tp2','l2','model',False,2),('l2-model-tp4','l2','model',False,4),
           ('direct-inference-tp2','direct','inference',False,2)]
PREREQUISITES={
    'l2-model-tp1':['l2-smoke-f','l2-smoke-r'],
    'l2-model-tp2':['l2-model-tp1'],
    'l2-model-tp4':['l2-model-tp2'],
    'direct-context':['direct-primitive'],
    'direct-inference-tp1':['direct-primitive','direct-context'],
    'direct-inference-tp2':['direct-inference-tp1'],
}


def valid_name(value):
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise argparse.ArgumentTypeError('Use only letters, digits, underscore and hyphen')
    return value


def run_queue(queue,campaign,attempt,prior=None):
    campaign.mkdir(exist_ok=False,parents=True)
    passed=set(prior or [])
    state=dict(started=time.time(),pending=queue,attempts=[],status='RUNNING')
    save(campaign/'ready.json',dict(pid=__import__('os').getpid(),started=time.time(),queue=queue,prior_passed=sorted(passed)))
    for index,(prefix,route,kind,reverse,tp) in enumerate(queue):
        try:
            check_policies()
        except CampaignHalted as exc:
            state.update(status='HALTED',error=str(exc),ended=time.time())
            save(campaign/'status.json',state); return state
        ident=campaign.name+'-'+prefix+'-'+attempt
        evidence=campaign/ident; evidence.mkdir()
        missing=[p for p in PREREQUISITES.get(prefix,[]) if p not in passed]
        record=dict(id=ident,stage=prefix,started=time.time(),route=route,tp=tp,status='RUNNING')
        state.update(current=ident,pending=queue[index+1:]); state['attempts'].append(record)
        save(campaign/'status.json',state)
        if missing:
            record.update(status='BLOCKED_PREREQUISITE',missing=missing)
        else:
            try:
                result=l2_case(ident,kind,reverse,tp,evidence) if route=='l2' else direct_case(ident,kind,evidence,tp)
                record.update(status='FUNCTION_COMPLETE',result=result)
                passed.add(prefix)
            except BaseException as exc:
                record.update(status='CLEANUP_UNKNOWN' if isinstance(exc,CleanupUnknown) else 'FAILED',traceback=traceback.format_exc())
                if isinstance(exc,(CampaignHalted,CleanupUnknown)) or time.time()>=EXCLUSIVE_UNTIL:
                    halt_errors=[]
                    for host in HOSTS:
                        try:
                            api(host,'/halt',dict(reason='campaign_halt_or_uncertain_cleanup' if time.time()<EXCLUSIVE_UNTIL else 'campaign_unexpected_failure_after_0600'),timeout=5)
                        except Exception as halt_exc:
                            halt_errors.append(dict(host=host,error=repr(halt_exc)))
                    state.update(status='CLEANUP_UNKNOWN' if isinstance(exc,CleanupUnknown) else 'HALTED',halt_errors=halt_errors)
        record['ended']=time.time(); save(evidence/'case_result.json',record); save(campaign/'status.json',state)
        if state['status']!='RUNNING':
            state['ended']=time.time(); save(campaign/'status.json',state); return state
    state.update(status='QUEUE_COMPLETE' if all(r['status']=='FUNCTION_COMPLETE' for r in state['attempts']) else 'NEEDS_REPAIR',ended=time.time())
    save(campaign/'status.json',state)
    return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--campaign-id',type=valid_name,default='campaign')
    parser.add_argument('--attempt',type=valid_name,default='a1')
    parser.add_argument('--select',help='Comma-separated stage names; defaults to all nine')
    parser.add_argument('--prior-status',type=Path,help='Verified prior campaign status for prerequisite passes')
    args=parser.parse_args()
    selected=args.select.split(',') if args.select else [q[0] for q in QUEUE]
    if len(set(selected))!=len(selected) or any(p not in {q[0] for q in QUEUE} for p in selected):
        parser.error('Unknown or duplicate selected stage')
    queue=[next(q for q in QUEUE if q[0]==name) for name in selected]
    prior=[]
    if args.prior_status:
        previous=json.loads(args.prior_status.read_text(encoding='utf-8'))
        prior=[r['stage'] for r in previous['attempts'] if r['status']=='FUNCTION_COMPLETE']
    if not args.run:
        print(json.dumps(dict(queue=queue,prerequisites=PREREQUISITES,prior_passed=prior,campaign=str(ROOT/args.campaign_id),attempt=args.attempt,exclusive_until=EXCLUSIVE_UNTIL),indent=2)); return
    state=run_queue(queue,ROOT/args.campaign_id,args.attempt,prior)
    raise SystemExit(0 if state['status']=='QUEUE_COMPLETE' else 1)


if __name__=='__main__': main()
