"""Opt-in CPU observer: normal gate first, idle owner leaves, D must fail closed.

This does not start engines or copy KV. The outer supervisor records every real
role wait; a successful negative observation does not make an engine exit normal.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def terminal_failure(status, error_type, typed_peer_error):
    if not typed_peer_error or error_type in ('TimeoutError', 'socket.timeout'):
        return False
    return (status is not None and 500 <= status < 600) or error_type in (
        'RemoteDisconnected', 'ConnectionResetError', 'ConnectionRefusedError')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root',type=Path,required=True)
    parser.add_argument('--member-task',required=True)
    parser.add_argument('--control',default='http://192.168.0.223:29870')
    parser.add_argument('--service',default='http://127.0.0.1:28080')
    parser.add_argument('--decoder',default='http://127.0.0.1:28002')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print(json.dumps(dict(status='PLAN_ONLY',member_task=args.member_task)));return
    assert args.member_task==args.case_root.name+'-member','Only this exact fresh group owner may be stopped'
    out=args.case_root/'peer_left_probe';out.mkdir(exist_ok=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def http(url,label,body=None,timeout=20,request_id=None):
        folder=out/label;folder.mkdir(parents=True)
        payload=None if body is None else json.dumps(body).encode()
        if payload is not None:(folder/'request.body.bin').write_bytes(payload)
        headers={'Content-Type':'application/json'}
        if request_id:headers['X-Request-Id']=request_id
        request=urllib.request.Request(url,data=payload,headers=headers)
        record=dict(url=url,started_mono_ns=time.monotonic_ns())
        try:
            try:response=opener.open(request,timeout=timeout)
            except urllib.error.HTTPError as error:response=error
            with response:
                raw=response.read();(folder/'response.body.bin').write_bytes(raw)
                record.update(status=response.status,headers=list(response.headers.items()))
            try:value=json.loads(raw)
            except ValueError:value=None
        except Exception as error:
            reason=getattr(error,'reason',error)
            record.update(status=None,error_type=type(reason).__name__,error=repr(error));value=None
        record['ended_mono_ns']=time.monotonic_ns()
        (folder/'http.raw.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
        return value,record
    def pool(status):
        matches=[r for r in status['storage_manager']['l2_adapters'] if r.get('protocol')=='SDMA']
        assert len(matches)==1,matches
        return matches[0]
    before,_=http(args.service+'/status','before_status')
    bp=pool(before);assert bp['is_healthy'] and bp['total_pins']==0,bp
    http(args.decoder+'/metrics','before_d_metrics')
    offset=(args.case_root/'d.log').stat().st_size
    service_offset=(args.case_root/'service.log').stat().st_size
    stopped,receipt=http(args.control+'/stop','owner_control_stop',
        dict(id=args.member_task,grace=20,reason='intentional_idle_pool_leave_fault_probe'),timeout=30)
    assert receipt['status']==200 and stopped['id']==args.member_task and stopped['state']=='EXITED' and stopped['real_wait'] and stopped['returncode']==0,stopped
    deadline=time.monotonic()+10;attempt=0
    while True:
        status,_=http(args.service+'/status','owner_left_status_'+str(attempt))
        observed=pool(status)
        if not observed['is_healthy']:break
        if time.monotonic()>deadline:raise TimeoutError('Pool leave did not become observable')
        attempt+=1;time.sleep(.1)
    assert 'peer' in observed['last_error'].lower() and 'left' in observed['last_error'].lower(),observed
    body=json.loads((args.case_root/'gate/request.json').read_text())
    # A new first chunk guarantees this valid long request cannot be satisfied
    # from the previous request's L1 objects if the engine happens to retain any.
    body['messages'][0]['content']='New failure propagation probe. Return exactly ORCHID.'
    body['max_tokens']=4
    request_id=args.case_root.name+'-peer-left-query'
    _,response=http(args.decoder+'/v1/chat/completions','d_after_peer_leave',body,timeout=25,request_id=request_id)
    deadline=time.monotonic()+5
    while True:
        with (args.case_root/'d.log').open('rb') as stream:
            stream.seek(offset);raw=stream.read()
        text=raw.decode(errors='replace')
        typed=any('RemoteRequestError' in line and 'peer' in line.lower() and 'left' in line.lower()
                  for line in text.splitlines())
        if typed or time.monotonic()>deadline:break
        time.sleep(.1)
    (out/'d_after_peer_leave.raw.log').write_bytes(raw)
    with (args.case_root/'service.log').open('rb') as stream:
        stream.seek(service_offset);service_raw=stream.read()
    (out/'service_after_peer_leave.raw.log').write_bytes(service_raw)
    after,_=http(args.service+'/status','after_status')
    ap=pool(after)
    http(args.decoder+'/metrics','after_d_metrics')
    assert not ap['is_healthy'] and ap['total_pins']==0,ap
    assert ap['load_bytes']==bp['load_bytes'],(bp,ap)
    assert ap['store_bytes']==bp['store_bytes'],(bp,ap)
    result=dict(http_status=response['status'],error_type=response.get('error_type'),
        typed_peer_error_in_original_d_log=typed,load_bytes_before=bp['load_bytes'],
        load_bytes_after=ap['load_bytes'],pins=ap['total_pins'],
        store_bytes_before=bp['store_bytes'],store_bytes_after=ap['store_bytes'],
        request_id=request_id,request_id_in_new_logs=request_id in text or request_id in service_raw.decode(errors='replace'),
        lifecycle='OUTER_SUPERVISOR_RECORDS_REAL_EXIT_CODES')
    (out/'observation.derived.json').write_text(json.dumps(result,indent=2))
    assert terminal_failure(response['status'],response.get('error_type'),typed),result
    print(json.dumps(dict(observed='EXPLICIT_PEER_LEFT_FAILURE_NO_NEW_BM_READ',**result)),flush=True)


if __name__=='__main__':main()
