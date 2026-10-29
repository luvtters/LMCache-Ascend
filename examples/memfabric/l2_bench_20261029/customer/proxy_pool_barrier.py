"""Experiment-only C1 proxy: drain P's L1/store work before issuing D."""
import asyncio
import importlib.util
import json
import os
import pathlib
import sys
import threading
import time
import httpx

def drained(status):
    manager=status.get('storage_manager',{})
    l1=manager.get('l1_manager',{})
    store=manager.get('store_controller',{})
    prefetch=manager.get('prefetch_controller',{})
    fields={**{'l1.'+k:l1.get(k) for k in ('total_object_count','memory_used_bytes','read_locked_count','write_locked_count','temporary_count','staging_object_count','staging_bytes')},
            **{'store.'+k:store.get(k) for k in ('pending_keys_count','in_flight_task_count','deferred_l1_deletion_count')},
            'prefetch.in_flight_request_count':prefetch.get('in_flight_request_count')}
    return status.get('is_healthy') is True and all(type(value) in (int,float) and value==0 for value in fields.values()),fields

async def pool_barrier():
    started=time.monotonic();polls=0
    async with httpx.AsyncClient(timeout=3) as client:
        while time.monotonic()-started<20:
            response=await client.get('http://127.0.0.1:28185/status');response.raise_for_status()
            ok,fields=drained(response.json());polls+=1
            if ok:
                print('L2_POOL_BARRIER '+json.dumps({'polls':polls,'wait_ms':(time.monotonic()-started)*1000,'fields':fields}),flush=True)
                return
            await asyncio.sleep(.005)
    raise TimeoutError('P pool publication/L1 drain did not complete; D was not sent')

def main():
    source=os.environ.get('LMCACHE_SOURCE')
    if not source:
        raise RuntimeError('Set LMCACHE_SOURCE to the compatible LMCache core checkout')
    path=pathlib.Path(source)/'examples/disagg_prefill_mp/disagg_proxy_server.py'
    spec=importlib.util.spec_from_file_location('compatible_disagg_proxy',path)
    proxy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=proxy;spec.loader.exec_module(proxy)
    # Both calls happen after the original handler awaits its P store event.
    original_stream=proxy.stream_service_response
    original_decoder=proxy.send_request_to_decoder
    async def strict_stream(client,endpoint,req_data):
        await pool_barrier()
        async for chunk in original_stream(client,endpoint,req_data):yield chunk
    async def strict_decoder(client,endpoint,req_data):
        await pool_barrier()
        return await original_decoder(client,endpoint,req_data)
    proxy.stream_service_response=strict_stream
    proxy.send_request_to_decoder=strict_decoder
    proxy.global_args=proxy.parse_args()
    import uvicorn
    threading.Thread(target=proxy.run_telemetry_server,args=(proxy.global_args.host,proxy.global_args.telemetry_port),daemon=True).start()
    uvicorn.run(proxy.app,host=proxy.global_args.host,port=proxy.global_args.port)

if __name__=='__main__':main()
