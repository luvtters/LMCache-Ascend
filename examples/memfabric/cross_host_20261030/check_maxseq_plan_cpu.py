"""Check the optional experiment argument without HTTP, models or SDK imports."""
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('campaign',Path(__file__).with_name('cross_campaign.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);role=root/'src/ascend_direct/examples/memfabric/cross_host_role.py';role.parent.mkdir(parents=True)
    role.write_text("TOPOLOGIES={'T3':{'p':('137',2),'d':('137',10),'service':('12',14)}}\n"
        "def build_plan(args):\n return dict(host='12' if args.role in ('service','trace12') else '137',physical_device=None,max_num_seqs=args.max_num_seqs)\n",encoding='utf8')
    for value in (1,2):
        calls=[];out=root/str(value);out.mkdir()
        with patch.object(c,'start',side_effect=lambda host,task,argv,**kw:calls.append(argv)), \
             patch.object(c,'wait',return_value={'returncode':0}),patch.object(c,'ready'),patch.object(c,'cleanup'),patch.object(c.time,'sleep'):
            c.direct_case('cpu','inference',out,4,role_root=root,max_num_seqs=value)
        assert len(calls)==7
        for argv in calls:
            assert ('--max-num-seqs' in argv)==(value==2)
            if value==2:assert argv[argv.index('--max-num-seqs')+1]=='2'
    try:c.direct_case('invalid','inference',root,max_num_seqs=0)
    except ValueError:pass
    else:raise AssertionError('Invalid max_num_seqs accepted')
print('CPU_ONLY PASS: default argv unchanged, explicit two-sequence role args, invalid value rejected before any remote operation')
