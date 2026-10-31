"""CPU checks for remote followup trace cutoffs and TP count contracts."""
import contextlib
import io
import unittest
import followup_direct_requests as f


class Checks(unittest.TestCase):
    def test_local_default_and_paired_remote_cli(self):
        base=['--gate-dir','unused-gate','--output','unused-output']
        self.assertIsNotNone(f.arguments(base+['--trace-dir','local-trace']).trace_dir)
        args=f.arguments(base+['--p-trace-url','http://p:29596/snapshot',
            '--d-trace-url','http://d:29596/snapshot','--session-id','run'])
        self.assertEqual(args.session_id,'run')
        for extra in ([], ['--p-trace-url','http://p:29596/snapshot'],
                      ['--p-trace-url','http://p:29596/snapshot','--d-trace-url','http://d:29596/snapshot']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                f.arguments(base+extra)

    def test_remote_snapshots_keep_independent_clocks_and_reject_identity_changes(self):
        old_p=dict(role='P',host_id='host-P',session_id='run',mono_ns=900,kind='npu_scratch',phase='complete')
        old_d=dict(role='D',host_id='host-D',session_id='run',mono_ns=5,kind='npu_scratch',phase='complete')
        snapshots={'p':dict(host_id='host-P',run_id='run',mono_ns=1000,records=[old_p]),
                   'd':dict(host_id='host-D',run_id='run',mono_ns=10,records=[old_d])}
        records,base,_=f.collect_remote_snapshots(snapshots.__getitem__,{'P':'p','D':'d'},'run')
        self.assertEqual(len(records),2)  # Scratch stays available to verify_d_trace.
        snapshots['p']=dict(snapshots['p'],mono_ns=1100)
        event=dict(old_d,mono_ns=20,kind='bm_copy')
        snapshots['d']=dict(snapshots['d'],mono_ns=30,records=[old_d,event])
        records,_,_=f.collect_remote_snapshots(snapshots.__getitem__,{'P':'p','D':'d'},'run',base)
        self.assertEqual(f.active_trace(records,base),[event])
        self.assertEqual(f.assert_no_new_p_traffic(records,base),0)
        for bad in (dict(snapshots['d'],run_id='other'),dict(snapshots['d'],host_id='changed')):
            with self.assertRaises(AssertionError):
                f.validate_snapshot(bad,'D','run',base['D'])
        bad=dict(snapshots['d'],records=[dict(event,session_id='other')])
        with self.assertRaises(AssertionError):
            f.validate_snapshot(bad,'D','run',base['D'])
        with self.assertRaises(AssertionError):
            f.assert_no_new_p_traffic(records+[dict(old_p,mono_ns=1001,kind='bm_copy')],base)

    def test_tp_counts_do_not_multiply_total_bytes(self):
        for tp in (1,2,4,8):
            expected=f.expected_followup_trace(1373,128,2,tp,180<<20)
            self.assertEqual(expected['copies'],10*2*tp)
            self.assertEqual(expected['bytes'],360<<20)
            self.assertEqual(expected['per_rank_copies'],20)
            self.assertEqual(expected['per_rank_bytes'],(360<<20)//tp)


if __name__=='__main__':
    unittest.main()
