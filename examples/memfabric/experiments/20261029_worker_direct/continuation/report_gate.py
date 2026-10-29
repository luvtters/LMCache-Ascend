"""Read-only completion gate: a successful command is not a completed comparison."""
import argparse
import json
import math
from pathlib import Path


def check(report, expected_groups, repeats):
    reasons=[]
    if report.get('status')!='COMPLETE':reasons.append('summary_not_COMPLETE')
    for key in ('missing','invalid','unresolved_monitor_observations'):
        if report.get(key)!=[]:reasons.append(key+'_not_explicitly_empty')
    groups=report.get('groups')
    if not isinstance(groups,list) or len(groups)!=expected_groups:
        return reasons+['unexpected_group_count']
    conditions={}
    for row in groups:
        if not isinstance(row,dict):
            reasons.append('invalid_group');continue
        key=(row.get('family'),row.get('input_tokens'),row.get('concurrency'))
        conditions.setdefault(key,[]).append(row.get('backend'))
        rounds=row.get('rounds',[])
        if (row.get('all_checks_passed') is not True or not isinstance(rounds,list)
                or any(type(value)is not int for value in rounds)
                or len(set(rounds))<repeats):reasons.append('group_not_admitted_or_insufficient_repeats')
        for metric in ('mean_ttft_ms','pooled_output_tps'):
            value=row.get(metric)
            if (type(value) not in (int,float) or not math.isfinite(value) or value<=0):
                reasons.append('invalid_'+metric)
    if any(sorted(backends)!=list('ABC') for backends in conditions.values()):
        reasons.append('condition_missing_unique_ABC')
    return sorted(set(reasons))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--groups',type=int,choices=(3,9),required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    try:
        report=json.loads(args.summary.read_text(encoding='utf-8'))
        reasons=check(report,args.groups,args.repeats)
    except (OSError,ValueError,TypeError,AttributeError) as error:
        report={};reasons=['summary_unreadable_or_invalid:'+type(error).__name__]
    print(json.dumps(dict(gate_status='NOT_REPORTABLE' if reasons else 'PASS',
                          report_status=report.get('status'),reasons=reasons)))
    return 2 if reasons else 0


if __name__=='__main__':raise SystemExit(main())
