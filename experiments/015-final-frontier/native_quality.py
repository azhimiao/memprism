"""Count-independent native scoring with bounded actual missing requests.

Fixed complete denominator, actual exact-request cache verification, no retries,
durable full prompt/response ledger. Uses public native contracts unchanged.
Every new request respects caps; any failure stops and retains not_run rows.
This runner never loads gold while generating service evidence.
"""
from pathlib import Path
import sys
import argparse
import json
import math
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'experiments/011-pareto-iteration')]
from pareto_common import write,sha,digest,snapshot,evaluator_sources,check_sources,append
from pareto_quality import seed_verified_cache,cache_directories
from offline_eval.public_contracts import load_contract
from offline_eval.bounded_gateway import BoundedGateway
from offline_eval.chat import token_count
from reference_request_broker import SharedRequestBroker
from run_reference_evaluator import complete
import source_gate


def validate_price_contract(gateway,max_usd):
    if gateway.get('model')!='openai/gpt-4o-mini':raise ValueError('frozen native model changed')
    for key,expected in [('input_usd_per_million',.15),('output_usd_per_million',.6)]:
        value=gateway.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value!=expected:
            raise ValueError('frozen native price contract changed')
    if isinstance(max_usd,bool) or not isinstance(max_usd,(int,float)) or not math.isfinite(max_usd) or max_usd<0:
        raise ValueError('invalid native allocation')


def answer_messages(row,case,contract):
    context='\n\n'.join(h['content'] for h in row['hits'])
    if token_count(context)>4000 or len(context)>24000:raise ValueError('frozen Answer context over budget')
    if case['dataset']=='personamem_v1':
        return contract.namespace['official_messages']({'question':case['question'],'all_options':case['all_options'],
            'context_messages':[{'role':'user','content':context}]},'gpt-4o-mini')
    return [{'role':'user','content':contract.namespace['render_answer_prompt']({'question':case['question'],
        'retrieved_context':context,'speaker_1_name':case.get('speaker_1_name','speaker 1'),
        'speaker_2_name':case.get('speaker_2_name','speaker 2')})}]


def preflight(cases_path,labels_path,rows_path,output,profiles,max_usd,closure_path=None):
    if output.exists():raise ValueError('new preflight required')
    source_proof=(source_gate.proof(cases_path,rows_path,profiles,closure_path=closure_path)
                  if closure_path is not None else source_gate.proof(cases_path,rows_path,profiles))
    cases=json.loads(cases_path.read_text(encoding='utf-8'));bycase={r['case_id']:r for r in cases}
    rows=json.loads(rows_path.read_text(encoding='utf-8')) if rows_path.suffix=='.json' else [json.loads(x) for x in rows_path.read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['profile'] in profiles]
    expected={(p,c['case_id']) for p in profiles for c in cases}
    if len(rows)!=len(expected) or {(r['profile'],r['case_id']) for r in rows}!=expected or any(r['status']!='completed' for r in rows):raise ValueError('incomplete retrieval denominator')
    gateway=json.loads((ROOT/'experiments/010-evaluator-freeze/config.json').read_text(encoding='utf-8'))
    validate_price_contract(gateway,max_usd)
    gateway.update(max_usd=max_usd,max_requests=len(rows)*2,per_call_input_tokens=6000,per_call_output_tokens=256,
        default_output_tokens=256,stage_family_max_usd={'reference':0,'evaluation':max_usd})
    cache_cfg={'quality':{'cache':{'roots':['experiments/010-evaluator-freeze/runs','experiments/011-pareto-iteration/runs',
        'experiments/014-confirmation/runs','experiments/015-final-frontier/runs']}}}
    seed=seed_verified_cache(gateway,cache_directories(cache_cfg))
    contracts={d:load_contract(d) for d in {c['dataset'] for c in cases}}
    labels={r['case_id']:r for r in json.loads(labels_path.read_text(encoding='utf-8'))}
    if set(labels)!=set(bycase):raise ValueError('labels denominator mismatch')
    prompts=[];allowance=0;missing_answers=0;judge_ceiling=2048
    for row in rows:
        case=bycase[row['case_id']];messages=answer_messages(row,case,contracts[case['dataset']])
        inputs=sum(token_count(m['content'])+token_count(m['role'])+8 for m in messages)+64
        if inputs>6000:raise ValueError('Answer prompt exceeds input cap')
        req={'model':gateway['model'],'messages':messages,'temperature':0,'max_tokens':256,'stream':False}
        hit=seed['cache'].get(digest(req))
        if hit is None:missing_answers+=1;allowance+=(inputs*gateway['input_usd_per_million']+256*gateway['output_usd_per_million'])/1e6
        if case['dataset']=='locomo':
            if hit:
                answer=hit['choices'][0]['message']['content'].strip()
                judge=contracts['locomo'].namespace['render_accuracy_prompt']({'question':case['question'],'gold_answer':labels[case['case_id']]['answers']},answer)
                judge_req={'model':gateway['model'],'messages':[{'role':'user','content':judge}],'temperature':0,'max_tokens':128,'stream':False}
                if digest(judge_req) not in seed['cache']:
                    count=token_count(judge)+token_count('user')+72
                    if count>judge_ceiling:raise ValueError('cached Answer Judge exceeds preregistered stage cap')
                    allowance+=(count*gateway['input_usd_per_million']+128*gateway['output_usd_per_million'])/1e6
            else:allowance+=(judge_ceiling*gateway['input_usd_per_million']+128*gateway['output_usd_per_million'])/1e6
        prompts.append({'profile':row['profile'],'case_id':row['case_id'],'dataset':case['dataset'],
                        'messages':messages,'request_sha256':digest(req),'reserved_input_tokens':inputs})
    if allowance>max_usd:raise ValueError('request-specific complete budget exceeds new allocation')
    output.mkdir(parents=True)
    write(output/'prompts.json',prompts);write(output/'cache-seed.json',seed)
    closure_sources=[]
    if closure_path is not None:
        import postprocess_closure
        closure_sources=[Path(postprocess_closure.__file__)]
    sources=snapshot(output,evaluator_sources()+[Path(__file__),Path(source_gate.__file__)]+closure_sources+[c.source_path for c in contracts.values()])
    result={'status':'ready','planned':len(rows),'profiles':profiles,'cases_path':str(cases_path.resolve()),
        'labels_path':str(labels_path.resolve()),'rows_path':str(rows_path.resolve()),
        'input_hashes':{str(p.resolve()):sha(p) for p in [cases_path,labels_path,rows_path]},'sources':sources,
        'source_gate':source_proof,
        'prompts_sha256':sha(output/'prompts.json'),'cache_seed_sha256':sha(output/'cache-seed.json'),
        'gateway_config':gateway,'judge_input_cap':judge_ceiling,'answer_input_cap':6000,'wall_seconds':3600,
        'request_specific_worst_catalog_allowance_usd':allowance,'missing_answer_requests':missing_answers,
        'quality_scope':'native unchanged prompts and parsing; no official equivalent or global optimum',
        'allocation':{'combined_user_ceiling_usd':1,'this_run_max_usd':max_usd,'all_caps_before_dispatch':True},
        'preregistered_no_retry':True,'model_calls':0}
    write(output/'preflight.json',result);return result


def execute(preflight_dir,output,cache_only=False):
    plan=json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    if plan['status']!='ready' or output.exists():raise ValueError('invalid execution request')
    if cache_only and (plan['gateway_config']['max_usd']!=0
        or plan['request_specific_worst_catalog_allowance_usd']!=0 or plan['missing_answer_requests']!=0
        or 'postprocess_closure' not in plan['source_gate']):raise ValueError('cache_only_requires_zero_budget_complete_closure')
    source_gate.recheck(plan['source_gate'])
    validate_price_contract(plan['gateway_config'],plan['gateway_config']['max_usd'])
    check_sources(plan['sources'])
    for path,expected in plan['input_hashes'].items():
        if sha(Path(path))!=expected:raise ValueError('frozen input changed')
    if sha(preflight_dir/'prompts.json')!=plan['prompts_sha256'] or sha(preflight_dir/'cache-seed.json')!=plan['cache_seed_sha256']:raise ValueError('frozen prompt/cache changed')
    seed=json.loads((preflight_dir/'cache-seed.json').read_text(encoding='utf-8'));check_sources(seed['source_hashes'])
    cases={r['case_id']:r for r in json.loads(Path(plan['cases_path']).read_text(encoding='utf-8'))}
    labels={r['case_id']:r for r in json.loads(Path(plan['labels_path']).read_text(encoding='utf-8'))}
    prompts=json.loads((preflight_dir/'prompts.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True);write(output/'preflight.json',plan)
    contracts={d:load_contract(d) for d in {c['dataset'] for c in cases.values()}}
    start=time.monotonic()
    gateway_base=BoundedGateway
    if cache_only:
        import threading
        class CacheOnlyGateway(BoundedGateway):
            def __init__(self,config,output):
                # No credentials, HTTP client, sockets or transport are created.
                self.config,self.output=config,Path(output);self.output.mkdir(parents=True)
                self.lock=threading.RLock();self.context={}
                self.records,self.cache=[],{};self.failed_requests=set()
                self.committed_usd=0.;self.stage_costs={};self.halted=None
            def _complete(self,payload):
                key=SharedRequestBroker.request_hash(self,payload)
                if key not in self.cache or key in self.failed_requests:
                    self.halted='cache_only_missing_request'
                    self._append({**self.context,'status':'local_blocked','error':self.halted,'request_sha256':key})
                    return 503,{'error':{'message':self.halted}}
                return super()._complete(payload)
            def close(self):pass
        gateway_base=CacheOnlyGateway
    class StageCappedGateway(gateway_base):
        def _complete(self,payload):
            inputs=sum(token_count(m['content'])+token_count(m['role'])+8 for m in payload['messages'])+64
            limit=plan['judge_input_cap'] if self.context['stage']=='native_judge' else plan['answer_input_cap']
            if inputs>limit or time.monotonic()-start>plan['wall_seconds']:
                self.halted='stage_input_or_time_cap';return 503,{'error':{'message':self.halted}}
            req={'model':self.config['model'],'messages':payload['messages'],'temperature':payload.get('temperature',0),
                 'max_tokens':payload.get('max_tokens',256),'stream':False}
            write(output/'prompts'/(digest(req)+'.json'),req)
            return super()._complete(payload)
    gateway=SharedRequestBroker(seed['cache'],seed['failed']).attach(StageCappedGateway(plan['gateway_config'],output/'model-calls'))
    results=[{'profile':p['profile'],'case_id':p['case_id'],'dataset':p['dataset'],'status':'not_run','native_score':None} for p in prompts]
    write(output/'quality-results.json',results);stopped=None;final_source_verified=False
    try:
        for row,prompt in zip(results,prompts):
            if stopped:row['stopping_error']=stopped;continue
            case=cases[row['case_id']];gold=labels[row['case_id']]
            try:
                answer=complete(gateway,'',stage='answer',case_id=row['case_id'],system=row['profile'],messages=prompt['messages'],max_tokens=256)
                row.update(answer=answer,answer_status='completed')
                if case['dataset']=='locomo':
                    judge=contracts['locomo'].namespace['render_accuracy_prompt']({'question':case['question'],'gold_answer':gold['answers']},answer)
                    response=complete(gateway,judge,stage='native_judge',case_id=row['case_id'],system=row['profile'],max_tokens=128)
                    scored=contracts['locomo'].score_saved(response,{})
                    row['native_judge']=response
                else:scored=contracts['personamem_v1'].score_saved(answer,{'correct_answer':gold['correct_answer']})
                if scored['native_score'] is None:raise ValueError('native parse failure')
                row.update(status='completed',native_score=scored['native_score'],native_parse_status=scored['parse_status'])
            except (ValueError,KeyError,TypeError) as error:
                row.update(status='failed',error_type=type(error).__name__);stopped=gateway.halted or 'quality_failure'
            append(output/'quality-events.jsonl',row);write(output/'quality-results.json',results);write(output/'cost.json',gateway.summary())
        check_sources(plan['sources'])
        source_gate.recheck(plan['source_gate'])
        final_source_verified=True
    except Exception as error:
        stopped='final_source_or_execution_failure:'+type(error).__name__
    finally:
        write(output/'quality-results.json',results);write(output/'cost.json',gateway.summary())
        result={'status':'completed' if final_source_verified and not stopped and all(r['status']=='completed' for r in results) else 'incomplete',
            'planned':len(results),'scored':sum(r['native_score'] is not None for r in results),'failed':sum(r['status']=='failed' for r in results),
            'cost':gateway.summary(),'fatal':stopped,'quality_results_sha256':sha(output/'quality-results.json'),
            'native_scope':plan['quality_scope'],'billed_total_usd':None,'formal_confirmation':False,
            'final_source_verified':final_source_verified}
        if cache_only:result.update(execution_mode='exact_cache_only',model_calls=0,network_calls=0,credential_access=False)
        write(output/'quality-manifest.json',result);gateway.close()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','quality']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=Path);p.add_argument('--labels',type=Path);p.add_argument('--rows',type=Path);p.add_argument('--profiles',nargs='+')
    p.add_argument('--max-usd',type=float);p.add_argument('--preflight',type=Path);p.add_argument('--closure',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--cache-only',action='store_true');a=p.parse_args()
    if a.action=='quality':
        if not a.execute:raise ValueError('explicit execute required')
        r=execute(a.preflight,a.output,cache_only=a.cache_only)
    else:r=preflight(a.cases,a.labels,a.rows,a.output,a.profiles,a.max_usd,closure_path=a.closure)
    print(json.dumps({k:r[k] for k in ['status','planned']},ensure_ascii=False))
