"""Original saved VISION-SLS error envelopes versus those same envelopes + CP.

All results are re-evaluated on frozen reference-policy traces. This script
does not retrain a representation/envelope, optimize a controller, or simulate.
"""
from pathlib import Path
from collections import OrderedDict
import ast
import hashlib
import io
import json
import math
import pickle
import sys
import zipfile
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path('/Users/johann/project/work_based_on_spvc')
BASE=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs')
WORK=Path(__file__).resolve().parent
PROTOCOL_PATH=WORK/'protocol.json'
protocol=json.loads(PROTOCOL_PATH.read_text())
RELATIVE_TOLERANCE=1e-12

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p):return json.loads(Path(p).read_text())

def load_saved_coefficient_tensor(path):
    """Read only the known 126x5 float32 tensor, without importing Torch.

    This restricted unpickler permits three tensor data constructors and an
    OrderedDict only. It cannot import arbitrary checkpoint classes or code.
    """
    class FloatStorageToken:pass
    with zipfile.ZipFile(path) as z:
        prefix=next(n.rsplit('/',1)[0] for n in z.namelist() if n.endswith('/data.pkl'))
        def rebuild(storage,offset,size,stride,requires_grad,hooks):
            assert offset==0 and tuple(size)==(126,5) and tuple(stride)==(5,1)
            assert storage.shape==(630,)
            return storage.reshape(size).copy()
        def parameter(tensor,requires_grad,hooks):return tensor
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                allowed={('torch','FloatStorage'):FloatStorageToken,
                         ('torch._utils','_rebuild_tensor_v2'):rebuild,
                         ('torch._utils','_rebuild_parameter'):parameter,
                         ('collections','OrderedDict'):OrderedDict}
                if (module,name) not in allowed:raise ValueError((module,name))
                return allowed[module,name]
            def persistent_load(self,pid):
                kind,dtype,key,location,count=pid
                assert kind=='storage' and dtype is FloatStorageToken and key=='0' and count==630
                return np.frombuffer(z.read(prefix+'/data/'+key),dtype='<f4')
        return Reader(io.BytesIO(z.read(prefix+'/data.pkl'))).load().astype(float)

def C_from_source(source):
    nodes=[n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.Assign) and
      any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='C' for t in n.targets)]
    assert len(nodes)==1
    return np.asarray(ast.literal_eval(nodes[0].value.args[0]),float)

def original_function(scene):
    source=ROOT/'VISION-SLS/dyn'/('unicycle_of_perception.py' if scene=='parking' else 'quad_10d_of_perception.py')
    if scene=='parking':
        asset=ROOT/'VISION-SLS/dyn/unicycle_observability_bound_ppoly_097_300000_val_3_0.97_0.04.npz'
        provenance=read_json(BASE/'vision_sls_10server_20261024/provenance.json')
        assert sha(asset)==provenance['dynamics_model_npz_sha256']['dyn/'+asset.name]
        p=np.load(asset);coeff=p['theta_poly'].reshape(-1).astype(float);floor=float(p['theta_bias'][0])
        assert len(coeff)==10 and int(p['degree'])==3
        def fn(x):
            x0=x[:,0]+3;x1=x[:,1]
            M=np.column_stack([np.ones(len(x)),x1,x1**2,x1**3,x0,x0*x1,x0*x1**2,x0**2,x0**2*x1,x0**3])
            value=np.maximum(M@coeff,floor)
            return np.repeat(value[:,None],3,axis=1)
    else:
        asset=ROOT/'VISION-SLS/dyn/theta_lp_large_new2_val.pt'
        provenance=read_json(BASE/'vision_sls_quad_10server_20261024/provenance.json')
        assert sha(asset)==provenance['quad_model_sha256']['expe/quad_perception/data/'+asset.name]
        coeff=load_saved_coefficient_tensor(asset);assert coeff.shape==(126,5)
        # Use the exact upstream monomial order, not sklearn/CP feature order.
        tree=ast.parse(source.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and isinstance(n.func.value,ast.Name) and n.func.value.id=='ca' and n.func.attr=='Function'
               and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='poly']
        assert len(calls)==1 and isinstance(calls[0].args[2],ast.List)
        terms=calls[0].args[2];assert len(terms.elts)==126
        allowed=(ast.List,ast.BinOp,ast.Mult,ast.Pow,ast.Add,ast.Sub,ast.UnaryOp,ast.UAdd,ast.USub,ast.Constant,ast.Name,ast.Load)
        assert all(isinstance(n,allowed) for n in ast.walk(terms))
        assert all(n.id in ['x0','x1','x2','x3','x4'] for n in ast.walk(terms) if isinstance(n,ast.Name))
        code=compile(ast.Expression(terms),'<original quad monomials>','eval')
        def fn(x):
            local={f'x{i}':x[:,j] for i,j in enumerate([0,1,2,6,7])}
            values=eval(code,{'__builtins__':{}},local)
            M=np.column_stack([np.broadcast_to(v,(len(x),)) for v in values])
            return np.maximum(M@coeff,0.)
    assert sha(source)==provenance['source_python_sha256']['dyn/'+source.name]
    return fn,C_from_source(source),{'source':str(source),'source_sha256':sha(source),
        'coefficient_file':str(asset),'coefficient_sha256':sha(asset),'verified_against_initial_reproduction_provenance':True,
        'original_model_source_hash_verified':True,
        'shape_preserved':'scalar shared across 3 components' if scene=='parking' else '5 original component widths'}

def metrics(point_scores,multiplier):
    covered=point_scores<=multiplier+RELATIVE_TOLERANCE
    strict=point_scores<=multiplier
    whole=covered.all(axis=1)
    return {'points':int(covered.size),'points_covered':int(covered.sum()),'point_coverage':float(covered.mean()),
        'trajectories':len(covered),'trajectories_covered':int(whole.sum()),'trajectory_coverage':float(whole.mean()),
        'uncovered_trajectory_indices_0_based':np.flatnonzero(~whole).tolist(),
        'strict_zero_tolerance_points_covered':int(strict.sum()),
        'strict_zero_tolerance_trajectories_covered':int(strict.all(axis=1).sum()),
        'maximum_normalized_error_after_scaling':float(point_scores.max()/multiplier)}

record_ids={('parking','box'):'O01',('quad','box'):'O02',('parking','l2'):'O03',('quad','l2'):'O04'}
results={'date':protocol['date'],'protocol_path':str(PROTOCOL_PATH),'protocol_sha256':sha(PROTOCOL_PATH),
    'scope':protocol['scope'],'relative_comparison_tolerance':RELATIVE_TOLERANCE,'scenes':{},'comparison_records':[]}
array_output={}
for scene,settings in protocol['scenes'].items():
    folder=Path(settings['source_directory']);fn,C,original=original_function(scene)
    phases={};traces={};checks={}
    for phase in ['calibration','baseline']:
        path=folder/phase/'diagnostic_traces.npz';d=np.load(path)
        summary=read_json(folder/phase/'summary.json')
        assert summary['state_cp_model'] is None and summary.get('direct_cp_model') is None
        expected_seed=settings['calibration_seed'] if phase=='calibration' else settings['test_seed']
        assert summary['seed']==expected_seed
        if phase=='calibration':assert summary['random_only'] is True
        n,ny,T=d['residuals'].shape
        assert T==settings['observations_per_trajectory']
        assert n==(settings['calibration_trajectories'] if phase=='calibration' else settings['test_trajectories'])
        x=d['observation_states'];offset=np.asarray(settings['image_state_offset'])
        expected_indices=np.r_[0,np.arange(T-1)]
        assert np.array_equal(x,d['states'][:,:,expected_indices])
        delta=d['observations']-np.einsum('ab,nbt->nat',C,x+offset[None,:,None])
        residual_error=float(np.max(np.abs(delta-d['residuals'])))
        assert residual_error<1e-10
        recomputed=fn(x.transpose(0,2,1).reshape(-1,x.shape[1])).reshape(n,T,ny).transpose(0,2,1)
        captured=d['error_bounds']
        if captured.ndim==2:captured=np.repeat(captured[:,None,:],ny,axis=1)
        bound_error=float(np.max(np.abs(recomputed-captured)))
        assert bound_error<1e-10,(scene,phase,bound_error)
        assert np.isfinite(captured).all() and np.all(captured>0)
        # Use exactly the saved original widths; independently regenerated above.
        normalized=np.abs(delta)/captured
        phases[phase]={'box':normalized.max(axis=1),'l2':np.linalg.norm(normalized,axis=1),
                       'bounds':captured,'residuals':delta}
        traces[phase]={'path':str(path),'sha256':sha(path),'summary_path':str(folder/phase/'summary.json'),
                       'summary_sha256':sha(folder/phase/'summary.json'),'seed':summary['seed'],'trajectories':n,'observations_per_trajectory':T}
        checks[phase]={'independent_residual_max_error':residual_error,
                       'independent_original_bound_max_error':bound_error,
                       'original_component_width_min':float(captured.min()),'original_component_width_max':float(captured.max()),
                       'original_component_width_mean':float(captured.mean()),
                       'original_parameters_loaded_by_recorded_run':True}
    assert traces['calibration']['sha256']!=traces['baseline']['sha256']
    assert sha(folder/'calibration/diagnostic_traces.npz')==read_json(folder/'doc_cp_model.json')['calibration_traces_sha256']
    entries={};split=settings['axis_stress_prefix_count'];n=settings['calibration_trajectories']
    k=math.ceil((n+1)*(1-protocol['alpha']))
    for definition in ['box','l2']:
        cal=phases['calibration'][definition];test=phases['baseline'][definition]
        scores=cal.max(axis=1);q=float(np.sort(scores)[k-1]) if k<=n else float('inf')
        assert np.isfinite(q) and q>0
        before=metrics(cal,1.);after=metrics(cal,q)
        groups={}
        for group,s in [('random',slice(split,None)),('stress',slice(0,split)),('mixed',slice(None))]:
            a=test[s]
            groups[group]={'before':metrics(a,1.),'after':metrics(a,q),
                'mean_original_component_width':float(phases['baseline']['bounds'][s].mean()),
                'mean_calibrated_component_width':float(q*phases['baseline']['bounds'][s].mean())}
            # Independent direct checks in physical residual coordinates.
            if definition=='box':
                direct=(np.abs(phases['baseline']['residuals'][s])<=q*phases['baseline']['bounds'][s]+1e-12).all(axis=1)
            else:
                direct=(np.linalg.norm(phases['baseline']['residuals'][s]/(q*phases['baseline']['bounds'][s]),axis=1)<=1+1e-12)
            assert int(direct.sum())==groups[group]['after']['points_covered']
            assert int(direct.all(axis=1).sum())==groups[group]['after']['trajectories_covered']
        missing=[]
        for i in np.flatnonzero((test[split:]>q+RELATIVE_TOLERANCE).any(axis=1)):
            row=int(i)+split;t=int(np.argmax(test[row]))
            missing.append({'random_subset_run_1_based':int(i)+1,'saved_run_index_0_based':row,
                'observation_number_1_based':t+1,'trajectory_score':float(test[row,t]),
                'q':q,'factor_beyond_calibrated_set':float(test[row,t]/q),
                'residual_vector':phases['baseline']['residuals'][row,:,t].tolist(),
                'original_width_vector':phases['baseline']['bounds'][row,:,t].tolist()})
        entries[definition]={'q':q,'calibration_rank':k,'calibration_before':before,'calibration_after':after,
            'record_id':record_ids[(scene,definition)],
            'test':groups,'random_test_missing_runs':missing,'original_parameters':original,
            'width_factor':q,'set_volume_factor_at_same_state':float(q**C.shape[0])}
        model={'record_id':record_ids[(scene,definition)],'scene':scene,'definition':definition,'alpha':protocol['alpha'],'q':q,'calibration_rank':k,
            'calibration_trajectories':n,'original_parameters':original,'scaling':'q times every original component width',
            'new_floor_added':False,'original_coefficients_refitted':False,
            'calibration_trace':traces['calibration'],'protocol_sha256':sha(PROTOCOL_PATH)}
        (WORK/f'{scene}_original_{definition}_cp_model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
        results['comparison_records'].append({'id':record_ids[(scene,definition)],'scene':scene,'definition':definition,
            'q':q,'random_before':groups['random']['before'],'random_after':groups['random']['after'],
            'model_path':str(WORK/f'{scene}_original_{definition}_cp_model.json'),'new_control_rollouts':0})
        array_output[f'{scene}_{definition}_calibration_trajectory_scores']=scores
        array_output[f'{scene}_{definition}_test_point_scores']=test
        array_output[f'{scene}_{definition}_test_trajectory_scores']=test.max(axis=1)
    # Verify original-code L2 readings against its saved independent audit.
    old_comparison=read_json(folder/'doc_cp_comparison.json')['methods']['baseline']
    assert entries['l2']['test']['mixed']['before']['trajectories_covered']==old_comparison['own_bound_l2_trajectories_covered']
    assert abs(entries['l2']['test']['mixed']['before']['point_coverage']-old_comparison['own_bound_l2_point_coverage'])<1e-12
    # A paired, same-data context row for the previously fitted new h (box only).
    sys.path.insert(0,str(ROOT/'work/vision_sls_doc_cp_20261026'))
    from doc_cp_features import positive_scale
    new_model=read_json(folder/'doc_cp_model.json');data=np.load(folder/'baseline/diagnostic_traces.npz')
    x=data['observation_states'];n_test,ny,T=phases['baseline']['residuals'].shape
    h=positive_scale(x.transpose(0,2,1).reshape(-1,x.shape[1]),new_model).reshape(n_test,T)
    new_scores=np.abs(phases['baseline']['residuals']).max(axis=1)/h
    new_context={}
    for group,s in [('random',slice(split,None)),('stress',slice(0,split)),('mixed',slice(None))]:
        new_context[group]={'before':metrics(new_scores[s],1.),'after':metrics(new_scores[s],new_model['cp_multiplier']),
            'q':new_model['cp_multiplier'],'mean_original_h':float(h[s].mean()),
            'mean_calibrated_h':float(h[s].mean()*new_model['cp_multiplier'])}
    results['scenes'][scene]={'source_traces':traces,'checks':checks,'definitions':entries,
        'previous_new_fitted_scale_context_box':new_context}

np.savez_compressed(WORK/'校准分数与测试逐步比值.npz',**array_output)
results['derived_arrays']={'path':str(WORK/'校准分数与测试逐步比值.npz'),'sha256':sha(WORK/'校准分数与测试逐步比值.npz')}
results['script_sha256']=sha(Path(__file__))
(WORK/'原界与原界加CP_完整对照.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
for scene,d in results['scenes'].items():
    for definition,v in d['definitions'].items():
        print(scene,definition,'q',v['q'],'cal',v['calibration_after']['trajectories_covered'])
        for group,g in v['test'].items():
            print(' ',group,'trajectory',g['before']['trajectories_covered'],'->',g['after']['trajectories_covered'],'/',g['after']['trajectories'],
                  'points',g['before']['points_covered'],'->',g['after']['points_covered'],'/',g['after']['points'])
