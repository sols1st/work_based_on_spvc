import copy
import pytest
from Aebs.vision_sls.finalize_robust_aebs import inclusion_evidence
from Aebs.vision_sls.run import sha256


def test_inclusion_requires_verified_identical_models_and_subset(tmp_path):
    paths=[tmp_path/n for n in ['policy.pt','sbc.pt','observation.pt']]
    for p in paths:p.write_bytes(p.name.encode())
    parent=dict(continuous_state_numerical_interval_pass=True,region_pass=True,
                actual_witnesses=0,unresolved_cells=0,observation_model='tube',
                source_policy_sha256=sha256(paths[0]),source_sbc_sha256=sha256(paths[1]),
                regions={'initial':{'upper':0},'unsafe':{'lower':24}},checked_cells=1234,
                guarantee_scope='test',arithmetic='float64')
    contract=dict(coefficients=[1.]*4,envelope_units='m',calibration={'q_hat':.45},
                  source_policy_sha256=sha256(paths[0]),source_observation_sha256=sha256(paths[2]))
    final=copy.deepcopy(contract);final['calibration']['q_hat']=.42
    result=inclusion_evidence(parent,contract,final,*paths)
    assert result['continuous_state_numerical_interval_pass'] and result['checked_cells']==0
    assert result['parent_checked_cells']==1234
    final['calibration']['q_hat']=.46
    with pytest.raises(ValueError,match='not contained'):inclusion_evidence(parent,contract,final,*paths)
    final['calibration']['q_hat']=.42
    for field in ['actual_witnesses','unresolved_cells']:
        bad={**parent,field:1}
        with pytest.raises(ValueError,match='failed/unknown'):inclusion_evidence(bad,contract,final,*paths)
    for path in paths:
        original=path.read_bytes();path.write_bytes(b'changed')
        with pytest.raises(ValueError,match='changed'):inclusion_evidence(parent,contract,final,*paths)
        path.write_bytes(original)
