"""Read-back audit of saved adaptation checkpoint and test isolation; no training."""
import argparse
import json
from pathlib import Path
import numpy as np
from .adapt_projection import cell_ids, choose_epoch
from .perception import load_projection, residual_summary
from .test_perception import sha


def main():
    import torch
    import h5py
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.result_dir
    if (root/'audit.json').exists(): p.error('audit.json already exists; preserve previous audit')
    config = json.loads((root/'config.json').read_text())
    metrics = json.loads((root/'metrics.json').read_text())
    selected = json.loads((root/'selection_before_test.json').read_text())
    mismatches = [name for name, value in config['source_sha256'].items() if sha(name) != value]
    if mismatches: raise ValueError('immutable source changed: %s' % mismatches)
    if sha(root/'projection_adapted.pt') != selected['checkpoint_sha256']:
        raise ValueError('saved checkpoint does not match selection record')
    if selected['test_features_encoded'] != 0 or not selected['selection_complete_before_test']:
        raise ValueError('test was encoded before checkpoint selection')
    if choose_epoch(metrics['training']['validation_mse_by_epoch']) != selected['best_epoch']:
        raise ValueError('selected epoch differs from validation minimum')
    state_file = next(Path(k) for k in config['source_sha256'] if k.endswith('states_large_val.pt'))
    ck = next(Path(k) for k in config['source_sha256'] if k.endswith('300000_post.pt'))
    adapted_state = torch.load(root/'projection_adapted.pt', map_location='cpu')
    original_state = torch.load(ck, map_location='cpu')
    if not torch.equal(adapted_state['C'], original_state['C']): raise ValueError('C changed')
    plan = np.load(root/'split_plan.npz', allow_pickle=False)
    splits, cells = plan['splits'], plan['cell_ids']
    sets = {s: set(cells[splits == s]) for s in ['train','validation','test']}
    if any(sets[a] & sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('spatial split leakage')
    from .core import RENDER_OFFSET
    old_states, _ = torch.load(state_file, map_location='cpu')
    old_states = np.column_stack([old_states.numpy(), np.zeros(len(old_states))])-RENDER_OFFSET
    replay = np.load(root/'historical_replay_indices.npy', allow_pickle=False)
    if not np.isin(cell_ids(old_states[replay]), list(sets['train'])).all():
        raise ValueError('historical replay leaks into heldout cells')
    rows = np.flatnonzero(splits == 'test')
    saved = np.load(root/'test_predictions.npz', allow_pickle=False)
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, c, _ = load_projection(root/'projection_adapted.pt', device)
    with h5py.File(root/'paired_data.h5', 'r') as h:
        if h.attrs['selected_checkpoint_sha256_before_test_encoding'] != selected['checkpoint_sha256']:
            raise ValueError('H5 selection record mismatch')
        if not np.array_equal(h['states'][rows], saved['states']): raise ValueError('wrong test state pairing')
        if not np.all(h['encoded'][:] == 1): raise ValueError('features incomplete')
        with torch.inference_mode():
            predicted = model(torch.tensor(h['dino_features'][rows], device=device)).cpu().numpy()
    gap = float(np.abs(predicted-saved['adapted']).max())
    if gap > 1e-4: raise ValueError('saved checkpoint/test prediction mismatch: %.9g' % gap)
    actual = residual_summary(predicted, saved['states'], c)
    report = dict(status='saved_checkpoint_and_test_protocol_audit_pass',
                  immutable_source_sha256_all_match=True, C_bitwise_unchanged=True,
                  selected_epoch=selected['best_epoch'], test_encoded_before_selection=0,
                  train_validation_test_cells_disjoint=True, replay_training_cells_only=True,
                  saved_checkpoint_sha256=selected['checkpoint_sha256'],
                  readback_device=device, readback_max_prediction_abs_gap=gap,
                  recomputed_test_mean_linf=actual['latent_linf_mean'],
                  recomputed_test_position_mae_m=actual['pose_xy_euclidean_mae_m'],
                  changed_projection_tensors=sum(not torch.equal(v, original_state[k])
                                                 for k,v in adapted_state.items() if k != 'C'),
                  formal_safety_certificate=False)
    (root/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
