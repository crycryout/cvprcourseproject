#!/usr/bin/env python3
"""Validate v2 serving protocol and emit planned cases, never measured results."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_matrix(c):
    require(c.get('protocol_version') == 2, 'only current protocol v2 is accepted')
    model, data, runtime, work = c['model'], c['dataset'], c['runtime'], c['workload']
    require(model['weights_frozen'] and not model['training'], 'this project does not train models')
    require(model['pixel_mask_required'], 'DETR padding requires valid pixel masks')
    require(model['shortest_edge'] <= model['longest_edge'] <= min(model['padded_height'], model['padded_width']),
            'resize exceeds fixed padded dimensions')
    require(data['calibration_images'] + data['held_out_images'] == 5000, 'invalid COCO val split')
    require(data['held_out_requires_freeze'], 'held-out evaluation requires frozen calibration')
    buckets = runtime['batch_buckets_candidates']
    require(buckets == sorted(set(buckets)) and buckets[0] == 1, 'buckets must be unique sorted and include 1')
    require(all(isinstance(b, int) and b > 0 for b in buckets), 'invalid bucket size')
    require(runtime['compute_streams'] == 1, 'v2 default supports one compute stream')
    require(runtime['slots_per_bucket'] == 2, 'v2 reference plan requires two slots')
    require(not runtime['drop_expired'] and not runtime['result_cache'], 'no selective expiration or cached detections')
    require(work['arrival_mode'] == 'open_loop_scheduled', 'closed-loop arrivals hide queueing')
    require(work['slo_denominator'] == 'all_offered_in_measurement_window', 'SLO denominator must include rejects')
    require(abs(sum(work['deadline_probabilities']) - 1) < 1e-9, 'deadline probabilities must sum to one')
    require(len(work['deadline_probabilities']) == len(work['deadline_multipliers']), 'deadline mixture mismatch')
    require(all(x > 0 for x in work['rate_multipliers']), 'rates must be positive')
    require(work['burst_active_ms'] * work['burst_active_rate_factor'] == work['burst_period_ms'], 'burst mean rate mismatch')
    require(work['measurement_seconds'] > 0 and work['drain_cap_seconds'] >= 0, 'invalid run duration')
    cases = []

    def add(policy, trace, rate, seed, stage):
        cases.append({'case_id': f'v2_{policy}_{trace}_rho{rate:.1f}_seed{seed}',
                      'protocol_version': 2, 'status': 'planned', 'policy': policy,
                      'arrival_type': trace, 'rate_multiplier': rate, 'trace_seed': seed,
                      'stage': stage, 'requires_frozen_calibration': True,
                      'absolute_rate_and_deadlines': 'resolve_from_frozen_artifact_not_guessed'})

    for policy, trace, rate, seed in itertools.product(c['policies']['main'], work['arrival_types'],
                                                      work['rate_multipliers'], work['trace_seeds']):
        add(policy, trace, rate, seed, 'main')
    for trace, rate, seed in itertools.product(['poisson', 'burst'], [0.9, 1.1], work['trace_seeds']):
        add(c['policies']['required_ablation'], trace, rate, seed, 'required_edf_ablation')
    require(len({x['case_id'] for x in cases}) == len(cases), 'duplicate cases')
    optional_count = len(work['arrival_types']) * len(work['rate_multipliers']) * len(work['trace_seeds'])
    max_seconds = sum(work[k] for k in ['warmup_seconds', 'measurement_seconds', 'drain_cap_seconds'])
    return {'protocol_version': 2, 'status': 'plan_not_results', 'cases': cases,
            'counts': {'main': sum(x['stage'] == 'main' for x in cases),
                       'required_ablation': sum(x['stage'] != 'main' for x in cases),
                       'optional_C0_if_valid': optional_count},
            'nominal_main_plus_ablation_gpu_hours_upper_excluding_setup': round(len(cases) * max_seconds / 3600, 3),
            'budget_note': 'This is scheduled runtime, not measured cost. Calibration/capture/compile/quality/debug are extra.',
            'optional_policy': c['policies']['optional_after_bounded_attempt']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/project.json'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    raw = args.config.read_bytes()
    result = build_matrix(json.loads(raw))
    result['config_sha256'] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(json.dumps({'validated': True, 'counts': result['counts'], 'output': str(args.output)}))
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
