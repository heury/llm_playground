#!/usr/bin/env python3
"""Validate a notebook review contract and calculate evidence-based rubric scores."""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

ROLES = {'student': 'S', 'teacher': 'T', 'author': 'E', 'researcher': 'R'}
STATES = {'pass', 'fail', 'unknown'}
EXECUTION_KEYS = {'source_edits', 'syntax_checks', 'marimo_checks', 'small_cpu_execution',
                  'ui_execution', 'full_training', 'downloads', 'package_installation', 'cloud_resources'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def expected_ids(role):
    return {f'{ROLES[role]}{i}.{j}' for i in range(1, 6) for j in range(1, 5)}


def unique(rows, key, label):
    require(isinstance(rows, list), f'{label}: expected list')
    require(all(isinstance(row, dict) and nonempty(row.get(key)) for row in rows), f'{label}: invalid {key}')
    indexed = {row[key]: row for row in rows}
    require(len(indexed) == len(rows), f'{label}: duplicate {key}')
    return indexed


def files(rows, root, check_files=True):
    indexed = unique(rows, 'path', 'files')
    require(bool(indexed), 'files must not be empty')
    for path, row in indexed.items():
        require(re.fullmatch(r'[0-9a-f]{64}', str(row.get('sha256', ''))), f'{path}: invalid SHA-256')
        if check_files:
            resolved = (root / path).resolve()
            require(resolved.is_relative_to(root.resolve()), f'{path}: file outside root')
            require(resolved.is_file(), f'{path}: file missing')
            actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
            require(actual == row['sha256'], f'{path}: hash mismatch')
    return indexed


def embedded_fingerprint(path, symbols):
    """Hash Python symbol ASTs so prose edits do not change the reference revision."""
    tree = ast.parse(path.read_text())
    dumps = []
    for symbol in sorted(symbols):
        parts = symbol.split('.')
        candidates = [node for node in ast.walk(tree)
                      if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and node.name == parts[0]]
        require(len(candidates) == 1, f'{symbol}: ambiguous or missing embedded symbol')
        node = candidates[0]
        for part in parts[1:]:
            matches = [n for n in node.body if getattr(n, 'name', None) == part]
            require(len(matches) == 1, f'{symbol}: missing nested symbol')
            node = matches[0]
        dumps.append(symbol + ':' + ast.dump(node, include_attributes=False))
    return hashlib.sha256('\n'.join(dumps).encode()).hexdigest()


def validate_contract(c, root, phase='baseline', check_files=True):
    require(c.get('template_only') is False, 'complete the template first')
    require(c.get('rubric_version') == '2.0', 'unsupported rubric version')
    require(type(c.get('contract_version')) is int and c['contract_version'] >= 1, 'invalid contract version')
    for key in ['contract_id', 'project', 'run_id', 'learner', 'environment']:
        require(nonempty(c.get(key)), f'{key}: required text')
    for key in ['objectives', 'required_theory']:
        require(isinstance(c.get(key), list) and c[key] and all(nonempty(x) for x in c[key]), f'{key}: required nonempty text list')
    for key in ['prerequisites', 'optional_theory', 'assumptions']:
        require(isinstance(c.get(key), list) and all(nonempty(x) for x in c[key]), f'{key}: expected text list')
    require(c.get('class_minutes') is None or (type(c['class_minutes']) is int and c['class_minutes'] > 0), 'invalid class_minutes')
    comparison = c.get('projects_comparison', {})
    require(type(comparison.get('applicable')) is bool and nonempty(comparison.get('reason')), 'projects_comparison: required decision and reason')
    baseline = files(c.get('target_files'), root, check_files and phase == 'baseline')
    if phase == 'after':
        after = files(c.get('after_files'), root, check_files)
        require(set(after) == set(baseline), 'after_files must match target paths; scope changes require a new contract')
    refs = c.get('reference_implementations')
    require(isinstance(refs, list) and refs, 'reference_implementations required')
    for ref in refs:
        require(isinstance(ref, dict) and nonempty(ref.get('reason')), 'reference reason required')
        require(isinstance(ref.get('symbols'), list) and ref['symbols'] and all(nonempty(x) for x in ref['symbols']), 'reference symbols required')
        if ref.get('kind', 'local') == 'external':
            require(nonempty(ref.get('source')) and nonempty(ref.get('revision')), 'external reference source and revision required')
        elif ref.get('kind') == 'embedded':
            require(ref.get('path') in baseline, 'embedded reference must belong to a target file')
            require(re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256', ''))), 'invalid embedded AST hash')
            if check_files:
                actual = embedded_fingerprint(root / ref['path'], ref['symbols'])
                require(actual == ref['sha256'], 'embedded implementation changed; establish a new baseline')
        else:
            require(ref.get('kind', 'local') == 'local', 'invalid reference kind')
            require(ref.get('path') not in baseline, 'use kind=embedded for references inside target notebooks')
            files([ref], root, check_files)
    units = unique(c.get('units'), 'id', 'units')
    require(bool(units), 'units required')
    for unit in units.values():
        require(nonempty(unit.get('section')), 'unit section required')
        for key in ['concepts', 'symbols']:
            require(isinstance(unit.get(key), list) and all(nonempty(x) for x in unit[key]), f'unit {key}: expected text list')
    checkpoints = unique(c.get('checkpoints'), 'id', 'checkpoints')
    require(set(checkpoints) == set().union(*(expected_ids(r) for r in ROLES)), 'contract must contain all 80 checkpoint IDs')
    for id, cp in checkpoints.items():
        role = cp.get('role')
        require(role in ROLES and id in expected_ids(role), f'{id}: incorrect role')
        applies = cp.get('units')
        require(isinstance(applies, list) and all(isinstance(u, str) for u in applies), f'{id}: invalid units')
        require(len(applies) == len(set(applies)) and set(applies) <= set(units), f'{id}: duplicate or unknown units')
        require(type(cp.get('critical')) is bool, f'{id}: critical must be boolean')
        if applies:
            require(cp.get('excluded_reason') is None, f'{id}: applicable item cannot have excluded_reason')
        else:
            require(nonempty(cp.get('excluded_reason')) and not cp['critical'], f'{id}: invalid exclusion')
    require(any(cp['critical'] for cp in checkpoints.values()), 'at least one critical checkpoint is required')
    scope = c.get('execution_scope')
    require(isinstance(scope, dict) and set(scope) == EXECUTION_KEYS and all(type(v) is bool for v in scope.values()), 'invalid execution_scope')
    validations = unique(c.get('required_validations'), 'id', 'required_validations')
    for v in validations.values():
        require(all(nonempty(v.get(k)) for k in ['trigger', 'command_or_procedure', 'cwd', 'pass_condition']), 'incomplete validation plan')
    if scope['source_edits']:
        require(bool(validations), 'updates require a validation plan')
    return checkpoints


def score(c, result, root, check_files=True):
    phase = result.get('phase')
    require(phase in {'baseline', 'after'}, 'phase must be baseline or after')
    cps = validate_contract(c, root, phase, check_files)
    for key in ['contract_id', 'contract_version', 'rubric_version']:
        require(result.get(key) == c[key], f'{key}: result/contract mismatch')
    role = result.get('role')
    require(role in ROLES, 'invalid result role')
    targets = c['target_files'] if phase == 'baseline' else c['after_files']
    require(result.get('target_hashes') == {f['path']: f['sha256'] for f in targets}, 'result target hashes mismatch')
    rows = unique(result.get('checkpoints'), 'id', 'result checkpoints')
    require(set(rows) == expected_ids(role), 'result must contain exactly 20 role checkpoints')
    states = {}
    core_unknown = core_fail = False
    counts = {f'{ROLES[role]}{i}': Counter() for i in range(1, 6)}
    for id, row in rows.items():
        cp = cps[id]
        observations = unique(row.get('units'), 'id', f'{id} observations')
        require(set(observations) == set(cp['units']), f'{id}: missing or unexpected unit observations')
        for unit in observations.values():
            require(unit.get('status') in STATES and nonempty(unit.get('evidence')), f'{id}: status and evidence required')
        observed = {u['status'] for u in observations.values()}
        state = 'excluded' if not observed else 'fail' if 'fail' in observed else 'unknown' if 'unknown' in observed else 'pass'
        core_unknown |= cp['critical'] and 'unknown' in observed
        core_fail |= cp['critical'] and state == 'fail'
        states[id] = state
        counts[id.split('.')[0]][state] += 1
    findings = unique(result.get('findings'), 'id', 'findings')
    major = core_fail
    for finding in findings.values():
        require(finding.get('severity') in {'critical', 'major', 'minor'}, 'invalid finding severity')
        require(finding.get('status') in {'open', 'partial', 'resolved'}, 'invalid finding status')
        require(nonempty(finding.get('evidence')), 'finding evidence required')
        linked = finding.get('checkpoints')
        require(isinstance(linked, list) and linked and all(isinstance(x, str) and x in rows for x in linked), 'invalid finding checkpoint IDs')
        major |= finding['severity'] in {'critical', 'major'} and finding['status'] != 'resolved'
    require(type(result.get('references_verified')) is bool, 'references_verified must be boolean')
    active = [v for v in counts.values() if sum(v[s] for s in STATES)]
    coverage = low = unknown = Fraction(0)
    metrics = []
    for id, count in counts.items():
        p, f, u = (count[s] for s in ['pass', 'fail', 'unknown'])
        n = p + f + u
        w = Fraction(100, len(active)) if n else Fraction(0)
        if n:
            coverage += w * Fraction(p + f, n) / 100
            low += w * Fraction(p, n)
            unknown += w * Fraction(u, n)
        metrics.append({'id': id, 'counts': {s: count[s] for s in ['pass', 'fail', 'unknown', 'excluded']},
                        'weight': float(w), 'score_0_to_5': float(Fraction(5*p, p+f)) if p+f else None,
                        'coverage': float(Fraction(p+f, n)) if n else None})
    if major:
        verdict, reason = 'revise', 'unresolved major/critical defect or failed critical checkpoint'
    elif not active or coverage < Fraction(4, 5) or core_unknown or not result['references_verified']:
        verdict, reason = 'hold', 'insufficient coverage, missing references or unknown critical checkpoint'
    elif low >= 85:
        verdict, reason = 'use', 'lower bound >= 85 and critical gates satisfied'
    elif low + unknown < 85:
        verdict, reason = 'revise', 'upper bound below 85'
    else:
        verdict, reason = 'hold', 'score interval crosses acceptance threshold'
    return {'role': role, 'phase': phase, 'contract_id': c['contract_id'], 'rubric_version': c['rubric_version'],
            'metrics': metrics, 'checkpoint_states': states,
            'observed_score': float(low/coverage) if coverage else None,
            'coverage': float(coverage), 'lower_bound': float(low), 'upper_bound': float(low+unknown),
            'score_final': bool(active) and coverage == 1, 'verdict': verdict, 'reason': reason,
            'execution_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['validate-contract', 'score']:
        p = sub.add_parser(name)
        p.add_argument('contract', type=Path)
        if name == 'score':
            p.add_argument('result', type=Path)
        else:
            p.add_argument('--phase', choices=['baseline', 'after'], default='baseline')
        p.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        contract = json.loads(args.contract.read_text())
        require(isinstance(contract, dict), 'contract must be an object')
        if args.command == 'validate-contract':
            validate_contract(contract, args.root, args.phase)
            output = {'valid': True, 'phase': args.phase, 'contract_id': contract['contract_id']}
        else:
            result = json.loads(args.result.read_text())
            require(isinstance(result, dict), 'result must be an object')
            output = score(contract, result, args.root)
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, TypeError, KeyError, SyntaxError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
