"""Explicit supervised-retrieval influence scopes, never a pooled OOF certificate.

This contract is independent of a numerical arm. A consumer must authenticate its
own view; a model's checkpoint-selection labels count as training influence too.
"""
from __future__ import annotations

from exact.utils.fitted_artifacts import fingerprint


def plan_views(memberships, *, valid_sources):
    if not memberships:
        raise ValueError('At least one actual consumer fold map is required')
    sources = set(next(iter(memberships.values())))
    valid = set(valid_sources)
    if not sources or sources & valid:
        raise ValueError('TRAIN and public VALID sources must be disjoint')
    fits, views = {}, {}

    def model(excluded):
        influence = sorted(sources - set(excluded))
        if not influence:
            raise ValueError('No fitting sources remain')
        key = fingerprint({'label_influence_sources': influence})
        fits[key] = {'label_influence_sources': influence,
                     'excluded_sources': sorted(sources - set(influence))}
        return key

    full = model(set())
    views['public_valid'] = {'heldout_sources': sorted(valid), 'training': [],
                            'application': [{'sources': sorted(valid), 'fit': full}]}
    for name, membership in sorted(memberships.items()):
        if set(membership) != sources or any(type(f) is not int or f < 0 for f in membership.values()):
            raise ValueError('Consumer fold map must cover the exact TRAIN population')
        folds = {f: {s for s, v in membership.items() if v == f}
                 for f in sorted(set(membership.values()))}
        if len(folds) < 3:
            raise ValueError('Nested retrieval needs at least three source folds')
        # The final downstream fit gets OOF upstream inputs for every TRAIN row.
        views[name + '/final_train'] = {
            'membership_sha256': fingerprint(membership), 'heldout_sources': sorted(valid),
            'training': [{'sources': sorted(group), 'fit': model(group)}
                         for group in folds.values()], 'application': []}
        for outer, heldout in folds.items():
            views[f'{name}/fold-{outer}'] = {
                'membership_sha256': fingerprint(membership),
                'heldout_sources': sorted(heldout),
                'training': [{'sources': sorted(group), 'fit': model(heldout | group)}
                             for inner, group in folds.items() if inner != outer],
                'application': [{'sources': sorted(heldout), 'fit': model(heldout)}]}
    result = {'schema': 'luna-retrieval-consumer-views-v1', 'sources': sorted(sources),
              'valid_sources': sorted(valid), 'memberships': memberships,
              'fits': fits, 'views': views}
    validate_views(result)
    return result


def validate_views(plan):
    sources, valid = set(plan['sources']), set(plan['valid_sources'])
    expected_views = {'public_valid'}
    for scheme, membership in plan['memberships'].items():
        if set(membership) != sources or any(type(f) is not int or f < 0 for f in membership.values()):
            raise ValueError('Consumer fold membership changed')
        expected_views.add(scheme + '/final_train')
        expected_views.update(f'{scheme}/fold-{f}' for f in set(membership.values()))
    if set(plan['views']) != expected_views:
        raise ValueError('Missing consumer fold view')
    for name, view in plan['views'].items():
        heldout = set(view['heldout_sources'])
        for role in ('training', 'application'):
            seen = set()
            for entry in view[role]:
                rows = set(entry['sources'])
                if not rows or seen & rows or len(rows) != len(entry['sources']):
                    raise ValueError('Missing/duplicate source rows in consumer view')
                seen |= rows
                fit = plan['fits'][entry['fit']]
                influence = set(fit['label_influence_sources'])
                if influence & (heldout | rows):
                    raise ValueError('Upstream label self-influence or held-out consumer leakage')
                if influence | set(fit['excluded_sources']) != sources or influence & set(fit['excluded_sources']):
                    raise ValueError('Fitting scope does not partition the TRAIN population')
                if fingerprint({'label_influence_sources': sorted(influence)}) != entry['fit']:
                    raise ValueError('Fitting scope identity changed')
            if role == 'training':
                expected = set() if name == 'public_valid' else sources - heldout
            else:
                expected = set() if name.endswith('/final_train') else heldout
            if seen != expected:
                raise ValueError('Consumer view omitted part of its population')
        if name == 'public_valid' and heldout != valid:
            raise ValueError('Public VALID population changed')
        if name != 'public_valid':
            scheme = name.rsplit('/', 1)[0]
            membership = plan['memberships'][scheme]
            if view['membership_sha256'] != fingerprint(membership):
                raise ValueError('Consumer fold membership changed')
            expected = valid if name.endswith('/final_train') else {
                s for s, f in membership.items() if f == int(name.rsplit('-', 1)[1])}
            if heldout != expected:
                raise ValueError('Wrong held-out consumer fold')
    return plan


def authenticate_influence(artifact, *, allowed_sources, forbidden_sources):
    """Include optimizer, checkpoint-selection and conservative input label influence."""
    metadata = artifact.metadata
    trained = {row['source'] for row in metadata['training_pairs']}
    selected = set(metadata['validation_sources'])
    influence = set(metadata['label_influence_sources'])
    allowed, forbidden = set(allowed_sources), set(forbidden_sources)
    if trained & selected or not (trained | selected) <= influence:
        raise ValueError('Model omitted checkpoint-selection label influence')
    if influence != allowed or influence & forbidden:
        raise ValueError('Model label influence violates consumer fitting scope')
    artifact.assert_unchanged()
    return {'fit_identity': metadata['fit_identity'], 'artifact_sha256': artifact.sha256,
            'label_influence_sources': sorted(influence),
            'optimizer_sources': sorted(trained), 'checkpoint_selection_sources': sorted(selected)}
