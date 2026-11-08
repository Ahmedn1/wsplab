"""Require real, predeclared, paired weight interventions before accepting a result."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import replace

from wsplab.passes.base import PassInvalid
from wsplab.patch.format import Patch


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def patch_identity(patch):
    h = hashlib.sha256(json.dumps(patch.tensors, sort_keys=True).encode())
    for key in sorted(patch.z.files):
        array = patch.z[key]
        h.update(key.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


class InterventionEditor:
    fence = 'forward'

    def __init__(self, editor, *, dest, dev_ids, holdout_ids, required, diagnostics):
        self.editor, self.name, self.dest = editor, editor.name, dest
        self.dev_ids, self.holdout_ids = dev_ids, holdout_ids
        self.required, self.diagnostic_limit = required, diagnostics
        self.diagnostics = 0
        self.tested = []
        self.confirmation = None
        self.frozen_plan = None
        self.frozen_hash = None
        self.session = None

    @property
    def rounds(self):
        return getattr(self.editor, 'rounds', None)

    def attach(self, session):
        self.session = session
        self.forward = session.tools['forward']
        self.work = self.forward.work
        session.tools['forward'] = self
        session.before_turn = self.before_turn
        session.messages = replace(session.messages, nudge=(
            'The required intervention experiment is not complete. A diagnosis-only final answer '
            'is not accepted. Follow the progress notice and issue one executable block. '
            'Forward passes remaining: {left}.'))

    def reply(self, system, messages):
        return self.editor.reply(system, messages)

    def current(self):
        return self.session.kernel.edits_used()[1]

    def best(self):
        return max(self.tested, key=lambda x: (x['gain'], -x['changed_entries'], x['candidate_id'])) if self.tested else None

    def complete(self):
        if len(self.tested) < self.required:
            return False
        best = self.best()
        current = self.current()
        if best['gain'] <= 0:
            return current is None
        if self.confirmation is None:
            return False
        expected = self.confirmation['edit_hash'] if self.confirmation['gain'] > 0 else None
        return current == expected

    def snapshot(self):
        return {'required': self.required, 'tested': len(self.tested), 'diagnostic_passes': self.diagnostics,
                'candidates': self.tested, 'confirmation': self.confirmation,
                'complete': self.complete() if self.session else False, 'at': time.time()}

    def save(self):
        target = self.dest / 'intervention-status.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(self.snapshot(), indent=2) + '\n')
        temp.replace(target)

    def before_turn(self, session):
        done = self.complete()
        session.policy.require_all_passes = not done
        best = self.best()
        if done:
            instruction = 'Required tests and final selection are complete. Finish without another block.'
        elif len(self.tested) < self.required:
            instruction = (f'Tested {len(self.tested)}/{self.required} required nonzero candidates. '
                           'Freeze interventions.json, save the next independent edit, set active_candidate.txt, '
                           'then request model=both on all 32 DEV IDs, n=1. Causal proof is not required to test a hypothesis.')
        elif best['gain'] > 0 and self.confirmation is None:
            instruction = (f'Restore DEV winner {best["candidate_id"]}, hash {best["edit_hash"]}, '
                           'freeze it and request model=both on all 32 HOLDOUT IDs, n=1. Do not retune.')
        elif self.confirmation and self.confirmation['gain'] > 0:
            instruction = (f'Restore the exact confirmed edit {self.confirmation["candidate_id"]}, '
                           f'hash {self.confirmation["edit_hash"]}, and finish. No further tuning is permitted.')
        else:
            instruction = 'Restore original weights with drop_edit for every edited tensor, then finish.'
        if session.state.runs >= 6 and not self.tested:
            instruction += ' The first-edit target has passed: stop diagnostic analysis and execute a candidate.'
        session.notice(instruction)
        self.save()

    def plan(self):
        try:
            value = json.loads((self.work / 'interventions.json').read_text())
        except (OSError, ValueError) as exc:
            raise PassInvalid('Write /work/interventions.json before testing edits') from exc
        if not isinstance(value, list) or len(value) != self.required:
            raise PassInvalid(f'Predeclare exactly {self.required} candidates')
        ids = []
        for entry in value:
            if not isinstance(entry, dict) or any(not entry.get(k) for k in ('id', 'tensors', 'operation', 'rationale')):
                raise PassInvalid('Each candidate needs id, tensors, operation, strength and rationale')
            if 'strength' not in entry or not isinstance(entry['id'], str):
                raise PassInvalid('Candidate id must be a string and strength must be explicit')
            if not isinstance(entry['tensors'], list) or not all(isinstance(t, str) for t in entry['tensors']):
                raise PassInvalid('Candidate tensors must be a list of exact tensor names')
            ids.append(entry['id'])
        if len(set(ids)) != len(ids):
            raise PassInvalid('Candidate IDs must be unique')
        fingerprint = canonical(value)
        if self.frozen_hash is not None and fingerprint != self.frozen_hash:
            raise PassInvalid('The intervention plan is frozen; restore its original contents')
        if self.frozen_hash is None:
            self.frozen_plan, self.frozen_hash = value, fingerprint
            (self.dest / 'interventions-frozen.json').write_text(json.dumps(value, indent=2) + '\n')
        return value

    def run(self, body, request_number):
        request = self.forward.parse(body)
        if request['type'] != 'generate' or request.get('prompts'):
            raise PassInvalid('This intervention experiment permits only generation on fixed DEV/HOLDOUT IDs')
        ids = request.get('problems')
        holdout = ids == self.holdout_ids
        if ids != self.dev_ids and not holdout:
            raise PassInvalid('Use all 32 DEV or all 32 HOLDOUT IDs in their original order')
        if request.get('n', 1) != 1 or request.get('max_tokens', 32768) != 32768:
            raise PassInvalid('Use n=1 and max_tokens=32768 throughout this experiment')
        if self.confirmation is not None:
            raise PassInvalid('Confirmation is complete; no further inference or tuning is permitted')
        if request.get('model') == 'original' and not holdout:
            if self.tested or self.diagnostics >= self.diagnostic_limit:
                raise PassInvalid('Diagnostic allowance is exhausted; test a nonzero edit')
            result = self.forward.run(body, request_number)
            self.diagnostics += 1
            self.save()
            return result
        if request.get('model') != 'both':
            raise PassInvalid('Edited candidates require a paired model=both request')
        if not self.current():
            raise PassInvalid('No edit is saved; use save_edit before requesting a paired test')
        plan = self.plan()
        if holdout:
            best = self.best()
            if len(self.tested) != self.required or not best or best['gain'] <= 0 or self.current() != best['edit_hash']:
                raise PassInvalid('HOLDOUT requires all candidate tests and the frozen positive-gain DEV winner')
        else:
            if self.session.state.budget - self.session.state.passes <= 1:
                raise PassInvalid('The last pass is reserved for HOLDOUT confirmation')
            try:
                active = (self.work / 'active_candidate.txt').read_text().strip()
            except OSError as exc:
                raise PassInvalid('Write the candidate ID to /work/active_candidate.txt') from exc
            spec = next((p for p in plan if p['id'] == active), None)
            if spec is None or any(t['candidate_id'] == active for t in self.tested):
                raise PassInvalid('Choose an untested predeclared candidate ID')
        # Materialize using the official builder, then verify the patch's real
        # dtype-rounded changes. A saved float tensor can still round to no edit.
        model = self.forward.edited()
        patch = Patch.load(model.parent / 'patch')
        changed = patch.manifest['changed_entries_total']
        fingerprint = patch_identity(patch)
        if not changed:
            raise PassInvalid('The saved edit changes zero model entries after rounding')
        if not holdout:
            if any(t['patch_hash'] == fingerprint for t in self.tested):
                raise PassInvalid('This checkpoint was already tested; distinct float files do not count as distinct edits')
            if {t['name'] for t in patch.tensors} != set(spec['tensors']):
                raise PassInvalid('Changed tensor names must match the predeclared candidate')
        result = self.forward.run(body, request_number)
        summary = result.summary
        if not isinstance(summary, dict) or not summary.get('ok') or summary.get('paired') is None:
            raise RuntimeError('Paired inference completed without usable comparison evidence')
        record = {'candidate_id': best['candidate_id'] if holdout else active,
                  'edit_hash': self.current(), 'patch_hash': fingerprint, 'changed_entries': changed,
                  'gain': summary['paired']['accuracy_difference'], 'paired': summary['paired'],
                  'request_number': request_number}
        if holdout:
            self.confirmation = record
        else:
            self.tested.append(record)
            shutil.copytree(patch.path, self.dest / 'tested-patches' / str(len(self.tested)))
        self.save()
        return result

    def validate_finished(self):
        self.plan()
        if not self.complete():
            self.save()
            raise RuntimeError('Required intervention tests or final confirmation were not completed')
