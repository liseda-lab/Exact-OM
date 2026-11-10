# Study feedback corrections — 10 November 2026

Implementation is pushed to `study-deploy`. Runtime commits are `2366786` (backend) and
`6e11742` (frontend). The production export was built on AppVM with Node 24.22.1 / Next
15.5.26 in `/home/pcotovio/exact-study/releases/6e11742`. Activation requires the VM owner's
sudo-authenticated service restart. The unchanged live service continues serving during
preparation. The temporary validation service is loopback-only.

## Delivered changes

- The admin page lists unrevoked, usable invitations for the selected revision, with session
  ID, current step, test/participant status and link generation. Selection drives replacement
  and revocation. Lists refresh after mutations and support pagination, empty and failure states.
  The original URL cannot be recovered from the stored digest. The authenticated, bounded
  `GET /api/v1/admin/studies/{revision}/invitations` returns no secrets or participant answers.
  Revoked links, declined sessions and closed/expired studies are excluded.
- Participant copy uses direct task instructions, singular "tool", the requested recording
  sentence and "Keep your private link secure". Repeated installation, pause, privacy/export
  and experience disclaimers were removed from the relevant screens. Supplied-file guidance
  asks participants to use only those files and the study pages. Methods remain optional,
  combinable and reported per case. Background questions refer to education or work.
- Two optional three-screen inspection guides cover Protégé and a text editor. Both include
  illustrations, a tiny downloadable fictional ontology, a task to try and an answer reveal.
  Back, skip, close and done never change required tutorial completion or assessment state.
  Guides are available from setup and the existing tutorial. They do not use scored cases.
- A route-level rendering failure now shows a recovery page with retry and reload controls.
  Unknown study stages also produce a visible message. Cookies and saved drafts are retained.

## Publication and compatibility

New revision `vm-review-20261110-copy` was validated and published to AppVM's existing
PostgreSQL database. Its frozen hash is
`4c5830ca8eacc487a2c0d484283748b8fca68d6c6f8b8d2461bc3579fde2eb13`.
Its private publication definition is
`/home/pcotovio/exact-study/publications/vm-review-20261110-copy.json`.

Use **new invitations for this revision** to review the new frozen text. Reissuing an existing
invitation preserves its original session and study revision. No sessions were migrated or
participant answers overwritten. No full ontology bundle was created. All 29 assets, cases,
schedules and grading data were reused. Development-review eligibility, test-only issuance,
disabled correctness metrics and lack of research-launch approvals remain unchanged.

`exact-study-forms/3-copy-20261110` is an explicitly admitted wording variant of forms/3.
The old form definition is unchanged and can coexist in the same database. The new variant
retains method codes, branching, required fields, exclusive choices, export schema and client
negotiation. Information and tutorial content have new version identities. Assessment content
and version are unchanged. `tools/revise_study_participant_copy.py` creates a new publication
file exclusively and refuses to overwrite an existing file or reuse the input revision.

## Verification and limits

- Frontend typecheck passed. All **142 unit tests passed**. VM production export succeeded.
- Initial study/disconnect regression run reported **35 passed / 1 opt-in skip**.
- Forms/study/v2 run reported **41 passed / 2 opt-in skips / 1 failed**. The failure was in
  the new test's assumption that a synthetic fixture had a `purpose` attribute. After correcting
  that test assertion, the remaining preservation check passed. Runtime code did not change
  for that correction. The opt-in PostgreSQL test paths were not enabled in those local runs.
- Real AppVM PostgreSQL API checks passed for create, list, replace and revoke, including removal
  from the list. The diagnostic invitation was revoked.
- **Five browser tests passed** against the new VM build and actual new publication. Coverage
  includes copy, both optional guides and skip behavior, tiny-file downloads, baseline and
  explanation ranking readiness and draft recovery after reload, mocked admin selection/actions,
  and an injected rendering failure followed by successful recovery of the same session.
  These case tests seed the required tutorial through its API. They do not certify a fresh
  end-to-end human tutorial journey. The failure injection is not the user's reported incident.
- Before changes, fresh sessions also reached baseline and explanation ranking on the public
  service without JavaScript errors. **The reported intermittent black screen was not
  reproduced and its cause remains unresolved.** The recovery boundary is a mitigation.
- Desktop Protégé and 375 px text-editor guide screenshots were visually inspected. The mobile page had no horizontal overflow.
- Changed Python files pass Black/isort checks; Git whitespace checks pass.
- Local combined storage was approximately **51.5 GB**, below the shared 100 GB cap. The VM
  had approximately **33 GB free**, and the staged source/build occupied about **199 MB**.
  Existing data, frozen hardlinks and unrelated canonical-checkout work were preserved.

Guide wording was checked against primary documentation:
[Protégé menus](https://protegeproject.github.io/protege/menus/),
[Protégé views](https://protegeproject.github.io/protege/views/), and
[OWL Functional Syntax](https://www.w3.org/TR/owl2-syntax/).

## Activation

The prepared script `/home/pcotovio/exact-study/activate-nov10.sh` authenticates with sudo
before changing live files, fast-forwards the deployment checkout, retains prior hashed
assets for already-open tabs, preserves the old export, selects the new export and restarts
`exact-study`. Its final check waits for local readiness. From the owner's workstation:

```sh
ssh -t AppVM '~/exact-study/activate-nov10.sh'
```

After activation, verify public readiness, the authenticated invitation listing and a fresh
link for `vm-review-20261110-copy`. Existing participant links keep their frozen text.
