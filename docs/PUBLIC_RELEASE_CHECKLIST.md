# Public-release checklist

This repository is private while it is cleaned and reviewed. Do not change its
visibility until every P0 item has an identified owner and is complete. A scan
of the current working tree alone is insufficient; review the exact candidate
commit and its complete Git history.

## P0: candidate identity and history

- [ ] Record the candidate branch, commit, submodule commits, tag, and release
      artifact hashes.
- [ ] Confirm that the repository contains only the curated standalone history,
      not imported internal research history, reflogs, bundles, patches, backup
      archives, or editor recovery files.
- [ ] Compare the candidate tree with the approved extraction manifest and
      explain every added binary and every source directory.
- [ ] Test an anonymous clean clone, including recursive submodule checkout,
      after the intended visibility and permissions are configured.

## P0: secrets and private infrastructure

- [ ] Scan the full Git history, submodules, Git LFS objects, release assets,
      archives, source, configs, evidence, logs, and PDF metadata with at least
      two complementary secret scanners.
- [ ] Manually review high-entropy values and scanner suppressions. Never waive
      a finding only because the repository was previously private.
- [ ] Remove tokens, keys, cookies, credential files, signed URLs, private
      storage locations, hostnames, scheduler namespaces, service projects,
      account IDs, and organization-specific launch settings.
- [ ] Review historical experiment identifiers and display names in CSV/JSON
      and plotting code. Retain only identifiers approved for public provenance.
- [ ] Confirm `.gitignore` excludes model caches, checkpoints, datasets,
      credentials, local environments, results, and compiled artifacts.
- [ ] Rotate any credential that was ever committed, even if the file was later
      removed or the value appears expired.

## P0: code licensing and attribution

- [ ] Obtain approval to publish the standalone and modified code under the
      root `LICENSE`.
- [ ] Verify the source and license of every extracted file. Retain original
      copyright, patent, trademark, SPDX, and attribution notices that apply.
- [ ] Confirm that modified Apache-2.0-derived files carry a prominent
      modification notice where required, and that `NOTICE` is complete.
- [ ] Review the selected `autonumerics_zero` subtree independently; do not
      assume its root license resolves copied third-party fragments.
- [ ] Audit the patched FlashAttention-4 submodule and every nested dependency
      at the pinned commit. Preserve its separate license and notices.
- [ ] Generate a third-party dependency/license inventory for the locked Python
      and CUDA environments.
- [ ] Confirm that the project name, company names, logos, and hardware names
      comply with trademark and branding requirements.

## P0: models, datasets, and derived evidence

- [ ] Confirm that no model weights, tokenizer files, checkpoints, dataset
      samples, evaluation caches, or provider credentials are present in the
      tree, history, LFS, or release assets.
- [ ] Review every configured model and dataset against its access,
      redistribution, derivative-output, and acceptable-use terms.
- [ ] Confirm publication rights for each retained CSV, JSON, PDF, and PNG,
      including derived benchmark and training-history data.
- [ ] Decide whether historical run names, run IDs, timestamps, and service
      metadata may be public; redact or replace them without weakening the
      scientific audit trail when approval is absent.
- [ ] Document that users must obtain external models and datasets themselves;
      repository access must not imply access to those assets.

## P0: provenance and claim integrity

- [ ] Validate every paper number against the exact released evidence file and
      record its hash.
- [ ] Mark historical dirty-worktree, reconstructed-command,
      source-derived, and unbound artifacts with their actual provenance class.
- [ ] Verify that the standalone implementation is tested as a new
      implementation and is not presented as runtime attestation of historical
      outputs.
- [ ] Keep the paired pre-training workflow offline-only unless a complete new
      training manifest and authorized inputs are provided.
- [ ] State plainly that this repository does not support an exact reproduction
      of the reported 100B-token training trajectories.
- [ ] Treat runs on hardware other than GB200/SM100 as new measurements.

## P0: clean-clone verification

- [ ] Install the base package from a clean clone with no organization
      credentials or machine-specific paths.
- [ ] Run all CPU tests and configuration validation.
- [ ] Regenerate a representative CPU fit and validate its output schema.
- [ ] Build `spline_ops` in the documented SM100 environment and run its
      numerical tests.
- [ ] Run B1--B4 component probes and the fractional-`exp2` probe with all
      environment, clock, geometry, warm-up, and repetition metadata recorded.
- [ ] Run each supported open-weight patch in a legally provisioned clean
      environment; verify both patched scope and native restoration.
- [ ] Verify all documentation links and commands from the release candidate.

## P0: dependency and supply-chain review

- [ ] Replace floating dependencies and moving branches with reviewed immutable
      versions or documented compatibility ranges.
- [ ] Confirm that submodule URLs are durable and anonymously readable.
- [ ] Run dependency vulnerability, license, and binary scans.
- [ ] Review build scripts for network downloads and command execution; document
      every expected network boundary.
- [ ] Produce deterministic hashes for the source archive and released evidence
      bundle.

## P1: release quality

- [ ] Add CPU continuous integration for package import, tests, schema
      validation, fitting smoke tests, and documentation links.
- [ ] Add an opt-in SM100 validation workflow that cannot expose credentials or
      publish model artifacts.
- [ ] Add contribution, security, support, and release-notes documentation.
- [ ] Create a versioned release and verify `CITATION.cff` against it.
- [ ] Repeat the anonymous clone and archive-extraction smoke tests on the final
      release assets.

Release approval and correspondence: robert.stats.hu@gmail.com
