#!/usr/bin/env bash
# Make the NVIDIA CDI specs parseable by podman 4.9.3 (Ubuntu noble / Mint 22).
#
# nvidia-ctk >= 1.17 emits CDI spec version 0.7.0 with an `additionalGids` field on each
# device. podman 4.9.3 bundles container-device-interface 0.6.x, which rejects the whole spec:
#
#   failed to parse CDI Spec "/etc/cdi/nvidia.yaml": json: unknown field "additionalGids"
#   Error: setting up CDI devices: unresolvable CDI devices nvidia.com/gpu=all
#
# The field only matters when the container process runs as a non-root user that needs
# supplementary groups for /dev/nvidia*; the EMS image runs as root, so dropping it costs
# nothing. Downgrade the declared version to 0.6.0 to match.
#
# Must run as root (the specs are root-owned). Idempotent — safe to re-run.
#
# nvidia-cdi-refresh.service regenerates /var/run/cdi/nvidia.yaml at boot and on driver or
# toolkit upgrades, reintroducing the field; install-cdi-refresh.sh makes that service run this
# after every regeneration. Paths given as arguments are patched instead of the default.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "fix-cdi: must run as root (try: sudo bash $0)" >&2
  exit 1
fi

python3 - "$@" <<'PY'
import os
import sys

import yaml

paths = sys.argv[1:] or ['/var/run/cdi/nvidia.yaml']
patched = 0

for path in paths:
    if not os.path.exists(path):
        print(f'fix-cdi: {path} absent, skipping')
        continue

    with open(path) as fh:
        spec = yaml.safe_load(fh)

    def strip(edits):
        # nvidia-ctk spells it additionalGids; the CDI schema calls it additionalGIDs.
        return any(edits.pop(k, None) is not None for k in ('additionalGids', 'additionalGIDs'))

    changed = strip(spec.get('containerEdits', {}))
    for device in spec.get('devices', []):
        changed |= strip(device.get('containerEdits', {}))

    if spec.get('cdiVersion') != '0.6.0':
        spec['cdiVersion'] = '0.6.0'
        changed = True

    if not changed:
        print(f'fix-cdi: {path} already clean')
        continue

    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        yaml.safe_dump(spec, fh, default_flow_style=False)
    os.replace(tmp, path)
    patched += 1
    print(f'fix-cdi: patched {path} -> cdiVersion 0.6.0, additionalGids removed')

print(f'fix-cdi: {patched} spec(s) patched')
PY
