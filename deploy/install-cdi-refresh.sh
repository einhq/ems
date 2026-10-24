#!/usr/bin/env bash
# Keep the NVIDIA CDI spec usable by podman 4.9.3 across reboots and driver/toolkit upgrades.
#
# nvidia-container-toolkit ships nvidia-cdi-refresh.service, which regenerates
# /var/run/cdi/nvidia.yaml from the devices actually present: at boot, and (through
# nvidia-cdi-refresh.path) whenever the driver or toolkit is installed or upgraded. Its output
# carries the real /dev/dri/cardN, but in a form podman 4.9.3 rejects (see fix-cdi.sh). This
# adds fix-cdi.sh to that service as an ExecStartPost, so every regeneration comes out
# compatible, and removes /etc/cdi/nvidia.yaml: a spec written once by hand that names whatever
# card number the host had then, and that podman would fall back to.
#
# The drop-in lives under /etc/systemd/system and the script under /usr/local/sbin, so a toolkit
# upgrade replacing the packaged unit keeps both. Must run as root. Idempotent.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "install-cdi-refresh: must run as root (try: sudo bash $0)" >&2
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT=/usr/local/sbin/ems-fix-cdi
DROPIN=/etc/systemd/system/nvidia-cdi-refresh.service.d/podman-compat.conf

systemctl cat nvidia-cdi-refresh.service >/dev/null \
  || { echo "install-cdi-refresh: nvidia-cdi-refresh.service not found (nvidia-container-toolkit >= 1.18?)" >&2; exit 1; }

install -m 0755 "$HERE/fix-cdi.sh" "$SCRIPT"
mkdir -p "$(dirname "$DROPIN")"
cat > "$DROPIN" <<EOF
# Installed by ein/ems deploy/install-cdi-refresh.sh: podman 4.9.3 cannot parse the spec as
# nvidia-ctk writes it.
[Service]
ExecStartPost=$SCRIPT /var/run/cdi/nvidia.yaml
EOF

if [ -e /etc/cdi/nvidia.yaml ]; then
  rm /etc/cdi/nvidia.yaml
  echo "install-cdi-refresh: removed /etc/cdi/nvidia.yaml"
fi

systemctl daemon-reload
systemctl enable nvidia-cdi-refresh.service nvidia-cdi-refresh.path
systemctl restart nvidia-cdi-refresh.service
systemctl --no-pager --lines=5 status nvidia-cdi-refresh.service || true
grep -E '^cdiVersion|additionalGids|/dev/dri/card' /var/run/cdi/nvidia.yaml | sort | uniq -c
