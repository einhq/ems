"""EMS's start as root, given up for the user `ems` once its keys are read.

The container starts as root because the keys file (EMS_KEYS_FILE) is its operator's and readable by
them alone: `ems.keys` beside the deployment's `.env`, mounted as it is. Then EMS becomes `ems`, with
no capability of its own and none in its bounding set but SYS_ADMIN and SYS_CHROOT. Only fusermount3,
the one program here that is setuid, takes them up, to mount a caller's files (remote.py); Chromium's
sandbox has them inside the user namespace it makes (pages.py). A parser that a crafted file takes
over runs as `ems`, which owns nothing here but the scratch folder.
"""

import ctypes
import errno
import os
import pwd

USER = "ems"
PR_CAPBSET_DROP = 24
CAP_SYS_CHROOT = 18
CAP_SYS_ADMIN = 21
KEPT = (CAP_SYS_CHROOT, CAP_SYS_ADMIN)


def become_ems() -> None:
    """Run on as `ems`, its groups and home its own. Nothing to do when EMS is not root, or where the
    image has no such user (a development box); a bounding set that cannot be narrowed (no
    SETPCAP) stops EMS rather than leaving it wider than it says."""
    if os.geteuid() != 0:
        return
    try:
        user = pwd.getpwnam(USER)
    except KeyError:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    with open("/proc/sys/kernel/cap_last_cap") as handle:
        last = int(handle.read())
    for cap in range(last + 1):
        if cap in KEPT:
            continue
        if libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            if code != errno.EINVAL:
                raise OSError(code, f"cannot drop capability {cap} from the bounding set: {os.strerror(code)}")
    os.setgroups([])
    os.setgid(user.pw_gid)
    # From root to another user: the permitted, effective and ambient capabilities go with it.
    os.setuid(user.pw_uid)
    os.environ.update(HOME=user.pw_dir, USER=USER, LOGNAME=USER)
