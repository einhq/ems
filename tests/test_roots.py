import os

import pytest

from ems.errors import EmsError
from ems.roots import Roots, parse_roots


def test_parse_roots():
    roots = parse_roots("/a:ro, /b:rw ,/c")
    assert [(r.path, r.writable) for r in roots] == [("/a", False), ("/b", True), ("/c", False)]
    with pytest.raises(ValueError):
        parse_roots("/a:wx")
    with pytest.raises(ValueError):
        parse_roots("relative:ro")


def test_input_must_be_under_a_root(roots):
    ro, rw = roots
    (ro / "a.jpg").write_bytes(b"x")
    r = Roots(parse_roots(f"{ro}:ro,{rw}:rw"))
    assert r.input(str(ro / "a.jpg")) == str(ro / "a.jpg")
    with pytest.raises(EmsError) as e:
        r.input("/etc/hostname")
    assert e.value.code.name == "path_outside_roots" and e.value.permanent
    with pytest.raises(EmsError) as e:
        r.input(str(ro / "missing.jpg"))
    assert e.value.code.name == "not_found"
    with pytest.raises(EmsError) as e:
        r.input("relative.jpg")
    assert e.value.code.name == "path_outside_roots"


def test_symlink_that_leaves_the_roots_is_refused(roots, tmp_path):
    ro, rw = roots
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    os.symlink(outside, ro / "link.txt")
    r = Roots(parse_roots(f"{ro}:ro"))
    with pytest.raises(EmsError) as e:
        r.input(str(ro / "link.txt"))
    assert e.value.code.name == "path_outside_roots"


def test_symlink_into_another_root_resolves_there(roots):
    ro, rw = roots
    (rw / "real.jpg").write_bytes(b"x")
    os.symlink(rw / "real.jpg", ro / "link.jpg")
    r = Roots(parse_roots(f"{ro}:ro,{rw}:rw"))
    assert r.input(str(ro / "link.jpg")) == str(rw / "real.jpg")


def test_output_needs_a_writable_root(roots):
    ro, rw = roots
    r = Roots(parse_roots(f"{ro}:ro,{rw}:rw"))
    assert r.output(str(rw / "out.avif")) == str(rw / "out.avif")
    with pytest.raises(EmsError) as e:
        r.output(str(ro / "out.avif"))
    assert e.value.code.name == "path_outside_roots"
    with pytest.raises(EmsError) as e:
        r.output(str(rw / "missing-dir" / "out.avif"))
    assert e.value.code.name == "not_found"
    os.symlink("/etc/hostname", rw / "trap")
    with pytest.raises(EmsError):
        r.output(str(rw / "trap"))
