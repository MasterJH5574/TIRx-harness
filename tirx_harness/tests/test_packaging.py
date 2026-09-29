"""Editable installs must carry the native frontend's companion files."""

import runpy
from pathlib import Path

import setuptools
from setuptools import Distribution, Extension


def test_editable_frontend_copies_identity_and_licenses(tmp_path, monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    definitions = runpy.run_path(str(Path(__file__).resolve().parents[2] / "setup.py"))
    source = tmp_path / "src"
    package = source / "tirx_harness" / "numsim"
    package.mkdir(parents=True)
    extension = Extension("tirx_harness.numsim._tvm_rust_ext", sources=[], py_limited_api=True)
    distribution = Distribution({
        "packages": ["tirx_harness.numsim"],
        "package_dir": {"": str(source)},
        "ext_modules": [extension],
    })
    command = definitions["RustBuildExt"](distribution)
    command.build_lib = str(tmp_path / "build")
    command.ensure_finalized()
    library = Path(command.get_ext_fullpath(extension.name))
    library.parent.mkdir(parents=True)
    library.write_bytes(b"native frontend")
    identity = library.with_name("_tvm_rust_ext.identity")
    identity.write_text("test-build:revision\n")
    licenses = library.parent / "_thirdparty_licenses" / "tvm-rust-ext"
    licenses.mkdir(parents=True)
    for name in ("LICENSE", "NOTICE"):
        (licenses / name).write_text(name)

    # Setuptools' editable build copies extensions from build_lib to src.
    command.copy_extensions_to_source()

    assert (package / library.name).read_bytes() == library.read_bytes()
    assert (package / identity.name).read_text() == identity.read_text()
    for name in ("LICENSE", "NOTICE"):
        assert (package / "_thirdparty_licenses" / "tvm-rust-ext" / name).read_text() == name
