"""Unit tests for generated-source inspection helpers."""

from tirx_harness.dump_kernel import dump_cuda, extract_symbols, parse_ptxas


class _Module:
    def __init__(self, source: str = "", imports: list["_Module"] | None = None) -> None:
        self._source = source
        self.imports = imports or []

    def inspect_source(self, _format: str) -> str:
        return self._source


class _Executable:
    def __init__(self, module: _Module) -> None:
        self.mod = module


def test_dump_cuda_walks_imported_modules() -> None:
    cuda = 'extern "C" __global__ void _kernel_kernel() {}'
    executable = _Executable(_Module(imports=[_Module(cuda)]))

    assert dump_cuda(executable) == cuda


def test_extract_symbols_handles_launch_bounds_and_duplicates() -> None:
    cuda = """
    __global__ void __launch_bounds__(128) first_kernel() {}
    __global__ void second_kernel() {}
    __global__ void first_kernel() {}
    """

    assert extract_symbols(cuda) == ["first_kernel", "second_kernel"]


def test_parse_ptxas_extracts_resource_counts() -> None:
    log = """
    ptxas info    : Used 153 registers, 2 barriers, 16 bytes smem
    ptxas info    : 32 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
    """

    assert parse_ptxas(log) == {
        "registers": 153,
        "spill_stores": 8,
        "spill_loads": 4,
        "stack": 32,
        "smem": 16,
        "barriers": 2,
    }
