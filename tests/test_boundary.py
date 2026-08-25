"""Package boundaries: the PydanticAI boundary holds, and parser core
stays free of parsing dependencies."""

import ast
import sys
from pathlib import Path

import docspectra


def test_pydantic_ai_confined_to_agents():
    package = Path(docspectra.__file__).parent
    offenders = [str(p.relative_to(package)) for p in package.rglob('*.py')
                 if 'pydantic_ai' in p.read_text(encoding='utf-8')
                 and p.relative_to(package).parts[0] != 'agents']
    assert offenders == []


def test_parser_core_imports_no_parsing_dependency():
    """Source-level allowlist: every import in parser/ is stdlib or
    docspectra — deterministic, no interpreter state."""
    package = Path(docspectra.__file__).parent / 'parser'
    foreign = []
    for path in package.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or '']
            else:
                continue
            for module in modules:
                if module.split('.')[0] not in sys.stdlib_module_names \
                        and not module.startswith('docspectra'):
                    foreign.append(f'{path.name}: {module}')
    assert foreign == []
