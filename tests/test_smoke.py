"""Package smoke tests for the scaffold."""


def test_version_resolves():
    import docspectra

    assert docspectra.__version__


def test_workspace_stack_importable():
    import xtremeparse

    assert xtremeparse.__version__
