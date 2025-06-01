"""Tests for the __init__ module."""

from importlib import reload

from pytest_mock import MockerFixture

import sdeul


def test_version_with_package(mocker: MockerFixture) -> None:
    package_version = "1.2.3"
    mocker.patch("importlib.metadata.version", return_value=package_version)
    mocker.patch("sdeul.__package__", new="sdeul")
    reload(sdeul)
    assert sdeul.__version__ == package_version
