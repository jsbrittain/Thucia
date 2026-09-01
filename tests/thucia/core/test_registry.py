import pytest
from thucia.core.registry import PluginNotFoundError
from thucia.core.registry import Registry


@pytest.fixture
def registry():
    return Registry("test")


def test_register_explicit_key(registry):
    @registry.register("foo")
    class Foo:
        pass

    assert registry.get("foo") is Foo
    assert registry.get("FOO") is Foo  # case-insensitive
    assert registry.names() == ["foo"]


def test_register_via_ref(registry):
    @registry.register()
    class Bar:
        ref = "bar"

    assert registry.get("bar") is Bar


def test_register_requires_key(registry):
    with pytest.raises(ValueError, match="no key"):

        @registry.register()
        class NoRef:
            pass


def test_get_missing_raises_plugin_not_found(registry):
    with pytest.raises(PluginNotFoundError, match="'nope'"):
        registry.get("nope")
    assert isinstance(PluginNotFoundError("r", "k"), KeyError)


def test_has_and_all(registry):
    @registry.register("x")
    class X:
        pass

    assert registry.has("x")
    assert not registry.has("y")
    assert registry.all() == {"x": X}


def test_unregister_and_clear(registry):
    @registry.register("a")
    class A:
        pass

    @registry.register("b")
    class B:
        pass

    registry.unregister("a")
    assert registry.names() == ["b"]
    registry.clear()
    assert registry.names() == []
