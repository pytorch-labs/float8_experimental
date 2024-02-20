import pytest


@pytest.fixture
def x_fail_activation_hooks(request):
    use_activation_hooks = request.getfixturevalue("use_activation_hooks")
    if use_activation_hooks:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="use_activation_hooks is not supported for AOT", strict=True
            )
        )


@pytest.fixture
def x_fail_activation_hooks_with_delayed(request):
    linear_type = request.getfixturevalue("linear_type")
    use_activation_hooks = request.getfixturevalue("use_activation_hooks")
    if use_activation_hooks and linear_type == linear_type.DELAYED:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="use_activation_hooks is not supported for AOT", strict=True
            )
        )
