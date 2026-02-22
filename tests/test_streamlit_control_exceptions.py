from jupr_app.ui.streamlit_exceptions import (
    is_streamlit_control_exception,
    rethrow_if_streamlit_control,
)


class RerunException(Exception):
    pass


class StopException(Exception):
    pass


def test_rerun_exception_detected():
    assert is_streamlit_control_exception(RerunException())


def test_stop_exception_detected():
    assert is_streamlit_control_exception(StopException())


def test_normal_exception_not_detected():
    assert not is_streamlit_control_exception(Exception("x"))


def test_rethrow_reraises_rerun_exception():
    exc = RerunException()
    try:
        rethrow_if_streamlit_control(exc)
    except RerunException as raised:
        assert raised is exc
    else:
        raise AssertionError("expected RerunException to be re-raised")


def test_rethrow_reraises_stop_exception():
    exc = StopException()
    try:
        rethrow_if_streamlit_control(exc)
    except StopException as raised:
        assert raised is exc
    else:
        raise AssertionError("expected StopException to be re-raised")


def test_rethrow_ignores_normal_exception():
    rethrow_if_streamlit_control(Exception("x"))
