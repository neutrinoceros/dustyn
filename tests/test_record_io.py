import numpy as np
import pytest

from dustyn.core.record import Record


def _get_record(tstart, tend, shape):
    states = np.random.randn(np.product(shape)).reshape(shape)
    times = np.linspace(tstart, tend, shape[0])

    return Record(
        times=times,
        states=states,
        metadata=dict(
            CFL=0.1,
            label="test record",
            geometry=None,
            dimensionality=2,
        ),
    )


def test_write_then_read(tmp_path):
    record = _get_record(0, 1, shape=(101, 4))
    savedir = tmp_path / "mysave"
    record.save(savedir)
    record2 = Record.load(savedir)
    np.testing.assert_array_equal(record.states, record2.states)
    np.testing.assert_array_equal(record.times, record2.times)
    assert record.metadata == record2.metadata


def test_create_invalid(tmp_path):
    record1 = _get_record(0.0, 1.0, shape=(101, 4))
    record2 = _get_record(1.001, 2.0, shape=(100, 4))

    savedir = tmp_path / "mysave"
    record1.save(savedir)
    msg = (
        f"{str(savedir)!r} already exists. "
        "Use mode='append' to add a new entry "
        "or mode='overwrite' to erase the previous record."
    )
    with pytest.raises(FileExistsError, match=msg):
        # test default behaviour
        record2.save(savedir)

    with pytest.raises(FileExistsError, match=msg):
        # test default behaviour (explicit)
        record2.save(savedir, mode="create")


def test_append(tmp_path):
    record1 = _get_record(0.0, 1.0, shape=(101, 4))
    record2 = _get_record(1.001, 2.0, shape=(100, 4))

    savedir = tmp_path / "mysave"
    record1.save(savedir)
    record2.save(savedir, mode="append")
    record3 = Record.load(savedir)
    np.testing.assert_array_equal(
        record3.states, np.concatenate([record1.states, record2.states])
    )
    np.testing.assert_array_equal(
        record3.times, np.concatenate([record1.times, record2.times])
    )


def test_append_then_overwrite(tmp_path):
    record1 = _get_record(0.0, 1.0, shape=(101, 4))
    record2 = _get_record(1.001, 2.0, shape=(100, 4))
    record3 = _get_record(0.0, 1.0, shape=(100, 4))

    savedir = tmp_path / "mysave"
    record1.save(savedir)
    record2.save(savedir, mode="append")
    with pytest.raises(
        RuntimeError, match="Cannot overwrite a data dir with more than one record"
    ):
        record3.save(savedir, mode="overwrite")


def test_overwrite(tmp_path):
    record1 = _get_record(0.0, 1.0, shape=(101, 4))
    record2 = _get_record(1.001, 2.0, shape=(100, 4))

    savedir = tmp_path / "mysave"
    record1.save(savedir)
    record2.save(savedir, mode="overwrite")
    record3 = Record.load(savedir)
    np.testing.assert_array_equal(record3.states, record2.states)
    np.testing.assert_array_equal(record3.times, record2.times)


def test_load_extras_multiple_records(tmp_path):
    record1 = _get_record(0.0, 1.0, shape=(101, 4))
    record2 = _get_record(1.001, 2.0, shape=(100, 4))

    savedir = tmp_path / "mysave"
    record1.save(
        savedir,
        extra={
            "field0": record1.states + 1,
        },
    )
    record2.save(
        savedir,
        extra={
            "field0": record2.states + 1,
        },
        mode="append",
    )

    Record.load(savedir)  # check that loading without extras still works
    record4 = Record.load(savedir, extra=True)

    np.testing.assert_array_equal(
        record4.field0, 1 + np.concatenate([record1.states, record2.states])
    )
