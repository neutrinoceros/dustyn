import numpy as np

from dustyn.core.record import Record


def test_io(tmp_path):

    states = np.random.randn(101 * 4).reshape(101, 4)
    times = np.random.randn(101)

    record = Record(
        times=times,
        states=states,
        metadata=dict(
            CFL=0.1,
            label="test record",
            geometry=None,
            dimensionality=2,
        ),
    )
    savedir = tmp_path / "mysave"
    record.save(savedir)
    record2 = Record.load(savedir)
    np.testing.assert_array_equal(record.states, record2.states)
    np.testing.assert_array_equal(record.times, record2.times)
    assert record.metadata == record2.metadata
