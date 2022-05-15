import pytest

from utils.helpers.run_id_handler import RunIdHandler


def is_strictly_monotonically_increasing(l):
    for i, el in enumerate(l[1:]):
        if el <= l[i]:
            return False
    return True


class TestGenerateRunId:
    @pytest.mark.parametrize("n_runs", [20])
    @pytest.mark.parametrize("n_ids", [1, 2, 10, 10 ** 2, 10 ** 3, 10 ** 4])
    @pytest.mark.parametrize("bootstrap", [False, True])
    def test_generate_run_id(
        self,
        bootstrap: bool,
        n_ids: int,
        n_runs: int,
    ):
        for _ in range(n_runs):
            run_ids = [
                RunIdHandler.generate_run_id(bootstrap=bootstrap) for _ in range(n_ids)
            ]

            # make sure there are no duplicates
            assert len(run_ids) == len(set(run_ids))

            # make sure they are correctly ordered
            assert is_strictly_monotonically_increasing(l=run_ids)
