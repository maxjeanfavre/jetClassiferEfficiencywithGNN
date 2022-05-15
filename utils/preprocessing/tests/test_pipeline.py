import uuid

import pytest

from utils import Paths
from utils.preprocessing.identity import Identity
from utils.preprocessing.normalizer import Normalizer
from utils.preprocessing.pipeline import PreprocessingPipeline


class TestPipeline:
    @pytest.mark.parametrize("normalizer_fitted", [1, 5])
    @pytest.mark.parametrize("normalizer_unfitted", [1, 5])
    @pytest.mark.parametrize("identity_fitted", [1, 5])
    @pytest.mark.parametrize("identity_unfitted", [1, 5])
    def test_save_load_equality(
        self,
        identity_unfitted: int,
        identity_fitted: int,
        normalizer_unfitted: int,
        normalizer_fitted: int,
        test_paths: Paths,
    ):
        pp = PreprocessingPipeline(
            column_preprocessors={
                **{
                    f"Identity_unfitted_{i}": Identity()
                    for i in range(identity_unfitted)
                },
                **{
                    f"Identity_fitted_{i}": Identity(fitted=True)
                    for i in range(identity_fitted)
                },
                **{
                    f"Normalizer_unfitted_{i}": Normalizer()
                    for i in range(normalizer_unfitted)
                },
                **{
                    f"Normalizer_fitted_{i}": Normalizer(mean=1.1, std_dev=0.128)
                    for i in range(normalizer_fitted)
                },
            }
        )

        path = test_paths.root_path / str(uuid.uuid4())

        pp.save(path=path)

        pp_loaded = PreprocessingPipeline.load(path=path)

        path.unlink(missing_ok=False)

        assert pp == pp_loaded
