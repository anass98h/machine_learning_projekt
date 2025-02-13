import custom_transformers
import pandas as pd
import pandas.testing as pdt
import pytest


@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
        "D": [10, 11, 12]
    })
    return df


class TestCombineCorrelatedFeatures:

    def test_combine_single_pair_of_columns(self, sample_df):
        pair_to_combine = [(0, 2)]
        combiner = custom_transformers.CombineCorrelatedFeatures(pair_to_combine)

        actual = combiner.transform(sample_df)

        expected_added_column = pd.DataFrame({"combined_A_C": [7, 16, 27]})
        expected = sample_df.copy().join(expected_added_column)

        pdt.assert_frame_equal(actual, expected)


    def test_combine_no_columns(self, sample_df):
        pairs_to_combine = []
        combiner = custom_transformers.CombineCorrelatedFeatures(pairs_to_combine)

        actual = combiner.transform(sample_df)
        expected = sample_df
        pdt.assert_frame_equal(actual, expected)


    def test_combine_multiple_pairs_of_columns(self, sample_df):
        pairs_to_combine = [(0, 1), (2, 3)]
        combiner = custom_transformers.CombineCorrelatedFeatures(pairs_to_combine)

        actual = combiner.transform(sample_df)

        expected_added_columns = pd.DataFrame({
            "combined_A_B": [4, 10, 18],
            "combined_C_D": [70, 88, 108]
        })
        expected = sample_df.copy().join(expected_added_columns)

        pdt.assert_frame_equal(actual, expected)


    def test_combine_out_of_range_indices(self, sample_df):
        pair_to_combine = [(0, 4)]
        combiner = custom_transformers.CombineCorrelatedFeatures(pair_to_combine)

        with pytest.raises(IndexError):
            combiner.transform(sample_df)


    def test_combine_non_integer_indices(self, sample_df):
        pair_to_combine = [("A", "B")]
        combiner = custom_transformers.CombineCorrelatedFeatures(pair_to_combine)

        with pytest.raises(IndexError):
            combiner.transform(sample_df)


    def test_combine_more_than_two_columns(self, sample_df):
        pair_to_combine = [(1, 2, 3)]
        combiner = custom_transformers.CombineCorrelatedFeatures(pair_to_combine)

        with pytest.raises(ValueError):
            combiner.transform(sample_df)


class TestFeatureWeights:

    def test_weight_one_feature(self, sample_df):
        weights = {1: 2}
        weighter = custom_transformers.FeatureWeights(weights)

        actual = weighter.transform(sample_df)

        expected = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [8, 10, 12],
            "C": [7, 8, 9],
            "D": [10, 11, 12]
        })

        pdt.assert_frame_equal(actual, expected)


    def test_weight_multiple_features(self, sample_df):
        weights = {0: 5, 1: 4, 2: 3, 3: 2}
        weighter = custom_transformers.FeatureWeights(weights)

        actual = weighter.transform(sample_df)

        expected = pd.DataFrame({
            "A": [5, 10, 15],
            "B": [16, 20, 24],
            "C": [21, 24, 27],
            "D": [20, 22, 24]
        })

        pdt.assert_frame_equal(actual, expected)


    def test_weight_no_features(self, sample_df):
        weights = {}
        weighter = custom_transformers.FeatureWeights(weights)

        actual = weighter.transform(sample_df)
        expected = sample_df
        pdt.assert_frame_equal(actual, expected)


    def test_weight_float_weights(self, sample_df):
        weights = {0: 1.5}
        weighter = custom_transformers.FeatureWeights(weights)

        actual = weighter.transform(sample_df)

        expected = pd.DataFrame({
            "A": [1.5, 3, 4.5],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
            "D": [10, 11, 12]
        })

        pdt.assert_frame_equal(actual, expected)


    def test_weight_out_of_range_indices(self, sample_df):
        weights = {4: 2}
        weighter = custom_transformers.FeatureWeights(weights)

        actual = weighter.transform(sample_df)
        expected = sample_df
        pdt.assert_frame_equal(actual, expected)


    def test_weight_non_integer_indices(self, sample_df):
        weights = {1.1: 2}
        weighter = custom_transformers.FeatureWeights(weights)
        with pytest.raises(IndexError):
            weighter.transform(sample_df)


    def test_weight_non_numeric_weights(self, sample_df):
        weights = {0: "a"}
        weighter = custom_transformers.FeatureWeights(weights)
        with pytest.raises(TypeError):
            weighter.transform(sample_df)


class TestSymmetricalColumns:

    def test_symmetrical_single_pair_of_columns(self, sample_df):
        symmetry_pairs = [(0, 2)]
        symmetrizer = custom_transformers.symmetricalColumns(symmetry_pairs)

        actual = symmetrizer.transform(sample_df)

        expected_added_column = pd.DataFrame({"sym_A_C": [8, 10, 12]})
        expected = sample_df.copy().join(expected_added_column)

        pdt.assert_frame_equal(actual, expected)


    def test_symmetrical_no_columns(self, sample_df):
        symmetry_pairs = []
        symmetrizer = custom_transformers.symmetricalColumns(symmetry_pairs)

        actual = symmetrizer.transform(sample_df)
        expected = sample_df
        pdt.assert_frame_equal(actual, expected)


    def test_symmetrical_multiple_pairs_of_columns(self, sample_df):
        symmetry_pairs = [(0, 1), (2, 3)]
        symmetrizer = custom_transformers.symmetricalColumns(symmetry_pairs)

        actual = symmetrizer.transform(sample_df)

        expected_added_columns = pd.DataFrame({
            "sym_A_B": [5, 7, 9],
            "sym_C_D": [17, 19, 21]
        })
        expected = sample_df.copy().join(expected_added_columns)

        pdt.assert_frame_equal(actual, expected)


    def test_symmetrical_out_of_range_indices(self, sample_df):
        symmetry_pairs = [(0, 4)]
        symmetrizer = custom_transformers.symmetricalColumns(symmetry_pairs)

        with pytest.raises(IndexError):
            symmetrizer.transform(sample_df)


    def test_symmetrical_non_integer_indices(self, sample_df):
        symmetry_pairs = [("A", "B")]
        symmetrizer = custom_transformers.symmetricalColumns(symmetry_pairs)

        with pytest.raises(IndexError):
            symmetrizer.transform(sample_df)


    def test_symmetrical_more_than_two_columns(self, sample_df):
        symmetry_pairs = [(1, 2, 3)]
        symmetrizer = custom_transformers.symmetricalColumns(symmetry_pairs)

        with pytest.raises(ValueError):
            symmetrizer.transform(sample_df)