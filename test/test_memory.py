from divopt.memory import MemoryUnit


def test_outputs():
    memory_unit = MemoryUnit(distance_threshold=0.5, score_threshold=0.5, n_jobs=2)
    # Define some sample query SMILES and scores
    query_smiles = ["CCO", "CCN", "CCC"]
    scores = [0.8, 0.6, 0.9]

    # Call the score_and_update_memory method
    result = memory_unit.score_and_update_memory(query_smiles, scores)

    # Assert that the result is a list
    assert isinstance(result, list)

    # Assert that the length of the result is equal to the length of the query SMILES
    assert len(result) == len(query_smiles)

    # Assert that each element in the result is either 0 or 1
    assert all(element in [0, 1] for element in result)


def test_zero_score_centroid():
    memory_unit = MemoryUnit(distance_threshold=0.5, score_threshold=0.5, n_jobs=2)

    # this compound will be a centroid and receives a score of 0
    query_smiles = ["CCC"]
    scores = [0.8]

    result = memory_unit.score_and_update_memory(query_smiles, scores)
    assert result == [0]

    # check if same compound with low score will also get zero score
    query_smiles = ["CCC"]
    scores = [0.4]

    result = memory_unit.score_and_update_memory(query_smiles, scores)
    assert result == [0]


def test_score_not_high_enough():
    memory_unit = MemoryUnit(
        distance_threshold=0.5, score_threshold=0.5, n_jobs=2
    )

    # check if second compound gets a score of 0
    query_smiles = ["CCC", "CCC"]
    scores = [0.0, 0.0]

    result = memory_unit.score_and_update_memory(query_smiles, scores)
    assert result == [1, 1]


def test_distance_threshold_one():
    memory_unit = MemoryUnit(
        distance_threshold=1.0, score_threshold=0.5, n_jobs=2
    )

    # check if second compound gets a score of 0 as the distance threshold is high
    query_smiles = ["CCC", "CCCNNNNN"]
    scores = [0.8, 0.4]

    result = memory_unit.score_and_update_memory(query_smiles, scores)
    assert result == [0, 0]


def test_distance_threshold_zero():
    memory_unit = MemoryUnit(
        distance_threshold=0.0, score_threshold=0.5, n_jobs=2
    )

    # check if second compound gets a score of 1 as the distance threshold is low
    query_smiles = ["CCC", "CCCNNNNN"]
    scores = [0.8, 0.4]

    result = memory_unit.score_and_update_memory(query_smiles, scores)
    assert result == [0, 1]
