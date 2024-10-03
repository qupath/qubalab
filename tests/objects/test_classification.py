from qubalab.objects.classification import Classification


def test_name():
    expected_name = "name"
    classification = Classification(expected_name)

    name = classification.name

    assert expected_name == name


def test_color():
    expected_color = (2, 20, 56)
    classification = Classification(None, expected_color)

    color = classification.color

    assert expected_color == color


def test_cache_when_None_name_provided():
    classification = Classification.get_cached_classification(None)

    assert classification == None


def test_cache_when_empty():
    name = "name"
    color = (2, 20, 56)

    classification = Classification.get_cached_classification(name, color)

    assert classification == Classification(name, color)


def test_cache_when_not_empty_and_same_name():
    cached_name = "name"
    cached_color = (2, 20, 56)
    other_name = cached_name
    other_color = (4, 65, 7)
    cached_classification = Classification.get_cached_classification(cached_name, cached_color)

    classification = Classification.get_cached_classification(other_name, other_color)

    assert classification != Classification(other_name, other_color) and classification == cached_classification


def test_cache_when_not_empty_and_different_name():
    cached_name = "name"
    cached_color = (2, 20, 56)
    other_name = "other name"
    other_color = (4, 65, 7)
    cached_classification = Classification.get_cached_classification(cached_name, cached_color)

    classification = Classification.get_cached_classification(other_name, other_color)

    assert classification == Classification(other_name, other_color) and classification != cached_classification
