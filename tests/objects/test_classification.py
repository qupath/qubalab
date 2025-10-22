from qubalab.objects.classification import Classification


def test_name():
    expected_names = ("name",)
    classification = Classification(expected_names)

    names = classification.names

    assert expected_names == names


def test_color():
    expected_color = (2, 20, 56)
    classification = Classification("name", expected_color)

    color = classification.color

    assert expected_color == color


def test_None_when_names_is_None():
    classification = Classification(None)
    assert classification is None


def test_cache_when_empty():
    name = "name"
    color = (2, 20, 56)

    classification = Classification(name, color)

    assert classification is Classification(name, color)


def test_cache_when_not_empty_and_same_name():
    cached_name = "name"
    cached_color = (2, 20, 56)
    other_name = cached_name
    other_color = (4, 65, 7)
    cached_classification = Classification(cached_name, cached_color)

    classification = Classification(other_name, other_color)

    assert (
        classification is Classification(other_name, other_color)
        and classification is cached_classification
    )


def test_cache_when_not_empty_and_different_name():
    cached_name = "name"
    cached_color = (2, 20, 56)
    other_name = "other name"
    other_color = (4, 65, 7)
    cached_classification = Classification(cached_name, cached_color)

    classification = Classification(other_name, other_color)

    assert (
        classification == Classification(other_name, other_color)
        and classification != cached_classification
    )


def test_names_input():
    names = ("a", "b")
    class1 = Classification(names)
    class2 = Classification(list(names))
    assert class1 is class2
