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