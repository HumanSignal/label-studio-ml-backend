import pytest

from label_studio_ml.model import LabelStudioMLBase


@pytest.mark.parametrize(
    "label_config, model_names, tag_name, expected",
    [
        # Test case 1: Empty labels_attrs
        (
            """<View>
                <Image name="image" value="$image"/>
                <RectangleLabels name="label" toName="image">
                </RectangleLabels>
            </View>""",
            ['car', 'truck'],
            "label",
            {}
        ),
        # Test case 2: With predicted_values
        (
            """<View>
                <Image name="image" value="$image"/>
                <RectangleLabels name="label" toName="image">
                    <Label value="Car" predicted_values="car,truck"/>
                    <Label value="Airplane" predicted_values="airplane"/>
                </RectangleLabels>
            </View>""",
            ['car', 'truck', 'airplane'],
            "label",
            {
                'car': 'Car',
                'truck': 'Car',
                'airplane': 'Airplane'
            }
        ),
        # Test case 3: Without predicted_values
        (
            """<View>
                <Image name="image" value="$image"/>
                <RectangleLabels name="label" toName="image">
                    <Label value="Car"/>
                    <Label value="Airplane"/>
                </RectangleLabels>
            </View>""",
            ['car', 'airplane'],
            "label",
            {
                'car': 'Car',
                'airplane': 'Airplane'
            }
        ),
        # Test case 4: Partial matching labels
        (
            """<View>
                <Image name="image" value="$image"/>
                <RectangleLabels name="label" toName="image">
                    <Label value="Car" predicted_values="car"/>
                    <Label value="Plane" predicted_values="airplane"/>
                    <Label value="Flower"/>
                </RectangleLabels>
            </View>""",
            ['car', 'airplane', 'bicycle'],
            "label",
            {
                'car': 'Car',
                'airplane': 'Plane'
            }
        ),
        # Test case 5: Mixed matching labels
        (
            """<View>
                <Image name="image" value="$image"/>
                <RectangleLabels name="label" toName="image">
                    <Label value="Car" predicted_values="car"/>
                    <Label value="Plane" predicted_values="airplane"/>
                    <Label value="Bicycle"/>
                </RectangleLabels>
            </View>""",
            ['car', 'airplane', 'bicycle'],
            "label",
            {
                'car': 'Car',
                'airplane': 'Plane',
                'bicycle': 'Bicycle'
            }
        ),
        # Test case 6: Mixed matching labels with capitalized char in model names and one space
        (
                """<View>
                    <Image name="image" value="$image"/>
                    <RectangleLabels name="label" toName="image">
                        <Label value="Car" predicted_values="car"/>
                        <Label value="Plane" predicted_values="airplane"/>
                        <Label value="Bicycle space"/>
                    </RectangleLabels>
                </View>""",
                ['car', 'airplane', 'Bicycle space'],
                "label",
                {
                    'car': 'Car',
                    'airplane': 'Plane',
                    'Bicycle space': 'Bicycle space'
                }
        ),
        # Test case 7: Label not in model
        (
            """<View>
                <Image name="image" value="$image"/>
                <RectangleLabels name="label" toName="image">
                    <Label value="Car" predicted_values="car,truck"/>
                    <Label value="Airplane" predicted_values="jet"/>
                </RectangleLabels>
            </View>""",
            ['car', 'truck'],
            "label",
            {
                'car': 'Car',
                'truck': 'Car'
            }
        ),
    ]
)
def test_build_label_map(label_config, model_names, tag_name, expected):
    obj = LabelStudioMLBase(project_id="42", label_config=label_config)
    result = obj.build_label_map(tag_name, model_names)
    assert result == expected
