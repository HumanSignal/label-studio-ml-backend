from model import SamMLBackend

_TEST_CONFIG = '''
<View>
    <Image name="image" value="$image" zoom="true"/>
    <BrushLabels name="tag" toName="image">
        <Label value="Banana" background="#FF0000"/>
        <Label value="Orange" background="#0d14d3"/>
    </BrushLabels>
    <KeyPointLabels name="tag2" toName="image" smart="true" >
        <Label value="Banana" background="#000000" showInline="true"/>
        <Label value="Orange" background="#000000" showInline="true"/>
    </KeyPointLabels>
    <RectangleLabels name="tag3" toName="image"  >
        <Label value="Banana" background="#000000" showInline="true"/>
        <Label value="Orange" background="#000000" showInline="true"/>
    </RectangleLabels>
</View>
'''

_TEST_TASK = [{'data': {
    'image': 'https://s3.amazonaws.com/htx-pub/datasets/images/125245483_152578129892066_7843809718842085333_n.jpg'}}]


def test_predict_with_no_context_returns_empty_list():
    model = SamMLBackend(label_config=_TEST_CONFIG)
    tasks = _TEST_TASK
    assert model.predict(tasks) == []


def test_predict_with_keypoints_calls_predictor_predict():
    model = SamMLBackend(label_config=_TEST_CONFIG)
    tasks = _TEST_TASK
    context = {
        'result': [{
            'original_width': 1080,
            'original_height': 1080,
            'image_rotation': 0,
            'value': {
                'x': 49.441786283891545,
                'y': 59.96810207336522,
                'width': 0.3189792663476874,
                'labels': ['Banana'],
                'keypointlabels': ['Banana']
            },
            'is_positive': True,
            'id': 'fBWv1t0S2L',
            'from_name': 'tag2',
            'to_name': 'image',
            'type': 'keypointlabels',
            'origin': 'manual'
        }]}
    result = model.predict(tasks, context)
    assert len(result) == 1
    assert len(result[0]['result']) == 1
    assert result[0]['result'][0]['value']['format'] == 'rle'
    assert len(result[0]['result'][0]['value']['rle']) == 6800  # exact number of pixels
    assert result[0]['result'][0]['value']['brushlabels'] == ['Banana']


def test_predict_with_rectangle_calls_predictor_predict():
    model = SamMLBackend(label_config=_TEST_CONFIG)
    tasks = _TEST_TASK
    context = {
        'result': [
            {
                'original_width': 1080,
                'original_height': 1080,
                'type': 'rectanglelabels',
                'value': {'x': 50, 'y': 50, 'width': 10, 'height': 10, 'rectanglelabels': ['Orange']}
            }
        ]
    }
    result = model.predict(tasks, context)
    assert len(result) == 1
    assert len(result[0]['result']) == 1
    assert result[0]['result'][0]['value']['format'] == 'rle'
    assert len(result[0]['result'][0]['value']['rle']) == 951  # exact number of pixels
    assert result[0]['result'][0]['value']['brushlabels'] == ['Orange']
