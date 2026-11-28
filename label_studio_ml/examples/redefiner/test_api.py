"""
Tests for the ReDefiNER backend. Install the test requirements and run pytest in this folder:

    pip install -r requirements-test.txt
    pytest

The predict tests download the model from Hugging Face (about 0.8 GB) and run it on the CPU.
"""
import json

import pytest

from model import ReDefiNERModel, rulebook_from_labels, rulebook_from_setting

LABEL_CONFIG = """<View>
  <Labels name="label" toName="text">
    <Label value="PERSON" background="red" hint="A person's name, without titles such as Dr or Prof."/>
    <Label value="ORGANIZATION" background="blue" hint="A company, university or other institution."/>
  </Labels>
  <Text name="text" value="$text"/>
</View>"""

TEXT = "Prof Ada Lovelace corresponded with the Royal Society."


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.delenv("RULEBOOK", raising=False)
    from _wsgi import init_app
    app = init_app(model_class=ReDefiNERModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def predict(client, text=TEXT, label_config=LABEL_CONFIG):
    request = {'tasks': [{'id': 1, 'data': {'text': text}}], 'label_config': label_config, 'project': '7.1700000000'}
    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    return json.loads(response.data)['results'][0]


def test_rulebook_from_labels():
    assert rulebook_from_labels(LABEL_CONFIG, "label") == (
        "[RULES]\n"
        "Extract every entity of the supplied types. Return exact spans from the text.\n\n"
        "[TYPE] PERSON\n"
        "[DESCRIPTION] A person's name, without titles such as Dr or Prof.\n\n"
        "[TYPE] ORGANIZATION\n"
        "[DESCRIPTION] A company, university or other institution.\n"
    )


def test_rulebook_from_labels_without_hints():
    config = '<View><Labels name="ner" toName="t"><Label value="DRUG"/></Labels><Text name="t" value="$t"/></View>'
    assert "[TYPE] DRUG\n" in rulebook_from_labels(config, "ner")
    assert "[DESCRIPTION]" not in rulebook_from_labels(config, "ner")


def test_rulebook_setting(tmp_path):
    text = "[RULES]\nG1 A rule.\n\n[TYPE] PERSON\n"
    assert rulebook_from_setting(text, "7") == text
    (tmp_path / "book.txt").write_text(text)
    assert rulebook_from_setting(str(tmp_path / "book.txt"), "7") == text
    (tmp_path / "7.txt").write_text(text)
    assert rulebook_from_setting(str(tmp_path), "7.1700000000") == text   # the project id Label Studio sends
    assert rulebook_from_setting(str(tmp_path), "8.1700000000") is None   # no file: the labels are used
    with pytest.raises(ValueError):
        rulebook_from_setting(str(tmp_path / "missing.txt"), "7")


def test_predict(client):
    result = predict(client)
    spans = [(r['value']['start'], r['value']['end'], r['value']['text'], r['value']['labels']) for r in result['result']]
    assert spans == [(5, 17, 'Ada Lovelace', ['PERSON']), (40, 53, 'Royal Society', ['ORGANIZATION'])]
    assert all(r['from_name'] == 'label' and r['to_name'] == 'text' and r['type'] == 'labels' for r in result['result'])
    assert result['score'] == min(r['score'] for r in result['result'])
    assert result['model_version'].startswith('redefiner-')


def test_rulebook_types_outside_the_project_are_dropped(client, monkeypatch):
    # the rulebook also asks for ORGANIZATION, but this project only has PERSON
    monkeypatch.setenv("RULEBOOK", rulebook_from_labels(LABEL_CONFIG, "label"))
    config = '<View><Labels name="label" toName="text"><Label value="PERSON"/></Labels><Text name="text" value="$text"/></View>'
    result = predict(client, label_config=config)
    assert [r['value']['labels'] for r in result['result']] == [['PERSON']]


def test_editing_a_hint_changes_the_model_version(client):
    edited = LABEL_CONFIG.replace("A company, university or other institution.", "A company.")
    assert predict(client)['model_version'] != predict(client, label_config=edited)['model_version']


def test_empty_text(client):
    result = predict(client, text="")
    assert result['result'] == []
