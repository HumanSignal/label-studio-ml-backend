import os
import prestodb
import traceback
from flask import Flask, request, jsonify, Response
from label_studio_sdk.client import LabelStudio
from typing import List

_server = Flask(__name__)


def init_app():
    return _server


@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'UP',
        'model_class': MODEL_CLASS.__name__
    })


@_server.route('/upload', methods=['POST'])
def upload_to_watsonx():
    # First, collect data from the request object passed by label studio
    input_request = request.json
    action = input_request["action"]
    annotation = input_request["annotation"]
    task = input_request["task"]

    # Connect to Label Studio
    client = connect_ls()
    data = get_data(annotation, task, client)

    # Then, connect to WatsonX.data via prestodb
    eng_username = os.getenv("WATSONX_ENG_USERNAME")
    eng_password = os.getenv("WATSONX_API_KEY")
    eng_host = os.getenv("WATSONX_ENG_HOST")
    eng_port = os.getenv("WATSONX_ENG_PORT")
    catalog = os.getenv("WATSONX_CATALOG")
    schema = os.getenv("WATSONX_SCHEMA")
    table = os.getenv("WATSONX_TABLE")

    if None in [eng_username, eng_password, eng_host, eng_port, catalog, schema, table]:
        raise Exception("You must provide the required WATSONX variables in your docker-compose.yml file!")

    try:
        with prestodb.dbapi.connect(host=eng_host, port=eng_port, user=eng_username, catalog=catalog,
                                    schema=schema, http_scheme='https',
                                    auth=prestodb.auth.BasicAuthentication(eng_username, eng_password)) as conn:

            cur = conn.cursor()
            # dynamically create table schema
            table_create, table_info_keys = create_table(table, data)
            cur.execute(table_create)

            if action == "ANNOTATION_CREATED":
                # upload new annotation to watsonx
                values = tuple([data[key] for key in table_info_keys])
                insert_command = f"""INSERT INTO {table} VALUES {values}"""
                _server.logger.debug(insert_command)
                cur.execute(insert_command)

            elif action == "ANNOTATION_UPDATED":
                # update existing annotation in watsonx by deleting the old one and uploading a new one
                delete = f"""DELETE from {table} WHERE ID={data["ID"]}"""
                _server.logger.debug(delete)
                cur.execute(delete)
                values = tuple([data[key] for key in table_info_keys])
                insert_command = f"""INSERT INTO {table} VALUES {values}"""
                _server.logger.debugint(insert_command)
                cur.execute(insert_command)

            elif action == "ANNOTATIONS_DELETED":
                # delete existing annotation in watsonx
                delete = f"""DELETE from {table} WHERE ID={data["ID"]}"""
                _server.logger.debug(delete)
                cur.execute(delete)

            conn.commit()
    except Exception as e:
        _server.logger.debug(traceback.format_exc())
        _server.logger.debug(e)


def connect_ls():
    try:
        base_url = os.getenv("LABEL_STUDIO_URL")
        api_key = os.getenv("LABEL_STUDIO_API_KEY")

        if None in [base_url, api_key]:
            raise Exception(
                "You must provide your LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY in your docker-compose.yml file!")

        client = LabelStudio(
            base_url=base_url,
            api_key=api_key
        )

        return client

    except Exception as e:
        _server.logger.debug(traceback.format_exc())
        _server.logger.debug(e)


def get_data(annotation, task, client):
    """Collect the data to be uploaded to WatsonX.data"""
    info = {}

    try:
        users = client.users.list()
        id = task["id"]
        annotator_complete = annotation["completed_by"]
        annotator_update = annotation["updated_by"]
        annotator_complete = next((x.email for x in users if x.id == annotator_complete), "")
        annotator_update = next((x.email for x in users if x.id == annotator_update), "")
        info.update({"ID": int(id), "completed_by": annotator_complete, "updated_by": annotator_update})
        for key, value in task["data"].items():
            if isinstance(value, List):
                value = value[0]
            elif isinstance(value, str) and value.isnumeric():
                value = int(value)

            if isinstance(value, str):
                value = value.strip("\"")
            info.update({key: value})

        for result in annotation["result"]:
            _server.logger.debug(result)
            val_dict_key = list(result["value"].keys())[0]
            value = result["value"][val_dict_key]
            key = result["from_name"]
            if isinstance(value, List):
                value = value[0]
            elif isinstance(value, str) and value.isnumeric():
                value = int(value)

            if isinstance(value, str):
                value = value.strip("\"")
            info.update({key: value})

        _server.logger.debug(f"INFO {info}")
        return info
    except Exception as e:
        _server.logger.debug(traceback.format_exc())
        _server.logger.debug(e)


def create_table(table, data):
    """
    Create the command for building a new table
    """
    table_info = {}
    for key, value in data.items():
        if isinstance(value, int):
            table_info.update({key: "bigint"})
        else:
            table_info.update({key: "varchar"})

    table_info_keys = sorted(table_info.keys())
    table_info_keys.insert(0, table_info_keys.pop(table_info_keys.index("ID")))
    nl = ",\n"
    strings = [f"{key} {table_info[key]}" for key in table_info_keys]
    table_create = f"""
    CREATE TABLE IF NOT EXISTS {table} ({nl.join(strings)})

    """
    _server.logger.debug(table_create)
    return table_create, table_info_keys
