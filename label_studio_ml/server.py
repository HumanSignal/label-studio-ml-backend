import os
import subprocess
import logging
import argparse
import shutil

import colorama
import re

from colorama import Fore


from .model import get_all_classes_inherited_LabelStudioMLBase


colorama.init()
logger = logging.getLogger(__name__)


def get_args():
    root_parser = argparse.ArgumentParser(add_help=False)

    root_parser.add_argument(
        '--root-dir', dest='root_dir', default='.',
        help='Projects root directory')

    parser = argparse.ArgumentParser(description='Label studio')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # init sub-command parser
    parser_init = subparsers.add_parser('init', aliases=['create'], help='Initialize Label Studio', parents=[root_parser])
    parser_init.add_argument(
        'project_name',
        help='Path to directory where project state will be initialized')
    parser_init.add_argument(
        '--script', '--from', dest='script',
        help='Machine learning script of the following format: /my/script/path:ModelClass')
    parser_init.add_argument(
        '--force', dest='force', action='store_true',
        help='Force recreating the project if exists')

    # start sub-command parser
    parser_start = subparsers.add_parser('start', help='Initialize Label Studio', parents=[root_parser])
    parser_start.add_argument(
        'project_name',
        help='Path to directory where project state will be initialized')

    parser_start.add_argument('--basic-auth-user', dest="basic_auth_user",
                              default=os.environ.get('ML_SERVER_BASIC_AUTH_USER', None),
                              help='Basic auth user')
    
    parser_start.add_argument('--basic-auth-pass', dest="basic_auth_pass",
                              default=os.environ.get('ML_SERVER_BASIC_AUTH_PASS', None),
                              help='Basic auth pass')
    
    # start deploy to gcp
    parser_deploy = subparsers.add_parser('deploy', help='Deploy Label Studio', parents=[root_parser])
    parser_deploy.add_argument(
        'provider',
        help='Provider where to deploy')
    parser_deploy.add_argument(
        'project_name',
        help='Path to directory where project state will be initialized')
    parser_deploy.add_argument(
        '--script', '--from', dest='script',
        help='Machine learning script of the following format: /my/script/path:ModelClass')
    parser_deploy.add_argument(
        '--force', dest='force', action='store_true',
        help='Force recreating the project if exists')
    parser_deploy.add_argument(
        '--gcp-project-id', dest='gcp_project',
        help='GCP project ID')
    parser_deploy.add_argument(
        '--gcp-region', dest='gcp_region',
        help='GCP region')
    parser_deploy.add_argument(
        '--label-studio-host', dest='label_studio_host', default='https://app.heartex.com',
        help='Label Studio hostname')
    parser_deploy.add_argument(
        '--label-studio-api-key', dest='label_studio_api_key', required=True,
        help='Label Studio API key')

    args, subargs = parser.parse_known_args()
    return args, subargs


def create_dir(args):
    print('==========================')
    print('Welcome to Label Studio ML!')
    print('==========================')

    output_dir = os.path.join(args.root_dir, args.project_name)
    if os.path.exists(output_dir) and args.force:
        shutil.rmtree(output_dir)
    elif os.path.exists(output_dir):
        print(Fore.RED + f'Model directory "{output_dir}" already exists. Please remove it or run:\n'
              f'label-studio-ml create {args.project_name} --force')
        return

    default_configs_dir = os.path.join(os.path.dirname(__file__), 'default_configs')
    shutil.copytree(default_configs_dir, output_dir, ignore=shutil.ignore_patterns('*.py', '*.tmpl'))

    # extract script name and model class
    if not args.script:
        script_path = os.path.join(default_configs_dir, 'model.py')
        print(f'You don\'t specify script path: by default, "{script_path}" is used')
    else:
        script_path = args.script

    def model_def_in_path(path):
        is_windows_path = path[1:].startswith(':\\')
        return ':' in path[2:] if is_windows_path else ':' in path

    if model_def_in_path(script_path):
        script_path, model_class = args.script.rsplit(':', 1)
    else:
        model_classes = get_all_classes_inherited_LabelStudioMLBase(script_path)
        if len(model_classes) > 1:
            raise ValueError(
                'You don\'t specify target model class, and we\'ve found {num} possible candidates within {script}. '
                'Please specify explicitly which one should be used using the following format:\n '
                '{script}:{model_class}'.format(num=len(model_classes), script=script_path, model_class=model_classes[0]))
        model_class = model_classes[0]

    if not os.path.exists(script_path):
        raise FileNotFoundError(script_path)

    def use(filename):
        filepath = os.path.join(os.path.dirname(script_path), filename)
        if os.path.exists(filepath):
            shutil.copy2(filepath, output_dir)

    script_base_name = os.path.basename(script_path)
    use(script_base_name)
    use('requirements.txt')
    use('requirements-base.txt')
    use('requirements-test.txt')
    use('test_api.py')
    use('README.md')

    wsgi_script_file = os.path.join(default_configs_dir, '_wsgi.py.tmpl')
    with open(wsgi_script_file) as f:
        wsgi_script = f.read()
    wsgi_script = wsgi_script.format(
        script=os.path.splitext(script_base_name)[0],
        model_class=model_class
    )
    wsgi_name = os.path.basename(wsgi_script_file).split('.tmpl', 1)[0]
    with open(os.path.join(output_dir, wsgi_name), mode='w') as fout:
        fout.write(wsgi_script)

    model_file = os.path.join(output_dir, script_base_name)
    print(Fore.GREEN + 'Congratulations! ML Backend has been successfully initialized in ' + output_dir)
    print(Fore.RESET + '\nHere are the next steps:')
    print('\n1. Try it out by running:\n' + Fore.CYAN + 'label-studio-ml start ' + output_dir + Fore.RESET +
          '\nYou should be able to connect to it in Label Studio project Settings > Machine Learning > Add Model '
          'and provide with the following URL: http://localhost:9090')

    print('\n2. Go to ' + Fore.CYAN + model_file + Fore.RESET + ' and modify it as you wish:\n' +
          '- predict() - define your prediction logic here\n' + '- fit() - define your training logic here (optional)')

    print('\n3. Deploy your model with docker:\n' + Fore.CYAN + f'cd {output_dir}\ndocker-compose up' + Fore.RESET)

    print('\n4. Have fun! :)')


def start_server(args, subprocess_params):

    project_dir = os.path.join(args.root_dir, args.project_name)
    wsgi = os.path.join(project_dir, '_wsgi.py')

    cmd_args = []
    if args.basic_auth_user and args.basic_auth_pass:
        cmd_args = ["--basic-auth-user", args.basic_auth_user,
                    "--basic-auth-pass", args.basic_auth_pass]
    
    cmd = 'python ' + wsgi + ' ' + ' '.join(cmd_args) + ' ' + ' '.join(subprocess_params)
    
    os.system(cmd)


def deploy_to_gcp(args):
    # create project with
    # create_dir(args)
    # prepare params for gcloud: dir with script, project id, region and service name
    output_dir = os.path.join(args.root_dir, args.project_name)
    project_id = args.gcp_project or os.environ.get("GCP_PROJECT")
    if not project_id:
        raise KeyError("Project id wasn't found in ENV variables!")
    region = args.gcp_region or os.environ.get("GCP_REGION", "us-central1")
    service_name = args.project_name.replace('_', '-')
    # check service name
    # if special_match(service_name):
    #     raise ValueError("Service name in GCP should contain only lower case ASCII letters and hyphen!")
    # check if auth token exists
    auth_token = subprocess.check_output(' '.join(["gcloud", "auth", "print-identity-token"]), shell=True)
    if not auth_token:
        raise PermissionError("You are not authenticated in gcloud! Please run gcloud auth login.")
    # configurate project
    subprocess.check_output(' '.join(["gcloud", "config", "set", "project", project_id]), shell=True)
    # deploy service
    subprocess.check_output(' '.join([
        "gcloud", "run", "deploy",
        service_name,
        "--source", output_dir,
        "--region", region,
        "--update-env-vars", f"LABEL_STUDIO_ML_BACKEND_V2=1,LABEL_STUDIO_HOSTNAME={args.label_studio_host},LABEL_STUDIO_API_KEY={args.label_studio_api_key}"
    ]), input=b"y", shell=True)


def special_match(strg, search=re.compile(r'[^a-z-]').search):
     return bool(search(strg))


def main():
    args, subargs = get_args()

    if args.command in ('init', 'create'):
        create_dir(args)
    elif args.command == 'start':
        start_server(args, subargs)
    elif args.command == 'deploy':
        if args.provider == 'gcp':
            deploy_to_gcp(args)


if __name__ == '__main__':
    main()
