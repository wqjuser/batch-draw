import launch
import os
import pkg_resources
import requests


def get_pypi_package_latest_version(pg_name):
    try:
        response = requests.get(f"https://pypi.org/pypi/{pg_name}/json")
        response.raise_for_status()
        package_info = response.json()
        return package_info["info"]["version"]
    except Exception as error:
        print(f"Error: Failed to get the latest version of {pg_name} from PyPI. {error}")
        return None


req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            package_name = package.split('==')[0] if '==' in package else package
            installed_version = pkg_resources.get_distribution(package_name).version
            latest_version = get_pypi_package_latest_version(package_name)

            if latest_version is not None and installed_version != latest_version:
                launch.run_pip(f"install --upgrade {package_name}",
                               f"batch_draw requirement: changing {package_name} version from {installed_version} to {latest_version}")
            elif not launch.is_installed(package):
                launch.run_pip(f"install --upgrade {package}", f"batch_draw requirement: {package}")
        except Exception as e:
            print(e)
            print(f"Warning: Failed to install {package}, some functions may not work.")
