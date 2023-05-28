import launch
import os
import pkg_resources

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
# copy from controlnet, thanks
with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name, package_version = package.split('==')
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != package_version:
                    launch.run_pip(f"install {package}",
                                   f"batch_draw requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif not launch.is_installed(package):
                launch.run_pip(f"install {package}", f"batch_draw requirement: {package}")
        except Exception as e:
            print(e)
            print(f"Warning: Failed to install {package}, some functions may not work.")
