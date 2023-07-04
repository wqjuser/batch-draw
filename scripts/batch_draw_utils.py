import os
import hashlib
import psutil
import platform
import uuid
import subprocess
import distro
import time
import datetime


def get_formatted_current_time():
    current_date = datetime.datetime.now().date()
    formatted_date = f"{current_date.month}月{current_date.day}日"
    current_time = datetime.datetime.now().time()
    formatted_time = f"{current_time.hour}_{current_time.minute}"
    return formatted_date + '_ai_' + formatted_time


def get_cpu_info():
    cpu_info = ""
    try:
        cpu_info = str(psutil.cpu_percent())
    except Exception as e:
        print("Error getting CPU info:", e)

    return cpu_info


def get_device_id():
    device_id = ""
    try:
        device_id = uuid.uuid1()
    except Exception as e:
        print("Error getting device id:", e)
    return device_id


def get_mac_address():
    mac_address_formatted = ""
    try:
        mac_address = uuid.getnode()
        mac_address_formatted = ':'.join(['{:02x}'.format((mac_address >> ele) & 0xff) for ele in range(0, 8 * 6, 8)][::-1])
    except Exception as e:
        print("Error getting mac address", e)
    return mac_address_formatted


def get_os():
    system_type = ""
    try:
        system_type = platform.system()
        if system_type == "Darwin":
            system_type = "MacOS"
    except Exception as e:
        print("Error getting system type", e)
    return system_type.lower()


def get_hard_disk_id():
    hard_disk_id = ""
    try:
        system_type = platform.system()
        if system_type == "Windows":
            cmd = "wmic diskdrive get SerialNumber"
            output = subprocess.check_output(cmd, shell=True).decode().split('\n')[1].strip()
            return output
        elif system_type == "Linux":
            cmd = "sudo hdparm -I /dev/sda | grep Serial"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            return output.split()[-1]
        elif system_type == "Darwin":
            cmd = "ioreg -r -c AppleAHCIDiskDriver -l | grep SerialNumber | awk '{print $4}'"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            return output.replace('"', '')
        else:
            print("不支持的操作系统")
            return hard_disk_id
    except Exception as e:
        print("Error getting hard_disk_id", e)
    return hard_disk_id


def get_os_version():
    os_version = ''
    try:
        system_type = platform.system()
        if system_type == "Windows":
            os_version = platform.win32_ver()
        elif system_type == "Linux":
            os_version = distro.info()
        elif system_type == "Darwin":
            os_version = platform.mac_ver()
        else:
            print("not support os")
    except Exception as e:
        print("Error getting os version", e)
    return f"{os_version}"


def generate_draft_id(is_upper: bool):
    new_uuid = uuid.uuid4()
    return str(new_uuid).upper() if is_upper else str(new_uuid)


def get_timestamp():
    timestamp = time.time()
    microsecond_timestamp = int(timestamp * 1e6)
    return microsecond_timestamp


def get_disk_serial_windows():
    serial = ""
    try:
        import wmi
        c = wmi.WMI()
        for disk in c.Win32_DiskDrive():
            serial = disk.SerialNumber.strip()
            break
    except Exception as e:
        print("Error getting disk serial on Windows:", e)

    return serial


def get_disk_serial_linux():
    serial = ""
    try:
        with os.popen('hdparm -I /dev/sda | grep Serial') as disk_serial:
            serial = disk_serial.read().strip().split()[-1]
    except Exception as e:
        print("Error getting disk serial on Linux:", e)

    return serial


def get_disk_serial_macos():
    serial = ""
    try:
        with os.popen('ioreg -c IOPlatformExpertDevice | grep IOPlatformSerialNumber') as disk_serial:
            serial = disk_serial.read().strip().split()[-1].replace('"', '')
    except Exception as e:
        print("Error getting disk serial on macOS:", e)

    return serial


def get_install_date_windows():
    install_date = ""
    try:
        import wmi
        c = wmi.WMI()
        for os_info in c.Win32_OperatingSystem():
            install_date = os_info.InstallDate
            break
    except Exception as e:
        print("Error getting OS install date on Windows:", e)

    return install_date


def get_install_date_linux():
    install_date = ""
    try:
        with os.popen('sudo dumpe2fs $(mount | grep "on / " | awk \'{print $1}\') 2>/dev/null | grep "Filesystem created"') as fs_date:
            install_date = fs_date.read().strip().split()[-3]
    except Exception as e:
        print("Error getting OS install date on Linux:", e)

    return install_date


def get_install_date_macos():
    install_date = ""
    try:
        with os.popen('sysctl kern.installationtime') as install_time:
            install_date = install_time.read().strip().split()[-1]
    except Exception as e:
        print("Error getting OS install date on macOS:", e)

    return install_date


def hash_machine_code(data):
    m = hashlib.sha256()
    m.update(data.encode("utf-8"))
    return m.hexdigest()


def get_unique_machine_code():
    os_version = get_os_version()

    if platform.system() == "Windows":
        disk_serial = get_disk_serial_windows()
        install_date = get_install_date_windows()
    elif platform.system() == "Linux":
        disk_serial = get_disk_serial_linux()
        install_date = get_install_date_linux()
    elif platform.system() == "Darwin":
        disk_serial = get_disk_serial_macos()
        install_date = get_install_date_macos()
    else:
        print("Unsupported platform:", platform.system())
        return None

    combined = disk_serial + install_date + os_version
    machine_code = hash_machine_code(combined)

    return machine_code
