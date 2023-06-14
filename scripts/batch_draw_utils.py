import os
import hashlib
import psutil
import platform


def get_cpu_info():
    cpu_info = ""
    try:
        cpu_info = str(psutil.cpu_percent())
    except Exception as e:
        print("Error getting CPU info:", e)

    return cpu_info


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


def get_os_version():
    os_version = platform.platform()
    return os_version


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
