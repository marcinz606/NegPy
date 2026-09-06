"""Native Windows folder controls that do not initialize Qt."""

import ctypes as ct
from ctypes import wintypes as wt
from pathlib import Path


_TaskCallback = ct.WINFUNCTYPE(ct.c_long, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM, wt.LPARAM)
_BrowseCallback = ct.WINFUNCTYPE(ct.c_int, wt.HWND, wt.UINT, wt.LPARAM, wt.LPARAM)


class _Button(ct.Structure):
    _pack_ = 1
    _fields_ = [("id", ct.c_int), ("text", wt.LPCWSTR)]


class _TaskConfig(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("size", wt.UINT),
        ("parent", wt.HWND),
        ("instance", wt.HINSTANCE),
        ("flags", wt.UINT),
        ("common_buttons", wt.UINT),
        ("title", wt.LPCWSTR),
        ("icon", ct.c_void_p),
        ("instruction", wt.LPCWSTR),
        ("content", wt.LPCWSTR),
        ("button_count", wt.UINT),
        ("buttons", ct.POINTER(_Button)),
        ("default_button", ct.c_int),
        ("radio_count", wt.UINT),
        ("radios", ct.POINTER(_Button)),
        ("default_radio", ct.c_int),
        ("verification", wt.LPCWSTR),
        ("expanded", wt.LPCWSTR),
        ("expanded_label", wt.LPCWSTR),
        ("collapsed_label", wt.LPCWSTR),
        ("footer_icon", ct.c_void_p),
        ("footer", wt.LPCWSTR),
        ("callback", _TaskCallback),
        ("callback_data", wt.LPARAM),
        ("width", wt.UINT),
    ]


class _BrowseInfo(ct.Structure):
    _fields_ = [
        ("owner", wt.HWND),
        ("root", ct.c_void_p),
        ("display", wt.LPWSTR),
        ("title", wt.LPCWSTR),
        ("flags", wt.UINT),
        ("callback", _BrowseCallback),
        ("parameter", wt.LPARAM),
        ("image", ct.c_int),
    ]


def _send(hwnd, message, parameter, text):
    send = ct.windll.user32.SendMessageW
    send.argtypes = (wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
    send.restype = wt.LPARAM
    buffer = ct.create_unicode_buffer(text)
    return send(hwnd, message, parameter, ct.addressof(buffer))


def _browse_directory(owner, initial: Path) -> Path | None:
    while not initial.is_dir() and initial != initial.parent:
        initial = initial.parent
    shell = ct.windll.shell32
    ole = ct.windll.ole32
    ole.CoInitializeEx.argtypes = (ct.c_void_p, wt.DWORD)
    ole.CoInitializeEx.restype = ct.c_long
    status = ole.CoInitializeEx(None, 2)  # Apartment-threaded COM for the shell folder picker.
    if status < 0:
        raise OSError(f"Cannot initialize the folder picker (0x{status & 0xFFFFFFFF:08X}).")
    try:

        @_BrowseCallback
        def initialized(hwnd, notification, _parameter, _data):
            if notification == 1:  # BFFM_INITIALIZED
                _send(hwnd, 0x467, 1, str(initial))  # BFFM_SETSELECTIONW
            return 0

        display = ct.create_unicode_buffer(260)
        config = _BrowseInfo(
            owner=owner, display=ct.cast(display, wt.LPWSTR), title="Select the folder for NegPy data", flags=0x51, callback=initialized
        )
        shell.SHBrowseForFolderW.argtypes = (ct.POINTER(_BrowseInfo),)
        shell.SHBrowseForFolderW.restype = ct.c_void_p
        item = shell.SHBrowseForFolderW(ct.byref(config))
        if not item:
            return None
        try:
            buffer = ct.create_unicode_buffer(32768)
            shell.SHGetPathFromIDListEx.argtypes = (ct.c_void_p, wt.LPWSTR, wt.DWORD, wt.DWORD)
            shell.SHGetPathFromIDListEx.restype = wt.BOOL
            if not shell.SHGetPathFromIDListEx(item, buffer, len(buffer), 0):
                raise OSError("Select a filesystem folder for NegPy data.")
            return Path(buffer.value)
        finally:
            ole.CoTaskMemFree.argtypes = (ct.c_void_p,)
            ole.CoTaskMemFree(item)
    finally:
        ole.CoUninitialize()


def _display_path(path: Path) -> str:
    # SysLink uses mnemonic ampersands, not HTML entities. Other markup characters are invalid in Windows filenames.
    return str(path).replace("&", "&&").replace("<", "‹").replace(">", "›").replace('"', "″")


def _content(source: Path, selected: Path) -> str:
    return (
        f"NegPy cannot write to:\n{_display_path(source)}\n\n"
        "Windows folder protection or access permissions may block this folder.\n\n"
        f'Use this data folder (click the path to change it):\n<a href="folder">{_display_path(selected)}</a>\n\n'
        "No files are copied or moved. A new folder starts with new settings and no saved edits. "
        "If you select an existing NegPy data folder, its data will be used. "
        "Your original data stays in the old folder. NegPy remembers your choice."
    )


def choose_directory(source: Path, suggested: Path) -> Path | None:
    from negpy.desktop.startup import _message

    selected = suggested

    @_TaskCallback
    def callback(hwnd, notification, _button, parameter, _data):
        nonlocal selected
        if notification != 3:  # TDN_HYPERLINK_CLICKED
            return 0
        try:
            link = ct.wstring_at(parameter)
            if link == "folder":
                choice = _browse_directory(hwnd, selected)
                if choice is not None:
                    selected = choice
                    _send(hwnd, 0x46C, 0, _content(source, selected))  # TDM_SET_ELEMENT_TEXT / TDE_CONTENT
            elif link == "help":
                _message(
                    "To keep using the original folder, allow the trusted NegPy.exe in Windows Security:\n\n"
                    "Virus & threat protection > Manage ransomware protection > "
                    "Allow an app through Controlled folder access > Add an allowed app.\n\n"
                    "Select the NegPy.exe you run, then cancel this dialog and start NegPy again. "
                    "Keep folder protection enabled. Do not add an antivirus exclusion. "
                    "If Windows is managed by your organization, contact your administrator.",
                    0x40,
                )
        except Exception as error:
            # Exceptions must not escape a ctypes callback into the native message loop.
            _message(f"NegPy could not complete this action:\n\n{error}", 0x10)
        return 0

    buttons = (_Button * 1)(_Button(100, "Use This Folder"))
    config = _TaskConfig(
        size=ct.sizeof(_TaskConfig),
        flags=0x9,
        common_buttons=0x8,
        title="NegPy data folder",
        instruction="Choose a writable data folder",
        content=_content(source, selected),
        button_count=1,
        buttons=buttons,
        default_button=2,
        footer='<a href="help">Keep using the original folder: Windows protection help</a>',
        callback=callback,
        width=360,
    )
    try:
        show = ct.windll.comctl32.TaskDialogIndirect
    except AttributeError as error:
        raise OSError("Windows could not load the data-folder dialog. Set NEGPY_USER_DIR to a writable folder.") from error
    show.argtypes = (ct.POINTER(_TaskConfig), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(wt.BOOL))
    show.restype = ct.c_long
    button = ct.c_int()
    result = show(ct.byref(config), ct.byref(button), None, None)
    if result < 0:
        raise OSError(f"Cannot show the data-folder dialog (0x{result & 0xFFFFFFFF:08X}).")
    return selected if button.value == 100 else None
