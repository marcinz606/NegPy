!define APPNAME "NegPy"
!define COMPANYNAME "NegPy"
!define DESCRIPTION "Professional Film Negative Processing"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0

!include "MUI2.nsh"

Name "${APPNAME}"
OutFile "dist\NegPy_Setup.exe"
InstallDir "$PROGRAMFILES64\${APPNAME}"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath "$INSTDIR"
    File /r "dist\NegPy\*.*"

    WriteUninstaller "$INSTDIR\uninstall.exe"

    # Shortcuts
    CreateShortcut "$SMPROGRAMS\${APPNAME}.lnk" "$INSTDIR\NegPy.exe" "" "$INSTDIR\media\icons\icon.ico"
    CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\NegPy.exe" "" "$INSTDIR\media\icons\icon.ico"
SectionEnd

Section "Uninstall"
    Delete "$SMPROGRAMS\${APPNAME}.lnk"
    Delete "$DESKTOP\${APPNAME}.lnk"
    RMDir /r "$INSTDIR"
SectionEnd
