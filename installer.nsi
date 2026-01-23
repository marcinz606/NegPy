!define APPNAME "NegPy"
!define COMPANYNAME "NegPy"
!define DESCRIPTION "Professional Film Negative Processing"

!ifndef VERSION
    !define VERSION "1.0.0"
!endif

!ifndef OUTFILE
    !define OUTFILE "NegPy_Setup.exe"
!endif

!include "MUI2.nsh"

!define MUI_ICON "media\icons\icon.ico"
!define MUI_UNICON "media\icons\icon.ico"

Name "${APPNAME}"
OutFile "dist\${OUTFILE}"
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

    # Registry info for Add/Remove Programs
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayIcon" "$\"$INSTDIR\NegPy.exe$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${COMPANYNAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSION}"

    # Shortcuts
    CreateShortcut "$SMPROGRAMS\${APPNAME}.lnk" "$INSTDIR\NegPy.exe" "" "$INSTDIR\NegPy.exe" 0
    CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\NegPy.exe" "" "$INSTDIR\NegPy.exe" 0
SectionEnd

Section "Uninstall"
    Delete "$SMPROGRAMS\${APPNAME}.lnk"
    Delete "$DESKTOP\${APPNAME}.lnk"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
    RMDir /r "$INSTDIR"
SectionEnd
