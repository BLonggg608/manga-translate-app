pyinstaller --noconfirm MangaTranslator.spec

set "PROJECT_ROOT=%~dp0"
set "SOURCE_DIR=%PROJECT_ROOT%dist\MangaTranslator"
set "TARGET_DIR=E:\manga-translator-app"

copy /Y "%SOURCE_DIR%\MangaTranslator.exe" "%TARGET_DIR%\" >nul
robocopy "%SOURCE_DIR%\_internal" "%TARGET_DIR%\_internal" /MIR /NDL /NFL /NJH /NJS