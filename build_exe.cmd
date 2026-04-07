pyinstaller --noconfirm MangaTranslator.spec

set "PROJECT_ROOT=%~dp0"

@REM source dir
set "SOURCE_DIR=%PROJECT_ROOT%dist\MangaTranslator"

@REM target dir (change this to your desired output path)
set "TARGET_DIR=E:\manga-translator-app"

copy /Y "%SOURCE_DIR%\MangaTranslator.exe" "%TARGET_DIR%\" >nul
robocopy "%SOURCE_DIR%\_internal" "%TARGET_DIR%\_internal" /MIR /NDL /NFL /NJH /NJS