REM # Copyright (c) Microsoft. All rights reserved.
REM # Licensed under the MIT license. See LICENSE file in the project root for full license information.

set user=%1
set repo=%2
set git_data_path=%3
set scriptdir="%~dp0"
echo %git_data_path%
mkdir "%git_data_path%"
cd "%git_data_path%"
mkdir %user%
cd %user%
rd /s /q %repo%
mkdir %repo%
cd %repo%
git init
set url=https://github.com/%user%/%repo%.git
git remote add origin %url%
git config core.sparsecheckout true
copy "%scriptdir%\sparse-checkout" .git\info\sparse-checkout
git pull --depth=1 origin master
for /f %%i in ('git rev-parse HEAD') do set commitid=%%i
rd /s /q .git
cd ..
move %repo% %repo%_%commitid%
mkdir %repo%
move %repo%_%commitid% %repo%\%commitid%

