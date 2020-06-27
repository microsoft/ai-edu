netsh advfirewall firewall add rule name="Jupyter 8000" dir=in action=allow protocol=TCP localport=8000
try {
    $ErrorActionPreference = "Stop"
    git clone https://github.com/bartuer/training_notebook.git
} catch {
    cd C:\training\training_notebook
    git pull
} finally{
   $ErrorActionPreference = "Continue";
}

Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope LocalMachine
Write-Host -ForegroundColor Yellow -BackgroundColor DarkGreen "after rebooting, cd C:\training; .\run.ps1"