$sw = [Diagnostics.Stopwatch]::StartNew()
$name="training.provision.$(get-random)"
invoke-command -scriptblock {
    Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    choco install docker-for-windows git python3 googlechrome vscode pycharm -y
    $h=hostname
    ipconfig | select-string -pattern "  IPv4 "|format-table @{Expression={$_.Line};Label="$h IPs"}
    choco -v
    docker -v
    git -v
    python3 -v
    pip3 install numpy
    pip3 install matplotlib
    mkdir C:\training
    cd C:\traning
    git clone https://github.com/bartuer/training_notebook.git
    docker pull caapi/ai
    invoke-webrequest https://raw.githubusercontent.com/microsoft/ai-edu/master/C-%E5%BC%80%E5%8F%91%E5%B7%A5%E5%85%B7%E4%B8%8E%E7%8E%AF%E5%A2%83/mpp/run.ps1 C:\training\run.ps1
    Write-Host "configure docker volume, share C folder, then restart machine"
    Write-Host "after rebooting, run.ps1"
} -ThrottleLimit 16 -AsJob -jobname $name
while (get-job -name $name|where-object {$_.state -eq "Running"}){}
receive-job -name $name
remove-job -name $name
$sw.Stop()
$sw.Elapsed