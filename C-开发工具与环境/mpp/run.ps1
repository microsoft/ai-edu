docker stop -t 0 ai
docker rm ai
docker run --name ai -d  -p 8000:9999  -v C:\training\training_notebook:/opt/notebook caapi/ai
$url = 'http://localhost:8000'
try {
    $chrome = (Get-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe').'(Default)'
    Start-Process "$chrome" $url
}
catch {
    Start-Process $url
}